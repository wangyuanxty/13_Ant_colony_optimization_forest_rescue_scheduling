"""
A2C training: per-step TD bootstrapping, batch update every K steps.
"""
import sys, os, time, random
import torch
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE)
from forest_rescue_rl.env import ForestRescueEnv
from forest_rescue_rl.model import RescuePolicy

def build_coords_fixed(env):
    n_p = env.n_patrol; gs = env.grid_size; n_g = gs * gs
    parts = [torch.zeros(1, 2)]
    parts.append(env.patrol_pts)
    fg = [[env._coord(gx,gy) for gy in range(gs)] for gx in range(gs)]
    fg_flat = [c for row in fg for c in row]
    parts.append(torch.tensor(fg_flat, dtype=torch.float32))
    parts.append(env.agent_pos.clone())
    coords = torch.cat(parts, dim=0).float()
    off = {'patrol': (1, 1+n_p), 'fire_grid': (1+n_p, 1+n_p+n_g),
           'agents': (1+n_p+n_g, 1+n_p+n_g+env.n_agents)}
    return coords, off

def _env2coord(env_idx, n_p, gs, off):
    if env_idx < 0: return 0
    if env_idx < n_p: return off['patrol'][0] + env_idx
    return off['fire_grid'][0] + (env_idx - n_p)

def _coord2env(coord_idx, n_p, off):
    if coord_idx == 0: return -1
    ps, pe = off['patrol']
    if ps <= coord_idx < pe: return coord_idx - ps
    fs, fe = off['fire_grid']
    if fs <= coord_idx < fe: return n_p + (coord_idx - fs)
    return -1

def train_a2c(n_episodes=500, log_interval=20, update_steps=10):
    device = torch.device('cuda')
    print(f"A2C | device={device} | update_every={update_steps}")

    env = ForestRescueEnv(device=device)
    policy = RescuePolicy(device=device)
    D = policy.dim
    critic = torch.nn.Sequential(
        torch.nn.Linear(D, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
    ).to(device)
    c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    gamma = 0.99
    v_stats = {'n': 0, 'mean': 0.0, 'std': 100.0}
    hist = {'p_loss': [], 'v_loss': [], 'ext': [], 'cov': [], 'dam': [], 'dist': []}

    for ep in range(n_episodes):
        state = env.reset(n_patrol=random.randint(20,50),
                          n_drones=random.randint(2,4),
                          n_helis=random.randint(2,3),
                          n_ground=random.randint(2,3))
        buf = {'lps': [], 'advs': [], 'v_targets': [], 'vals': []}
        ep_reward = 0.0

        for t in range(env.max_t):
            coords, off = build_coords_fixed(env)
            n_all = coords.shape[0]
            h_enc = policy.encode(coords)
            avail = env.available_actions(); taken = set()
            order = list(range(env.n_agents)); random.shuffle(order)
            actions = {}

            for ai in order:
                if env.agent_busy[ai]: continue
                at = env.agent_types[ai]
                cand = env.filter_in_range(ai, avail[at])
                mask = torch.zeros(n_all, dtype=torch.bool, device=h_enc.device)
                for c in cand:
                    ci = _env2coord(c, env.n_patrol, env.grid_size, off)
                    if ci >= 0 and ci not in taken: mask[ci] = True
                pid = off['agents'][0] + ai
                if mask.sum() <= 1 and mask[0].item():
                    target_ci = 0
                else:
                    target_ci, lp = policy.act(h_enc, pid, policy.TYPE[at], mask)
                    v = critic(h_enc[0, pid, :])
                    buf['lps'].append(lp); buf['vals'].append(v)
                if target_ci != 0: taken.add(target_ci)
                actions[ai] = _coord2env(target_ci, env.n_patrol, off)

            next_state, reward, done, info = env.step(actions)
            ep_reward += reward

            # TD target for next state
            if not done:
                coords2, off2 = build_coords_fixed(env)
                h_nxt = policy.encode(coords2)
                v_nxt = torch.mean(torch.stack(
                    [critic(h_nxt[0, off2['agents'][0]+ai, :])
                     for ai in range(env.n_agents)])).item()
            else:
                v_nxt = 0.0

            # TD advantage for this step's decisions
            v_target = reward + gamma * v_nxt
            for i in range(len(buf['vals']) - len(buf['v_targets'])):
                buf['v_targets'].append(v_target)

            # Flush buffer
            n_buf = len(buf['lps'])
            if n_buf >= update_steps * 3 or (done and n_buf > 0):
                _optimize(policy, critic, c_opt, buf, hist, v_stats)
                buf = {'lps': [], 'advs': [], 'v_targets': [], 'vals': []}

            state = next_state
            if done: break

        # Episode-end flush
        # Compute remaining TD targets
        remaining = len(buf['lps']) - len(buf['v_targets'])
        if remaining > 0:
            buf['v_targets'].extend([0.0] * remaining)  # done
        if len(buf['lps']) > 0:
            _optimize(policy, critic, c_opt, buf, hist, v_stats)

        torch.cuda.empty_cache()
        hist['ext'].append(info['extinguished']); hist['cov'].append(info['covered'])
        hist['dam'].append(info['damage']); hist['dist'].append(info['dist'])

        if ep % log_interval == 0:
            n = len(hist['p_loss'])
            a = lambda k, w: sum(hist[k][-w:]) / min(w, n) if n > 0 else 0
            print(f"ep {ep:5d} | p={a('p_loss',log_interval):.3f} "
                  f"v={a('v_loss',log_interval):.3f} | "
                  f"ext={a('ext',log_interval):.0f} cov={a('cov',log_interval):.0f}")

    torch.save(policy.net.state_dict(), os.path.join(BASE, "forest_rescue_rl", "net_final.pth"))
    torch.save(policy.decoder.state_dict(), os.path.join(BASE, "forest_rescue_rl", "decoder_final.pth"))
    torch.save(hist, os.path.join(BASE, "forest_rescue_rl", "train_history.pt"))
    _plot(hist, BASE)
    print("Done!")

def _optimize(policy, critic, c_opt, buf, hist, v_stats={'n': 0, 'mean': 0.0, 'std': 100.0}):
    if len(buf['lps']) == 0: return
    lps_t = torch.stack(buf['lps'])
    v_res = torch.tensor(buf['v_targets'], device=policy.device).float()
    vals_t = torch.stack(buf['vals']).squeeze(-1).float()

    # Update running stats for V targets
    n = v_stats['n']; new_n = n + v_res.numel()
    new_mean = (n * v_stats['mean'] + v_res.sum().item()) / new_n
    batch_mean = v_res.mean().item()
    # Welford for std
    delta = batch_mean - v_stats['mean']
    v_stats['mean'] = (n * v_stats['mean'] + v_res.numel() * batch_mean) / new_n
    vs = v_res.std().item() if v_res.numel() > 1 else 1.0
    v_stats['std'] = max(vs + 1e-8, v_stats['std'] * 0.999)
    v_stats['n'] = new_n

    # Normalize V targets using running stats
    v_std = v_stats['std'] if v_stats['std'] > 0 else 1.0
    v_mean = v_stats['mean']
    v_res_norm = (v_res - v_mean) / v_std
    vals_norm = (vals_t - v_mean) / v_std

    # Advantage: raw scale, then normalize for gradient
    adv_t = v_res - vals_t.detach()
    as_ = adv_t.std() if adv_t.numel() > 1 else torch.tensor(1.0, device=adv_t.device)
    adv_t = (adv_t - adv_t.mean()) / (as_ + 1e-8)

    p_loss = -(lps_t * adv_t).mean()
    v_loss = (v_res_norm - vals_norm).pow(2).mean()

    policy.opt.zero_grad(); c_opt.zero_grad()
    (p_loss + v_loss).backward()
    torch.nn.utils.clip_grad_norm_(policy.decoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    policy.opt.step(); c_opt.step()

    hist['p_loss'].append(p_loss.item())
    hist['v_loss'].append(v_loss.item())

def _plot(hist, base):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        X = range(len(hist['p_loss']))
        def smooth(y, w=50):
            if len(y) < w: return y
            return np.convolve(y, np.ones(w)/w, mode='valid')
        for ax, key, color, title in [
            (axes[0,0], 'p_loss', 'blue', 'Policy Loss'),
            (axes[0,1], 'v_loss', 'orange', 'Value Loss (TD MSE)'),
            (axes[0,2], 'ext', 'green', 'Fires Extinguished'),
            (axes[1,0], 'cov', 'red', 'Patrol Coverage'),
            (axes[1,1], 'dam', 'purple', 'Fire Damage'),
            (axes[1,2], 'dist', 'brown', 'Flight Distance'),
        ]:
            y = hist[key]
            if len(y) > 0:
                ax.plot(X, y, alpha=0.3, color=color)
                ax.plot(smooth(y), color=color, linewidth=2)
            ax.set_title(title); ax.set_xlabel('Update Step')
        plt.tight_layout()
        plt.savefig(os.path.join(base, "forest_rescue_rl", "training_curves.png"), dpi=100)
        plt.close()
        print("Plot saved.")
    except Exception as e:
        print(f"Plot: {e}")

if __name__ == '__main__':
    train_a2c()
