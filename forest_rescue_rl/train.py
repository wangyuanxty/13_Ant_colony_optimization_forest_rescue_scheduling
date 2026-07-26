"""
REINFORCE training. Baseline = per-episode mean return.
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
    parts.append(torch.tensor([c for row in fg for c in row], dtype=torch.float32))
    parts.append(env.agent_pos.clone())
    coords = torch.cat(parts, dim=0).float() / env.world_size
    return coords, {'patrol': (1, 1+n_p), 'fire_grid': (1+n_p, 1+n_p+n_g),
                    'agents': (1+n_p+n_g, 1+n_p+n_g+env.n_agents)}

def _e2c(env_idx, n_p, gs, off):
    if env_idx < 0: return 0
    if env_idx < n_p: return off['patrol'][0] + env_idx
    return off['fire_grid'][0] + (env_idx - n_p)

def _c2e(coord_idx, n_p, off):
    if coord_idx == 0: return -1
    ps, pe = off['patrol']; fs, fe = off['fire_grid']
    if ps <= coord_idx < pe: return coord_idx - ps
    if fs <= coord_idx < fe: return n_p + (coord_idx - fs)
    return -1

def train(n_episodes=1000, log_interval=50):
    device = torch.device('cuda')
    print(f"REINFORCE | device={device}")

    env = ForestRescueEnv(device=device)
    policy = RescuePolicy(device=device)
    best_loss = float('inf')
    hist = {'p_loss': [], 'ext': [], 'cov': [], 'dam': [], 'dist': []}
    start_ep = 0

    ckpt_path = os.path.join(BASE, "forest_rescue_rl", "ckpt_rf.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        policy.net.load_state_dict(ckpt['net'])
        policy.decoder.load_state_dict(ckpt['decoder'])
        policy.opt.load_state_dict(ckpt['opt'])
        best_loss = ckpt['best_loss']
        hist = ckpt['hist']; start_ep = ckpt['ep'] + 1
        print(f"Resumed from ep {start_ep}"); del ckpt

    for ep in range(start_ep, n_episodes):
        state = env.reset(n_patrol=random.randint(15,30), n_drones=random.randint(1,1),
                          n_helis=random.randint(1,2), n_ground=random.randint(1,1),
                          fire_prob=0.05, spread_prob=0.5)
        log_probs, rewards = [], []
        for _ in range(env.max_t):
            coords, off = build_coords_fixed(env)
            n_all = coords.shape[0]; h_enc = policy.encode(coords)
            avail = env.available_actions(); taken = set()
            order = list(range(env.n_agents)); random.shuffle(order)
            actions = {}
            for ai in order:
                if env.agent_busy[ai]: continue
                at = env.agent_types[ai]
                cand = env.filter_in_range(ai, avail[at])
                mask = torch.zeros(n_all, dtype=torch.bool, device=h_enc.device)
                for c in cand:
                    ci = _e2c(c, env.n_patrol, env.grid_size, off)
                    if ci >= 0 and ci not in taken: mask[ci] = True
                if mask.sum() == 0: continue
                pid = off['agents'][0] + ai
                tc, lp = policy.act(h_enc, pid, policy.TYPE[at], mask)
                log_probs.append(lp)
                taken.add(tc)
                actions[ai] = _c2e(tc, env.n_patrol, off)
            state, reward, done, info = env.step(actions)
            rewards.append(reward)
            if done: break

        if not log_probs: continue
        # Per-decision returns from per-timestep rewards
        rets = []; R = 0.0
        for r in reversed(rewards): R = r + 0.99 * R; rets.insert(0, R)
        n_t, n_d = len(rets), len(log_probs)
        dec_rets = [rets[min(i * n_t // n_d, n_t - 1)] for i in range(n_d)]

        rets_t = torch.tensor(dec_rets, device=device).float()
        lps_t = torch.stack(log_probs)
        adv = rets_t - rets_t.mean()  # baseline = mean of this episode's returns
        p_loss = -(lps_t * adv).mean()
        policy.opt.zero_grad(); p_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.decoder.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 10.0)
        policy.opt.step()

        hist['p_loss'].append(p_loss.item()); hist['ext'].append(info['extinguished'])
        hist['cov'].append(info['covered']); hist['dam'].append(info['damage'])
        hist['dist'].append(info['dist'])

        if p_loss.item() < best_loss:
            best_loss = p_loss.item()
            torch.save(policy.decoder.state_dict(), os.path.join(BASE, "forest_rescue_rl", "decoder_best.pth"))
        if ep % 50 == 0:
            torch.save({'net': policy.net.state_dict(),
                        'decoder': policy.decoder.state_dict(),
                        'opt': policy.opt.state_dict(),
                        'best_loss': best_loss, 'hist': hist, 'ep': ep}, ckpt_path)
        torch.cuda.empty_cache()
        if ep % log_interval == 0:
            n = len(hist['p_loss'])
            a = lambda k, w: sum(hist[k][-w:]) / min(w, n)
            print(f"ep {ep:5d} | p={a('p_loss',log_interval):.3f} | "
                  f"ext={a('ext',log_interval):.0f}")

    torch.save(policy.decoder.state_dict(), os.path.join(BASE, "forest_rescue_rl", "decoder_final.pth"))
    torch.save(hist, os.path.join(BASE, "forest_rescue_rl", "train_history.pt"))
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        X = range(len(hist['p_loss']))
        def smooth(y, w=50):
            if len(y) < w: return y
            return np.convolve(y, np.ones(w)/w, mode='valid')
        for ax, key, color, title in [
            (axes[0,0], 'p_loss', 'blue', 'Policy Loss'),
            (axes[0,1], 'ext', 'green', 'Fires Extinguished'),
            (axes[1,0], 'dam', 'purple', 'Fire Damage'),
            (axes[1,1], 'dist', 'brown', 'Flight Distance'),
        ]:
            y = hist[key]
            if len(y) > 0: ax.plot(X, y, alpha=0.3, color=color); ax.plot(smooth(y), color=color, linewidth=2)
            ax.set_title(title); ax.set_xlabel('Episode')
        plt.tight_layout(); plt.savefig(os.path.join(BASE, "forest_rescue_rl", "training_curves.png"), dpi=100); plt.close()
    except: pass
    print(f"Done! {n_episodes} episodes, best p_loss={best_loss:.3f}")

if __name__ == '__main__':
    train()
