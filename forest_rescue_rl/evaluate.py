"""
Evaluate trained policy vs baselines (Random, Nearest) on fixed test scenarios.
Metrics: fires extinguished, patrol coverage, flight distance, fire damage.
"""
import sys, os, time, random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forest_rescue_rl.env import ForestRescueEnv
from forest_rescue_rl.model import RescuePolicy
from forest_rescue_rl.train import build_coords_fixed, _env2coord, _coord2env

SEED = 42; N_EP = 10

def run_eval(env, policy_fn):
    m = {'ext': [], 'cov': [], 'dist': [], 'dam': [], 'time': []}
    for ep in range(N_EP):
        np.random.seed(SEED + ep); random.seed(SEED + ep)
        torch.manual_seed(SEED + ep)
        state = env.reset(n_patrol=40, n_drones=3, n_helis=2, n_ground=3,
                          fire_prob=0.05, spread_prob=0.5)
        t0 = time.time()
        for _ in range(env.max_t):
            actions = policy_fn(state, env)
            state, reward, done, info = env.step(actions)
            if done: break
        m['ext'].append(info['extinguished']); m['cov'].append(info['covered'])
        m['dist'].append(info['dist']); m['dam'].append(info['damage'])
        m['time'].append(time.time() - t0)
    return {k: (np.mean(v), np.std(v)) for k, v in m.items()}


def random_policy(state, env):
    avail = env.available_actions(); acts = {}
    for ai in range(env.n_agents):
        if env.agent_busy[ai]: continue
        cand = env.filter_in_range(ai, avail[env.agent_types[ai]])
        if not cand: continue  # idle if no valid target
        acts[ai] = random.choice(cand)
    return acts


def nearest_policy(state, env):
    avail = env.available_actions(); acts = {}; taken = set()
    for ai in range(env.n_agents):
        if env.agent_busy[ai]: continue
        at = env.agent_types[ai]
        cand = env.filter_in_range(ai, [c for c in avail[at] if c not in taken])
        if not cand: continue
        pos = env.agent_pos[ai]
        best = min(cand, key=lambda c: float(torch.norm(env._point_pos(c) - pos)))
        taken.add(best)
        acts[ai] = best
    return acts


def make_trained_fn(policy, device):
    def fn(state, env):
        coords, off = build_coords_fixed(env)
        h_enc = policy.encode(coords)
        avail = env.available_actions(); taken = set(); acts = {}
        n_all = coords.shape[0]
        for ai in range(env.n_agents):
            if env.agent_busy[ai]: continue
            at = env.agent_types[ai]; cand = env.filter_in_range(ai, avail[at])
            mask = torch.zeros(n_all, dtype=torch.bool, device=h_enc.device)
            for c in cand:
                ci = _env2coord(c, env.n_patrol, env.grid_size, off)
                if ci >= 0 and ci not in taken: mask[ci] = True
            pid = off['agents'][0] + ai
            if mask.sum() == 0: continue
            ci = policy.act_greedy(h_enc, pid, policy.TYPE[at], mask)
            if ci != 0: taken.add(ci)
            acts[ai] = _coord2env(ci, env.n_patrol, off)
        return acts
    return fn


if __name__ == '__main__':
    device = torch.device('cuda')
    env = ForestRescueEnv()

    print("Baselines (10 test episodes each):")
    r = run_eval(env, random_policy)
    print(f"  Random:  dam={r['dam'][0]:.0f}±{r['dam'][1]:.0f} "
          f"dist={r['dist'][0]:.0f}")

    n = run_eval(env, nearest_policy)
    print(f"  Nearest: dam={n['dam'][0]:.0f}±{n['dam'][1]:.0f} "
          f"dist={n['dist'][0]:.0f}")

    best_path = os.path.join(os.path.dirname(__file__), "decoder_best.pth")
    final_path = os.path.join(os.path.dirname(__file__), "decoder_final.pth")
    ckpt_path = best_path if os.path.exists(best_path) else final_path
    if os.path.exists(ckpt_path):
        print("\nTrained policy:")
        pol = RescuePolicy(device=device)
        pol.decoder.load_state_dict(torch.load(ckpt_path, map_location=device))
        t = run_eval(env, make_trained_fn(pol, device))
        print(f"  Trained: dam={t['dam'][0]:.0f}±{t['dam'][1]:.0f} "
              f"dist={t['dist'][0]:.0f}")
        print(f"  vs Random: dam={t['dam'][0]/r['dam'][0]*100:.0f}% "
              f"dist={t['dist'][0]/r['dist'][0]*100:.0f}%")
        print(f"  vs Nearest: dam={t['dam'][0]/n['dam'][0]*100:.0f}% "
              f"dist={t['dist'][0]/n['dist'][0]*100:.0f}%")
    else:
        print("\nNo trained checkpoint yet. Train first.")
