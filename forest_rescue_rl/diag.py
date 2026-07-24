"""Diagnostic: bypass Decoder to verify env task chain."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forest_rescue_rl.env import ForestRescueEnv

env = ForestRescueEnv()
# Turn on debug in env
import forest_rescue_rl.env as env_m
setattr(env_m, '_ON_ARRIVAL_DEBUG', True)
s = env.reset(n_patrol=5, n_drones=1, n_helis=1, n_ground=1,
              fire_prob=0.1, spread_prob=0.3)
ext = 0
for t in range(200):
    a = env.available_actions()
    acts = {}
    for ai in range(env.n_agents):
        if env.agent_busy[ai]:
            continue
        at = env.agent_types[ai]
        cand = [c for c in a[at] if c >= 0]
        acts[ai] = cand[0] if cand else -1
    s, r, done, info = env.step(acts)
    if info['extinguished'] != ext or t % 20 == 0:
        a = env.available_actions()
        print(f"t={t:3d} ext={info['extinguished']} cov={info['covered']} "
              f"drone_a={len(a['drone'])} heli_a={len(a['heli'])} "
              f"gnd_a={len(a['ground'])} tpt={env.transport_done.sum().item()} "
              f"burning={env.fire_grid.sum().item():.0f}")
        ext = info['extinguished']
    if done:
        break
print(f"Final: ext={info['extinguished']} cov={info['covered']}")
