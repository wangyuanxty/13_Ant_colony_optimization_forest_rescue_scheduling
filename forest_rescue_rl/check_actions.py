"""Check what actions the trained model actually picks."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forest_rescue_rl.env import ForestRescueEnv
from forest_rescue_rl.model import RescuePolicy
from forest_rescue_rl.train import build_coords_fixed, _env2coord, _coord2env

device = torch.device('cuda')
env = ForestRescueEnv()
env.reset(n_patrol=40, n_drones=3, n_helis=2, n_ground=3, fire_prob=0.005, spread_prob=0.1)
pol = RescuePolicy(device=device)
bp = os.path.join(os.path.dirname(__file__), 'decoder_best.pth')
if os.path.exists(bp):
    pol.decoder.load_state_dict(torch.load(bp, map_location=device))

counts = {'drone': 0, 'heli': 0, 'ground': 0, 'skip': 0}
for _ in range(100):
    coords, off = build_coords_fixed(env)
    h_enc = pol.encode(coords)
    avail = env.available_actions()
    taken = set(); acts = {}
    for ai in range(env.n_agents):
        if env.agent_busy[ai]: continue
        at = env.agent_types[ai]
        cand = env.filter_in_range(ai, avail[at])
        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=h_enc.device)
        for c in cand:
            ci = _env2coord(c, env.n_patrol, env.grid_size, off)
            if ci >= 0 and ci not in taken: mask[ci] = True
        if mask.sum() == 0: counts['skip'] += 1; continue
        pid = off['agents'][0] + ai
        ci = pol.act_greedy(h_enc, pid, pol.TYPE[at], mask)
        counts[at] += 1
        if ci != 0: taken.add(ci)
        acts[ai] = _coord2env(ci, env.n_patrol, off)
    env.step(acts)
print(counts)
