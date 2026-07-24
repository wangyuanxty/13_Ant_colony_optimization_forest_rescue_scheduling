"""Quick test: 10 episodes to verify gradient flow."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from forest_rescue_rl.env import ForestRescueEnv
from forest_rescue_rl.model import RescuePolicy
from forest_rescue_rl.train import run_episode

device = torch.device('cuda')
policy = RescuePolicy(device=device)
env = ForestRescueEnv(device=device)

for ep in range(10):
    rets, lps, vals, info = run_episode(env, policy)
    if not lps:
        print(f"ep {ep}: no decisions"); continue
    rets_t = torch.tensor(rets, device=device).float()
    vals_t = torch.stack(vals).squeeze(-1).float()
    lps_t = torch.stack(lps)
    adv = rets_t - vals_t.detach()
    p_loss = -(lps_t * adv).mean()
    bl_loss = (rets_t - vals_t).pow(2).mean()
    total = p_loss + bl_loss
    policy.opt.zero_grad(); policy.bl_opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
    policy.opt.step(); policy.bl_opt.step()
    loss = p_loss
    print(f"ep {ep}: n={len(lps)} loss={loss.item():.3f} ext={info['extinguished']} "
          f"dam={info['damage']:.0f} cov={info['covered']} dist={info['dist']:.0f}")
    if ep == 0:
        a = env.available_actions()
        print(f"  avail: drone={len(a['drone'])} heli={len(a['heli'])} ground={len(a['ground'])} "
              f"fires={env.fire_grid.sum().item():.0f} tpt={env.transport_done.sum().item()}")
print("Done!")
