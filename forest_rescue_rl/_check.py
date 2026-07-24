import torch
h = torch.load("forest_rescue_rl/train_history.pt")
n = len(h['p_loss'])
print(f"Episodes: {n}")
print(f"p_loss: {h['p_loss'][0]:.0f} -> {h['p_loss'][n//2]:.0f} -> {h['p_loss'][-1]:.0f}")
print(f"bl_loss: {h['bl_loss'][0]:.0f} -> {h['bl_loss'][n//2]:.0f} -> {h['bl_loss'][-1]:.0f}")
print(f"ext: {h['ext'][0]:.0f} -> {h['ext'][n//2]:.0f} -> {h['ext'][-1]:.0f}")
