import torch, os
c = torch.load(os.path.join(os.path.dirname(__file__), 'ckpt_rf.pt'), map_location='cpu')
print(f'ep={c["ep"]}')
p = c['hist']['p_loss']
for i in range(0, len(p), len(p)//10 or 1):
    if i < len(p): print(f'  p[{i:4d}]={p[i]:.0f}')
