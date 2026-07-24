"""
TSPFormer GPU training + inference on paper benchmark datasets.
Run with: D:\anaconda\envs\py312\python.exe quick_train_tspformer.py
"""
import sys, os, time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tspFormer", "tspformer"))

from tspformer.transNet import Tspformer
from utils.options import get_options
from utils.tspLength import compute_tour_length

BASE = os.path.dirname(__file__)

def parse_tsp_coords(path):
    coords = []; in_sec = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("NODE_COORD_SECTION"): in_sec = True; continue
            if line == "EOF": break
            if in_sec:
                p = line.split()
                if len(p) >= 3: coords.append((float(p[1]), float(p[2])))
    return np.array(coords, dtype=np.float32)

def normalize(coords):
    mn = coords.min(axis=0); mx = coords.max(axis=0)
    return (coords - mn) / (mx - mn + 1e-8)

def build_model(args, device):
    return Tspformer(args.dim_input_nodes, args.dim_emb, args.dim_ff,
                     args.nb_layers_encoder, args.nb_layers_decoder,
                     args.nb_heads, args.max_len_PE, batchnorm=args.batchnorm).to(device)

def train(model, args, device, n_nodes=20, n_epochs=500, batches=3000):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bl = build_model(args, device); bl.eval()
    bl.load_state_dict(model.state_dict())
    print(f"Training n={n_nodes}, {n_epochs}ep x {batches}batches, bsz={args.batch_size}, GPU")
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for _ in range(batches):
            x = torch.rand(args.batch_size, n_nodes, 2, device=device)
            t1, lp = model(x, deterministic=False)
            with torch.no_grad(): t2, _ = bl(x, deterministic=True)
            L1 = compute_tour_length(x, t1); L2 = compute_tour_length(x, t2)
            loss = torch.mean((L1 - L2) * lp)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); e1 = e2 = 0.0
        for _ in range(50):
            x = torch.rand(args.batch_size, n_nodes, 2, device=device)
            with torch.no_grad():
                t1, _ = model(x, deterministic=True)
                t2, _ = bl(x, deterministic=True)
            e1 += compute_tour_length(x, t1).mean().item()
            e2 += compute_tour_length(x, t2).mean().item()
        e1 /= 50; e2 /= 50
        if e1 + args.tol < e2: bl.load_state_dict(model.state_dict())
        if (ep + 1) % 10 == 0 or ep == 0:
            dt = time.time() - t0
            gap_n20_opt = 3.75; gap_n50_opt = 5.69
            opt_val = gap_n20_opt if n_nodes == 20 else (gap_n50_opt if n_nodes == 50 else 0)
            gap = (e1 / opt_val - 1.0) * 100 if opt_val else -1
            print(f"  ep {ep+1:4d} | L={e1:.3f} base={e2:.3f} | gap={gap:.1f}% | {dt:.0f}s")
    return model

def infer(model, coords, device):
    model.eval()
    x = torch.from_numpy(normalize(coords)).unsqueeze(0).to(device)
    with torch.no_grad(): tour, _ = model(x, deterministic=True)
    t = tour.squeeze(0).cpu().numpy()
    L = sum(np.sqrt(np.sum((coords[t[i]] - coords[t[(i+1)%len(t)]])**2))
             for i in range(len(t)))
    return L, t

if __name__ == '__main__':
    device = torch.device('cuda')
    args = get_options()
    args.batch_size = 64  # bigger batch for GPU
    print(f"Device: {device}, PT {torch.__version__}")

    # --- n=20 -> ulysses16 (30 epochs quick) ---
    m20 = build_model(args, device)
    train(m20, args, device, n_nodes=20, n_epochs=3, batches=200)
    torch.save(m20.state_dict(), os.path.join(BASE, "tspformer_n20.pth"))

    c = parse_tsp_coords(os.path.join(BASE, "ulysses16.tsp"))
    L, t = infer(m20, c, device)
    print(f"\n  ulysses16: TSPFormer={L:.1f}  paper_ACO=74.87  our_ACO=74.77")

    # --- n=50 -> eil51 (30 epochs quick) ---
    m50 = build_model(args, device)
    train(m50, args, device, n_nodes=50, n_epochs=3, batches=200)
    torch.save(m50.state_dict(), os.path.join(BASE, "tspformer_n50.pth"))

    c = parse_tsp_coords(os.path.join(BASE, "eil51.tsp"))
    L, t = infer(m50, c, device)
    print(f"\n  eil51: TSPFormer={L:.1f}  paper_ACO=503.63  our_ACO=456.51")


    print("\n=== DONE ===")
