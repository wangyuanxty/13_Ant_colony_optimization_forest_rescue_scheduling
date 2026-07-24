"""
Test CNN-Transformer pre-trained models on paper's 4 TSP datasets.
Run: D:\anaconda\envs\py312\python.exe cnn_transformer_test.py
"""
import sys, os, time
import torch
import numpy as np

BASE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE, "CNN_Transformer3"))
from model_search import TSP_net

class DotDict(dict):
    def __init__(self, **kwds): self.update(kwds); self.__dict__ = self

def parse_tsp(path):
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

def tour_len(x, tour):
    bsz = x.shape[0]; nb = x.shape[1]
    ar = torch.arange(bsz, device=x.device)
    prev = x[ar, tour[:, 0], :]; L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1, nb):
            cur = x[ar, tour[:, i], :]
            L += (cur - prev).pow(2).sum(dim=1).sqrt().round()
            prev = cur
        first = x[ar, tour[:, 0], :]
        L += (prev - first).pow(2).sum(dim=1).sqrt().round()
    return L

def load_model(args, ckpt_path, device):
    model = TSP_net(args.embedding, args.nb_neighbors, args.kernel_size,
                    args.dim_input_nodes, args.dim_emb, args.dim_ff,
                    args.nb_layers_encoder, args.nb_layers_decoder,
                    args.nb_heads, args.max_len_PE,
                    segm_len=args.segm_len, batchnorm=args.batchnorm)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_baseline'])
    return model.to(device).eval()

if __name__ == '__main__':
    device = torch.device('cuda')
    print(f"Device: {device}")

    args = DotDict(gpu_id='0', segm_len=5, dim_emb=128, dim_ff=512,
                   dim_input_nodes=2, nb_layers_encoder=6, nb_layers_decoder=2,
                   nb_heads=8, batchnorm=True, max_len_PE=10000)

    ckpt_dir = os.path.join(BASE, "CNN_Transformer3/checkpoint")
    tests = [
        ('tsp50_cnn', 'conv', 10, 11, ['ulysses16', 'eil51', 'ch150']),
        ('tsp100_cnn', 'conv', 10, 11, ['ulysses16', 'eil51', 'ch150', 'att532']),
    ]

    paper = {'ulysses16': 74.87, 'eil51': 503.63, 'ch150': 7393.77, 'att532': 132242.11}
    opt = {'eil51': 426, 'ch150': 6528}
    our_aco = {'ulysses16': 74.77, 'eil51': 456.51, 'ch150': 6916.02, 'att532': 109696.1}

    for name, emb, nb_n, k_sz, dss in tests:
        ckpt = os.path.join(ckpt_dir, f'{name}_m5.pkl')
        args.embedding = emb; args.nb_neighbors = nb_n; args.kernel_size = k_sz
        model = load_model(args, ckpt, device)
        print(f"\n{'='*60}\n  Model: {name}\n{'='*60}")

        for ds in dss:
            p = os.path.join(BASE, f'{ds}.tsp')
            if not os.path.exists(p): print(f"  {ds}: not found"); continue
            coords = parse_tsp(p); n = len(coords)
            B = 128 if n < 30 else (1500 if n < 80 else (2500 if n < 150 else 1000))
            # Normalize to [0,1] for model input, keep original for distance
            mn = coords.min(axis=0); mx = coords.max(axis=0)
            coords_norm = (coords - mn) / (mx - mn + 1e-8)
            x_norm = torch.from_numpy(coords_norm).unsqueeze(0).to(device)
            x_orig = torch.from_numpy(coords).unsqueeze(0).to(device)

            t0 = time.time()
            do_beam = (n > 20 and n <= 150)
            with torch.no_grad():
                tg, tbs, _, _ = model(x_norm, B, greedy=True, beamsearch=do_beam)
            dt = time.time() - t0

            Lg = tour_len(x_orig, tg).item()
            g_p = (Lg - paper[ds]) / paper[ds] * 100
            g_o = f", opt_gap={100*(Lg-opt[ds])/opt[ds]:.1f}%" if ds in opt else ""
            print(f"  {ds} n={n} B={B} {dt:.1f}s: greedy={Lg:.1f} "
                  f"Δpaper={g_p:+.1f}%{g_o}")

            if do_beam:
                xr = x_orig.repeat_interleave(B, dim=0); tbs = tbs.view(B, n)
                Lbs = tour_len(xr, tbs).view(1, B).min(dim=1)[0].item()
                b_p = (Lbs - paper[ds]) / paper[ds] * 100
                b_o = f", opt_gap={100*(Lbs-opt[ds])/opt[ds]:.1f}%" if ds in opt else ""
                print(f"    beamsearch={Lbs:.1f} Δpaper={b_p:+.1f}%{b_o}")

    print(f"\n{'='*60}\n  Reference\n{'='*60}")
    print(f"  {'Dataset':<12} {'paper_ACO':>10} {'our_ACO':>10}")
    for ds in ['ulysses16','eil51','ch150','att532']:
        print(f"  {ds:<12} {paper[ds]:>10.2f} {our_aco[ds]:>10.2f}")
