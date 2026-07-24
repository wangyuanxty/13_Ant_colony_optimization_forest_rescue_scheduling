"""
Baseline ACO benchmark using scikit-opt (https://github.com/guofei9987/scikit-opt).
Matches paper settings: 30 ants, 1000 iterations, 30 runs per dataset.
"""
import math, os, time
import numpy as np
from sko.ACA import ACA_TSP

# ── TSP Parser ───────────────────────────────────────────────────────────────

def parse_tsp(path):
    coords = []; etype = "EUC_2D"; in_sec = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("EDGE_WEIGHT_TYPE"):
                etype = line.split(":")[-1].strip()
            if line.upper().startswith("NODE_COORD_SECTION"):
                in_sec = True; continue
            if line == "EOF": break
            if in_sec:
                p = line.split()
                if len(p) >= 3: coords.append((float(p[1]), float(p[2])))
    return np.array(coords), etype


def build_dist(coords, etype):
    """Return NxN distance matrix (EUC_2D, same convention as paper)."""
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dij = math.sqrt(dx * dx + dy * dy)
            d[i, j] = dij; d[j, i] = dij
    return d


def cal_total_distance(routine, dist_matrix):
    """TSP tour length from permutation."""
    num_nodes = len(routine)
    return sum(dist_matrix[routine[i % num_nodes], routine[(i + 1) % num_nodes]]
               for i in range(num_nodes))


# ── Benchmark ────────────────────────────────────────────────────────────────

PAPER = {'ulysses16': 74.87, 'eil51': 503.63, 'ch150': 7393.77, 'att532': 132242.11}

def bench(ds_path, n_runs=30, size_pop=30, max_iter=1000):
    coords, etype = parse_tsp(ds_path)
    n = len(coords)
    d = build_dist(coords, 'EUC_2D')  # paper convention: all EUC_2D
    name = os.path.basename(ds_path).replace('.tsp', '')

    print(f"\n{'='*60}")
    print(f"  {name}  |  {n} cities  |  {n_runs} runs × ({size_pop} ants, {max_iter} iter)")
    print(f"{'='*60}")

    Ls = np.empty(n_runs); best = float('inf'); t0 = time.time()
    for i in range(n_runs):
        aca = ACA_TSP(func=lambda r: cal_total_distance(r, d), n_dim=n,
                      size_pop=size_pop, max_iter=max_iter,
                      distance_matrix=d)
        _, L = aca.run()
        Ls[i] = L
        if L < best: best = L
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"  run {i+1:2d}/{n_runs}  cur={L:>12.2f}  best={best:>12.2f}  "
                  f"avg={Ls[:i+1].mean():>12.2f}  {el:.0f}s")

    el = time.time() - t0
    print(f"  {'─'*56}")
    print(f"  min={Ls.min():>12.2f}  mean={Ls.mean():>12.2f}  "
          f"max={Ls.max():>12.2f}  std={Ls.std():>10.2f}  time={el:.0f}s")
    return {'name': name, 'n': n, 'min': Ls.min(), 'mean': Ls.mean(),
            'max': Ls.max(), 'std': Ls.std()}


def compare(rs):
    print(f"\n\n{'='*85}")
    print("  COMPARISON: scikit-opt ACO vs Paper ACO (Table 4)")
    print(f"{'='*85}")
    print(f"  {'Dataset':<12} {'n':>5} {'Our Mean':>12} {'Paper Mean':>12} "
          f"{'Δ%':>7} {'Our Min':>12} {'Our Max':>12} {'Std':>10}")
    print(f"  {'─'*83}")
    for r in rs:
        nm = r['name']; pm = PAPER.get(nm); om = r['mean']
        dp = f"{(om-pm)/pm*100:+.1f}%" if pm else "N/A"
        print(f"  {nm:<12} {r['n']:>5} {om:>12.2f} {pm:>12.2f} {dp:>7} "
              f"{r['min']:>12.2f} {r['max']:>12.2f} {r['std']:>10.2f}")
    print(f"{'='*85}")


if __name__ == '__main__':
    all_r = []
    for ds in ['ulysses16.tsp', 'eil51.tsp', 'ch150.tsp', 'att532.tsp']:
        if not os.path.exists(ds): print(f"SKIP: {ds}"); continue
        all_r.append(bench(ds))
    compare(all_r)
