"""
Fast att532 runner — swap-and-pop ACO (scikit-opt too slow for 500+ cities).
"""
import math, time, random

def parse(path):
    coords=[]
    with open(path) as f:
        in_sec=False
        for line in f:
            line=line.strip()
            if line.upper().startswith("NODE_COORD_SECTION"): in_sec=True; continue
            if line=="EOF": break
            if in_sec:
                p=line.split()
                if len(p)>=3: coords.append((float(p[1]),float(p[2])))
    return coords

def dist_matrix(coords):
    n=len(coords); d=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            dx=coords[i][0]-coords[j][0]; dy=coords[i][1]-coords[j][1]
            d[i][j]=math.sqrt(dx*dx+dy*dy); d[j][i]=d[i][j]
    return d

class FastACO:
    def __init__(self,dist,n_ants=30,beta=2.0,rho=0.1,seed=None):
        self.dist=dist; self.n=len(dist); self.na=n_ants; self.rho=rho
        self.rng=random.Random(seed)
        self.eb=[[0.0]*self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i!=j: self.eb[i][j]=(1.0/dist[i][j])**beta
        self.tau=[[1.0]*self.n for _ in range(self.n)]

    def run(self,max_iter=1000):
        best_L=float('inf'); tau=self.tau; eb=self.eb; d=self.dist
        n=self.n; na=self.na; rho=self.rho; rng=self.rng
        for _ in range(max_iter):
            tours=[]; Ls=[]
            for _ in range(na):
                t=self._ant(tau,eb,n,rng)
                L=sum(d[t[s]][t[s+1]] for s in range(len(t)-1))
                tours.append(t); Ls.append(L)
                if L<best_L: best_L=L; best_T=t
            for i in range(n):
                row=tau[i]
                for j in range(n): row[j]*=(1.0-rho)
            for k in range(na):
                delta=1.0/Ls[k]; t=tours[k]
                for s in range(len(t)-1):
                    i,j=t[s],t[s+1]; tau[i][j]+=delta; tau[j][i]+=delta
        return best_T,best_L

    def _ant(self,tau,eb,n,rng):
        unvisited=list(range(n)); cur=unvisited.pop(rng.randrange(n)); tour=[cur]
        for _ in range(n-1):
            tr=tau[cur]; er=eb[cur]; m=len(unvisited)
            denom=sum(tr[j]*er[j] for j in unvisited)
            if denom==0: idx=rng.randrange(m)
            else:
                r=rng.random()*denom; cum=0.0; idx=m-1
                for i,j in enumerate(unvisited):
                    cum+=tr[j]*er[j]
                    if cum>=r: idx=i; break
            nxt=unvisited[idx]; unvisited[idx]=unvisited[-1]; unvisited.pop()
            tour.append(nxt); cur=nxt
        tour.append(tour[0]); return tour

if __name__=='__main__':
    import sys
    coords=parse('att532.tsp'); d=dist_matrix(coords)
    n_runs=1; Ls=[]; best=float('inf'); t0=time.time()
    log=open('att532_result.txt','w')
    def p(msg):
        print(msg); print(msg,file=log); log.flush(); sys.stdout.flush()
    p(f"Starting att532: {n_runs} runs, 30 ants, 1000 iter")
    for i in range(n_runs):
        aco=FastACO(d,seed=i); _,L=aco.run(1000)
        Ls.append(L)
        if L<best: best=L
        el=time.time()-t0
        p(f"run {i+1}/{n_runs} cur={L:.1f} best={best:.1f} "
          f"avg={sum(Ls)/(i+1):.1f} {el:.0f}s")
    avg=sum(Ls)/len(Ls); std=(sum((x-avg)**2 for x in Ls)/len(Ls))**0.5
    p(f"\nmin={min(Ls):.1f} mean={avg:.1f} max={max(Ls):.1f} std={std:.1f}")
    p(f"paper_ACO=132242.11  Δ={(avg-132242.11)/132242.11*100:+.1f}%")
    log.close()
