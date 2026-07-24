"""
Forest Rescue RL Environment
Agents: drones (recon), helicopters (transport), ground teams (firefighting)
Fires: random ignition + spread. Weather: dynamic no-fly edges.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class ForestRescueEnv:
    def __init__(self, n_patrol=150, n_drones=3, n_helis=2, n_ground=1,
                 world_size=100.0, device='cpu'):
        self.world_size = world_size; self.device = device
        self.n_patrol = n_patrol
        self.n_drones = n_drones; self.n_helis = n_helis; self.n_ground = n_ground
        self.n_agents = n_drones + n_helis + n_ground
        self.agent_types = (['drone'] * n_drones + ['heli'] * n_helis
                            + ['ground'] * n_ground)
        self.speeds = {'drone': 120., 'heli': 60., 'ground': 10.}
        self.max_range = {'drone': 300., 'heli': 500., 'ground': 9999.}
        self.grid_size = 8; self.max_t = 100

    def reset(self, n_patrol=None, n_drones=None, n_helis=None, n_ground=None,
              fire_prob=0.01, spread_prob=0.3):
        if n_patrol is not None: self.n_patrol = n_patrol
        if n_drones is not None: self.n_drones = n_drones
        if n_helis is not None: self.n_helis = n_helis
        if n_ground is not None: self.n_ground = n_ground
        self.n_agents = self.n_drones + self.n_helis + self.n_ground
        self.agent_types = (['drone'] * self.n_drones + ['heli'] * self.n_helis
                            + ['ground'] * self.n_ground)
        # Patrol points
        self.patrol_pts = torch.rand(self.n_patrol, 2) * self.world_size
        self.patrol_visited = torch.zeros(self.n_patrol, dtype=torch.bool)
        # Agents at base (0,0)
        self.agent_pos = torch.zeros(self.n_agents, 2)
        self.agent_target = torch.full((self.n_agents,), -1, dtype=torch.long)
        self.agent_busy = torch.zeros(self.n_agents, dtype=torch.bool)
        self.agent_range_used = [0.0] * self.n_agents
        # Fires: grid-based
        self.fire_prob = fire_prob; self.spread_prob = spread_prob
        self.fire_grid = torch.zeros(self.grid_size, self.grid_size)
        self.transport_done = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
        # Weather: no-fly edges
        self.no_fly_edges = set()
        # Stats
        self.t = 0; self.max_t = 200
        self.fires_extinguished = 0; self.fire_damage = 0.0
        self.flight_dist = 0.0; self.patrol_covered = 0
        self._prev_covered = 0; self._prev_ext = 0
        return self._get_state()

    def step(self, actions: Dict[int, int]):
        """
        actions: {busy_agent_idx: target_point_idx}
        target_point_idx >= 0 → go to that point; -1 → base
        """
        # Assign targets
        for ai, tgt in actions.items():
            self.agent_target[ai] = tgt
            if tgt < 0:  # going to base — arrive immediately
                self.agent_busy[ai] = False
            else:
                self.agent_busy[ai] = True

        # Move agents
        for i in range(self.n_agents):
            if not self.agent_busy[i] or self.agent_target[i] < 0:
                continue
            tp = self._point_pos(self.agent_target[i])
            if tp is None: self.agent_busy[i] = False; continue
            d = torch.norm(tp - self.agent_pos[i])
            sp = self.speeds[self.agent_types[i]]
            if d <= sp:  # arrived
                self.agent_pos[i] = tp.clone()
                self.agent_range_used[i] += d
                self._on_arrival(i)
            else:
                self.agent_pos[i] += (tp - self.agent_pos[i]) / d * sp
                self.flight_dist += sp
                self.agent_range_used[i] += sp

        # Fire ignition + spread
        self.fire_grid += (torch.rand(self.grid_size, self.grid_size) < self.fire_prob).float()
        self.fire_grid = torch.clamp(self.fire_grid, 0, 1)
        # Spread
        spread = torch.rand(self.grid_size, self.grid_size) < self.spread_prob
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if self.fire_grid[gx, gy] > 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if spread[nx, ny]:
                                self.fire_grid[nx, ny] = 1.0
        self.fire_damage += self.fire_grid.sum().item()

        # Weather
        self._update_weather()

        # Per-step reward
        r_step = 0.0
        r_step += (self.patrol_covered - self._prev_covered) * 50.0
        r_step += (self.fires_extinguished - self._prev_ext) * 500.0
        r_step -= self.fire_grid.sum().item() * 0.1
        r_step -= 0.01
        r_step += len(actions) * 0.5
        self._prev_covered = self.patrol_covered
        self._prev_ext = self.fires_extinguished

        self.t += 1
        done = self.t >= self.max_t
        info = {'extinguished': self.fires_extinguished, 'damage': self.fire_damage,
                'dist': self.flight_dist, 'covered': self.patrol_covered}
        return self._get_state(), r_step, done, info
        return self._get_state(), reward, done, info

    # ── internal ──

    def _on_arrival(self, ai):
        t = self.agent_types[ai]; ti = self.agent_target[ai]
        if ti < 0:  # arrived at base — reset range
            self.agent_range_used[ai] = 0.0
        gx, gy = self._grid(self.agent_pos[ai])
        _DEBUG = False
        if _DEBUG: print(f"  _on_arrival: ai={ai} type={t} target={ti} pos={(gx,gy)} grid_fire={self.fire_grid[gx,gy].item()}")
        if t == 'drone' and ti < self.n_patrol and not self.patrol_visited[ti]:
            self.patrol_visited[ti] = True; self.patrol_covered += 1
        elif t == 'heli':
            if self.fire_grid[gx, gy] > 0:
                self.transport_done[gx, gy] = True
                if _DEBUG: print(f"    transport_done[{gx},{gy}]=True")
        elif t == 'ground':
            gx, gy = self._grid(self.agent_pos[ai])
            if self.fire_grid[gx, gy] > 0 and self.transport_done[gx, gy]:
                self.fire_grid[gx, gy] = 0.0; self.fires_extinguished += 1
        self.agent_busy[ai] = False; self.agent_target[ai] = -1

    def _get_state(self):
        # Fire coords needing transport
        fc = []; rc = []
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if self.fire_grid[gx, gy] > 0:
                    cx, cy = self._coord(gx, gy)
                    if not self.transport_done[gx, gy]:
                        fc.append([cx, cy])
                    else:
                        rc.append([cx, cy])
        fc = torch.tensor(fc, dtype=torch.float32) if fc else torch.zeros(0, 2)
        rc = torch.tensor(rc, dtype=torch.float32) if rc else torch.zeros(0, 2)
        return {
            'patrol': self.patrol_pts.clone(),
            'transport_targets': fc,
            'firefight_targets': rc,
            'agents': self.agent_pos.clone(),
            'agent_types': list(self.agent_types),
            'agent_busy': self.agent_busy.clone(),
            'base': torch.zeros(1, 2),
        }

    def _point_pos(self, idx):
        if idx < 0: return None  # base
        if idx < self.n_patrol: return self.patrol_pts[idx]
        # fire grid point
        gx = (idx - self.n_patrol) // self.grid_size
        gy = (idx - self.n_patrol) % self.grid_size
        return torch.tensor(self._coord(gx, gy), dtype=torch.float32)

    def _grid(self, pos):
        gx = min(int(pos[0].item() / self.world_size * self.grid_size), self.grid_size - 1)
        gy = min(int(pos[1].item() / self.world_size * self.grid_size), self.grid_size - 1)
        return gx, gy

    def _coord(self, gx, gy):
        return ((gx + 0.5) * self.world_size / self.grid_size,
                (gy + 0.5) * self.world_size / self.grid_size)

    def _update_weather(self):
        self.no_fly_edges = set()
        if np.random.random() < 0.1:
            cx = np.random.random() * self.world_size
            cy = np.random.random() * self.world_size
            r = np.random.random() * 30 + 10
            for i in range(self.n_patrol):
                if torch.norm(self.patrol_pts[i] - torch.tensor([cx, cy])) < r:
                    for j in range(i + 1, self.n_patrol):
                        if torch.norm(self.patrol_pts[j] - torch.tensor([cx, cy])) < r:
                            self.no_fly_edges.add((i, j))
                            self.no_fly_edges.add((j, i))

    def available_actions(self):
        """Return per-type available point indices. -1 = base.
        Range-constrained: targets must be reachable + return to base."""
        uv = torch.where(~self.patrol_visited)[0].tolist()
        tn = []; fr = []
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                if self.fire_grid[gx, gy] > 0:
                    idx = self.n_patrol + gx * self.grid_size + gy
                    if not self.transport_done[gx, gy]:
                        tn.append(idx)
                    else:
                        fr.append(idx)
        return {'drone': uv + ([-1] if not uv else []),
                'heli': tn + ([-1] if not tn else []),
                'ground': fr + ([-1] if not fr else [])}

    def filter_in_range(self, ai, candidates):
        """Remove candidates that exceed agent's remaining range."""
        at = self.agent_types[ai]
        max_r = self.max_range[at]
        remaining = max_r - self.agent_range_used[ai]
        pos = self.agent_pos[ai]
        def in_range(c):
            if c < 0: return True  # base always allowed
            tp = self._point_pos(c)
            if tp is None: return True
            d_to = float(torch.norm(tp - pos))
            d_back = float(torch.norm(tp - torch.zeros(2)))  # to base
            return (d_to + d_back) <= remaining * 0.8  # 80% safety margin
        return [c for c in candidates if in_range(c)]
