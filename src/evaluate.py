import heapq
from itertools import count as _count

import numpy as np

from src.envs import CustomFrozenLake


def is_dangerous(desc, row, col):
    """Retourne True si la case est adjacente à un trou."""
    nrow, ncol = len(desc), len(desc[0])
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < nrow and 0 <= c < ncol:
            if desc[r][c].decode("utf-8") == "H":
                return True
    return False


def evaluate_agent(env, Q, n_episodes=500, max_steps=200):
    ncol = env.ncol
    desc = env.desc

    successes, falls = 0, 0
    total_steps, total_danger_steps = 0, 0
    episodes_danger_steps = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        done = False
        ep_danger = 0

        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_steps += 1
            row, col = divmod(next_state, ncol)
            tile = desc[row][col].decode("utf-8")

            if is_dangerous(desc, row, col):
                total_danger_steps += 1
                ep_danger += 1

            if done:
                if tile == "G":
                    successes += 1
                elif tile == "H":
                    falls += 1
                break

            state = next_state

        episodes_danger_steps.append(ep_danger)

    timeouts = n_episodes - successes - falls

    return {
        "success_rate":          successes / n_episodes,
        "fall_rate":             falls     / n_episodes,
        "timeout_rate":          timeouts  / n_episodes,
        "avg_steps":             total_steps / n_episodes,
        "danger_rate":           total_danger_steps / max(total_steps, 1),
        "episodes_danger_steps": episodes_danger_steps,
    }


def safest_path(env, danger_penalty=3):
    """Dijkstra pénalisant les cases adjacentes à un trou.
    Retourne le chemin minimisant l'exposition aux situations dangereuses."""
    desc = env.desc
    nrow = len(desc)
    ncol = len(desc[0])

    start = None
    goals = []
    for r in range(nrow):
        for c in range(ncol):
            tile = desc[r][c].decode("utf-8")
            if tile == "S":
                start = r * ncol + c
            elif tile == "G":
                goals.append(r * ncol + c)
    goal_set = set(goals)

    tiebreak   = _count()
    pq         = [(0, next(tiebreak), start, [start])]
    best_cost  = {start: 0}
    best_path  = None
    best_total = float("inf")

    while pq:
        cost, _, state, path = heapq.heappop(pq)

        if cost > best_cost.get(state, float("inf")):
            continue
        if cost >= best_total:
            continue
        if state in goal_set:
            best_path, best_total = path, cost
            continue

        r, c = divmod(state, ncol)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < nrow and 0 <= nc_ < ncol:
                tile = desc[nr][nc_].decode("utf-8")
                if tile == "H":
                    continue
                ns        = nr * ncol + nc_
                step_cost = 1 + (danger_penalty if is_dangerous(desc, nr, nc_) else 0)
                new_cost  = cost + step_cost
                if new_cost < best_cost.get(ns, float("inf")):
                    best_cost[ns] = new_cost
                    heapq.heappush(pq, (new_cost, next(tiebreak), ns, path + [ns]))

    return best_path


def evaluate_safe_path_rate(env, Q, n_episodes=500, max_steps=200, tolerance=0.5):
    """Fraction des épisodes réussis dont la trajectoire chevauche le chemin Dijkstra
    à au moins `tolerance` (défaut 50 %)."""
    safe_path = safest_path(env)
    if safe_path is None:
        return None
    safe_set = set(safe_path)

    ncol = env.ncol
    desc = env.desc
    safe_count    = 0
    total_success = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        path = [state]

        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            next_state, _, terminated, truncated, _ = env.step(action)
            path.append(next_state)

            row, col = divmod(next_state, ncol)
            tile = desc[row][col].decode("utf-8")

            if terminated or truncated:
                if tile == "G":
                    total_success += 1
                    overlap = sum(1 for s in path if s in safe_set) / len(path)
                    if overlap >= tolerance:
                        safe_count += 1
                break

            state = next_state

    if total_success == 0:
        return 0.0
    return safe_count / total_success
