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


def _get_holes(env):
    return {
        r * env.ncol + c
        for r in range(env.nrow)
        for c in range(env.ncol)
        if env.desc[r][c].decode("utf-8") == "H"
    }


def _safe_actions(env, state, holes):
    n_actions = env.action_space.n
    safe = [
        a for a in range(n_actions)
        if all(ns not in holes for (_, ns, _, _) in env.unwrapped.P[state][a])
    ]
    return safe if safe else list(range(n_actions))


def _greedy_action(Q, state, allowed_actions=None):
    if allowed_actions is None:
        return int(np.argmax(Q[state]))

    masked = np.full(Q.shape[1], -np.inf)
    masked[allowed_actions] = Q[state, allowed_actions]
    return int(np.argmax(masked))


def evaluate_agent(env, Q, n_episodes=500, max_steps=200, apply_shield=False):
    ncol = env.ncol
    desc = env.desc
    holes = _get_holes(env) if apply_shield else None

    successes, falls = 0, 0
    total_steps, total_danger_steps = 0, 0
    episodes_danger_steps = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        done = False
        ep_danger = 0

        for _ in range(max_steps):
            safe_actions = _safe_actions(env, state, holes) if apply_shield else None
            action = _greedy_action(Q, state, safe_actions)
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


def evaluate_safe_path_rate(env, Q, n_episodes=500, max_steps=200, tolerance=0.5,
                            apply_shield=False):
    """Fraction des épisodes réussis dont la trajectoire chevauche le chemin Dijkstra
    à au moins `tolerance` (défaut 50 %)."""
    safe_path = safest_path(env)
    if safe_path is None:
        return None
    safe_set = set(safe_path)

    ncol = env.ncol
    desc = env.desc
    holes = _get_holes(env) if apply_shield else None
    safe_count    = 0
    total_success = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        path = [state]

        for _ in range(max_steps):
            safe_actions = _safe_actions(env, state, holes) if apply_shield else None
            action = _greedy_action(Q, state, safe_actions)
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
