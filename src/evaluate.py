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

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        done = False

        for _ in range(max_steps):
            action = int(np.argmax(Q[state]))
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_steps += 1
            row, col = divmod(next_state, ncol)
            tile = desc[row][col].decode("utf-8")

            if is_dangerous(desc, row, col):
                total_danger_steps += 1

            if done:
                if tile == "G":
                    successes += 1
                elif tile == "H":
                    falls += 1
                break

            state = next_state

    timeouts = n_episodes - successes - falls

    return {
        "success_rate": successes / n_episodes,
        "fall_rate":    falls     / n_episodes,
        "timeout_rate": timeouts  / n_episodes,
        "avg_steps":    total_steps / n_episodes,
        "danger_rate":  total_danger_steps / max(total_steps, 1),
    }
