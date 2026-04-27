import random
import numpy as np

from src.envs import CustomFrozenLake


def epsilon_greedy(Q, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    return int(np.argmax(Q[state]))


def _get_holes(env):
    desc = env.desc
    nrow, ncol = env.nrow, env.ncol
    return {
        r * ncol + c
        for r in range(nrow)
        for c in range(ncol)
        if desc[r][c].decode("utf-8") == "H"
    }


def _safe_actions(env, state, holes, n_actions):
    """Actions dont aucune transition possible ne mène dans un trou."""
    safe = [
        a for a in range(n_actions)
        if all(ns not in holes for (_, ns, _, _) in env.unwrapped.P[state][a])
    ]
    return safe if safe else list(range(n_actions))


def _fell(env, state):
    row, col = divmod(state, env.ncol)
    return env.desc[row][col].decode("utf-8") == "H"


def q_learning(env, gamma, lr, episodes, seed=0,
               eps_start=1.0, eps_min=0.05, eps_decay=0.9995,
               lr_min=0.01, lr_decay=0.9995):

    np.random.seed(seed)
    random.seed(seed)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    history = {"rewards": [], "falls": [], "timeouts": [], "steps": []}

    for ep in range(episodes):
        eps  = max(eps_min, eps_start * (eps_decay ** ep))
        lr_t = max(lr_min,  lr        * (lr_decay  ** ep))

        state, _ = env.reset(seed=seed + ep)
        total_reward, fell, timed_out, steps, done = 0.0, False, False, 0, False

        while not done:
            action = epsilon_greedy(Q, state, eps, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] += lr_t * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            steps += 1
            if terminated and _fell(env, next_state):
                fell = True
            if truncated and not terminated:
                timed_out = True
            state = next_state

        history["rewards"].append(total_reward)
        history["falls"].append(int(fell))
        history["timeouts"].append(int(timed_out))
        history["steps"].append(steps)

    return Q, history


def shielded_qlearning(env, gamma, lr, episodes, seed=0,
                       eps_start=1.0, eps_min=0.05, eps_decay=0.9995,
                       lr_min=0.01, lr_decay=0.9995):
    """Q-learning avec shield statique.

    Inspiré de : Odriozola-Olalde et al. (2025), Fear Field framework.
    Simplification pour environnements stationnaires : le shield est construit
    directement depuis la table de transitions P (modèle exact connu),
    sans réseau de neurones ni adaptation temporelle.
    À chaque pas, les actions menant dans un trou avec probabilité > 0
    sont masquées avant la sélection epsilon-greedy.
    """
    np.random.seed(seed)
    random.seed(seed)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    history = {"rewards": [], "falls": [], "timeouts": [], "steps": []}

    holes = _get_holes(env)

    for ep in range(episodes):
        eps  = max(eps_min, eps_start * (eps_decay ** ep))
        lr_t = max(lr_min,  lr        * (lr_decay  ** ep))

        state, _ = env.reset(seed=seed + ep)
        total_reward, fell, timed_out, steps, done = 0.0, False, False, 0, False

        while not done:
            safe = _safe_actions(env, state, holes, n_actions)

            if random.random() < eps:
                action = random.choice(safe)
            else:
                masked = np.full(n_actions, -np.inf)
                masked[safe] = Q[state, safe]
                action = int(np.argmax(masked))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] += lr_t * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            steps += 1
            if terminated and _fell(env, next_state):
                fell = True
            if truncated and not terminated:
                timed_out = True
            state = next_state

        history["rewards"].append(total_reward)
        history["falls"].append(int(fell))
        history["timeouts"].append(int(timed_out))
        history["steps"].append(steps)

    return Q, history


def sarsa(env, gamma, lr, episodes, seed=0,
          eps_start=1.0, eps_min=0.05, eps_decay=0.9995,
          lr_min=0.01, lr_decay=0.9995):

    np.random.seed(seed)
    random.seed(seed)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    history = {"rewards": [], "falls": [], "timeouts": [], "steps": []}

    for ep in range(episodes):
        eps  = max(eps_min, eps_start * (eps_decay ** ep))
        lr_t = max(lr_min,  lr        * (lr_decay  ** ep))

        state, _ = env.reset(seed=seed + ep)
        action = epsilon_greedy(Q, state, eps, n_actions)
        total_reward, fell, timed_out, steps, done = 0.0, False, False, 0, False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = epsilon_greedy(Q, next_state, eps, n_actions)

            # on-policy : met à jour avec l'action réellement choisie
            Q[state, action] += lr_t * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            total_reward += reward
            steps += 1
            if terminated and _fell(env, next_state):
                fell = True
            if truncated and not terminated:
                timed_out = True
            state, action = next_state, next_action

        history["rewards"].append(total_reward)
        history["falls"].append(int(fell))
        history["timeouts"].append(int(timed_out))
        history["steps"].append(steps)

    return Q, history
