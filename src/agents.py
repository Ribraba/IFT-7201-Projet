import random
import numpy as np

from src.envs import CustomFrozenLake


def epsilon_greedy(Q, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    return int(np.argmax(Q[state]))


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
