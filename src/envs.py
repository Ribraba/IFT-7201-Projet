import gymnasium as gym
from gymnasium import Wrapper


# ---------------------------------------------------------------------------
# Cartes Ibrahim (version active, inspirée du cliff-walking Sutton & Barto)
# ---------------------------------------------------------------------------
# Cas 1 : 4x4 déterministe (référence sans stochasticité)
# Cas 2 : 4x4 stochastique (même grille, isole l'effet de is_slippery)
# Cas 3 : 6x6 stochastique, falaise rangée 2 + objectif bas-gauche
#   → chemin court risqué (gauche) vs chemin long sûr (droite puis bas)
#   → version initiale trop difficile archivée dans archive_hard_v1
MAPS = {
    "easy": {
        "desc": ["SFFH", "FFFH", "FFFH", "FFGH"],
        "is_slippery": False,
    },
    "medium": {
        "desc": ["SFFH", "FFFH", "FFFH", "FFGH"],
        "is_slippery": True,
    },
    "hard": {
        "desc": ["SFFFFF", "FFFFFF", "HHHHFF", "FFFFFF", "FFFFFF", "GFFFFF"],
        "is_slippery": True,
    },

    # ---------------------------------------------------------------------------
    # Cartes Khalil (branche second_version) — testées avec l'entraînement Ibrahim
    # ---------------------------------------------------------------------------
    # easy_k / medium_k : 6×6, 3 objectifs stratégiques, 6 trous répartis
    # hard_k            : 6×6, 1 objectif central entouré de trous, 9 trous
    "easy_k": {
        "desc": ["SFFFFG", "FFHFHF", "FFFGHF", "FFHFHF", "FFFFFH", "FFFFFG"],
        "is_slippery": False,
    },
    "medium_k": {
        "desc": ["SFFFFG", "FFHFHF", "FFFGHF", "FFHFHF", "FFFFFH", "FFFFFG"],
        "is_slippery": True,
    },
    "hard_k": {
        "desc": ["SFFFFF", "FFHFHF", "FFHGFF", "FFFFFH", "FFFFFF", "HHFHHH"],
        "is_slippery": True,
    },
}


class CustomFrozenLake(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.nrow = env.unwrapped.nrow
        self.ncol = env.unwrapped.ncol
        self.desc = env.unwrapped.desc

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        current_state = self.env.unwrapped.s
        next_state, _, terminated, truncated, info = self.env.step(action)

        row, col = divmod(next_state, self.ncol)
        tile = self.desc[row][col].decode("utf-8")

        if tile == "H":
            reward = -1.0
        elif tile == "G":
            reward = 1.0
        elif next_state == current_state:  # bord de grille
            reward = -0.1
        else:
            reward = 0.0

        return next_state, reward, terminated, truncated, info


def make_env(map_name):
    cfg = MAPS[map_name]
    env = gym.make("FrozenLake-v1", desc=cfg["desc"], is_slippery=cfg["is_slippery"])
    return CustomFrozenLake(env)
