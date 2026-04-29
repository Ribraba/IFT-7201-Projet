import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


# Charte graphique Université Laval
COLORS = ['#e30513', '#ffc103', '#0099ff', '#515151',
          '#f7a941', '#51a27e', '#702e78', '#7b003d']

ALGO_COLOR = {"qlearning": COLORS[0], "sarsa": COLORS[2], "shielded_qlearning": COLORS[4]}
ALGO_LABEL = {"qlearning": "Q-learning", "sarsa": "SARSA", "shielded_qlearning": "Q-learning blindé"}

# Trie par longueur décroissante pour que "shielded_qlearning" soit testé avant "qlearning"
_ALGOS_BY_LEN = sorted(ALGO_COLOR.keys(), key=len, reverse=True)


def _env_of(exp_name):
    """Extrait le nom d'environnement depuis 'shielded_qlearning_hard' → 'hard'."""
    for algo in _ALGOS_BY_LEN:
        if exp_name.startswith(algo + "_"):
            return exp_name[len(algo) + 1:]
    return exp_name.split("_", 1)[1]


# Actions FrozenLake : 0=←  1=↓  2=→  3=↑
ACTION_ARROW = {0: "←", 1: "↓", 2: "→", 3: "↑"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})


def _smooth(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _set_axis_with_margin(ax, lower=None, upper=None, min_span=0.5, pad_ratio=0.08):
    """Ajuste dynamiquement l'échelle à partir des données tracées."""
    values = []
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size:
            values.append(y)
    if not values:
        return

    y = np.concatenate(values)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = max(y_max - y_min, min_span)
    pad = span * pad_ratio

    lo = y_min - pad if lower is None else lower
    hi = y_max + pad if upper is None else upper

    if lower is not None and upper is None:
        hi = max(hi, lower + min_span)
    elif upper is not None and lower is None:
        lo = min(lo, upper - min_span)
    elif lower is None and upper is None and hi - lo < min_span:
        center = (hi + lo) / 2
        lo = center - min_span / 2
        hi = center + min_span / 2

    ax.set_ylim(lo, hi)


def load_results(results_dir):
    experiments = {}
    subdirs = sorted(d for d in os.listdir(results_dir)
                     if os.path.isdir(os.path.join(results_dir, d)))

    for exp_name in subdirs:
        json_files = sorted(glob.glob(os.path.join(results_dir, exp_name, "run_*.json")))
        if not json_files:
            continue

        training_runs, eval_runs, Q_runs = [], [], []
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            training_runs.append(data["training"])
            eval_runs.append(data["evaluation"])
            if "Q" in data:
                Q_runs.append(np.array(data["Q"]))

        experiments[exp_name] = {
            "training":  training_runs,
            "evaluation": eval_runs,
            "Q_runs":    Q_runs,   # liste vide si pas encore disponible
        }
        print(f"  {exp_name}  ({len(json_files)} runs, Q={'oui' if Q_runs else 'non'})")

    return experiments


def plot_training_curves(experiments, env_names, save_dir="figures", window=500):
    """Courbes d'apprentissage sur 3 panneaux : récompense, taux de chutes, taux de boucles."""
    os.makedirs(save_dir, exist_ok=True)

    for env_name in env_names:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(f"Courbes d'apprentissage — {env_name}", fontsize=13, fontweight="bold")

        has_timeout_data = False

        for exp_name, data in experiments.items():
            if _env_of(exp_name) != env_name:
                continue

            algo  = next(k for k in ALGO_COLOR if exp_name.startswith(k))
            color = ALGO_COLOR[algo]
            label = ALGO_LABEL[algo]

            # --- Panneau 0 : récompense cumulée ---
            rewards = np.array([r["rewards"] for r in data["training"]])
            mean_r, std_r = np.mean(rewards, axis=0), np.std(rewards, axis=0)
            if len(mean_r) >= window:
                x = np.arange(len(_smooth(mean_r, window)))
                ms, ss = _smooth(mean_r, window), _smooth(std_r, window)
                axes[0].plot(x, ms, color=color, label=label, linewidth=2)
                axes[0].fill_between(x, ms - ss, ms + ss, color=color, alpha=0.18)

            # --- Panneau 1 : taux de chutes ---
            falls = np.array([r["falls"] for r in data["training"]])
            mean_f, std_f = np.mean(falls, axis=0), np.std(falls, axis=0)
            if len(mean_f) >= window:
                x = np.arange(len(_smooth(mean_f, window)))
                mf, sf = _smooth(mean_f, window) * 100, _smooth(std_f, window) * 100
                axes[1].plot(x, mf, color=color, label=label, linewidth=2)
                axes[1].fill_between(x, mf - sf, mf + sf, color=color, alpha=0.18)

            # --- Panneau 2 : taux de boucles (timeouts) ---
            timeouts_list = [r.get("timeouts", []) for r in data["training"]]
            if any(len(t) > 0 for t in timeouts_list):
                has_timeout_data = True
                timeouts = np.array([t for t in timeouts_list if len(t) > 0])
                mean_t, std_t = np.mean(timeouts, axis=0), np.std(timeouts, axis=0)
                if len(mean_t) >= window:
                    x = np.arange(len(_smooth(mean_t, window)))
                    mt, st = _smooth(mean_t, window) * 100, _smooth(std_t, window) * 100
                    axes[2].plot(x, mt, color=color, label=label, linewidth=2)
                    axes[2].fill_between(x, mt - st, mt + st, color=color, alpha=0.18)

        axes[0].set_ylabel("Récompense par épisode")
        axes[0].legend(loc="lower right")
        axes[1].set_ylabel("Taux de chutes (%)")
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
        axes[1].legend(loc="upper right")
        axes[2].set_ylabel("Taux de boucles (%)")
        axes[2].set_xlabel(f"Épisode (fenêtre = {window})")
        axes[2].yaxis.set_major_formatter(mticker.PercentFormatter())
        axes[2].legend(loc="upper right")
        # Échelles fixes et identiques pour tous les environnements → comparaison directe.
        axes[0].set_ylim(-1.05, 1.05)
        axes[1].set_ylim(-2, 102)
        axes[2].set_ylim(-2, 102)

        if not has_timeout_data:
            axes[2].text(0.5, 0.5,
                         "Données non disponibles\n(relancer run_experiments.py)",
                         ha="center", va="center", transform=axes[2].transAxes,
                         fontsize=10, color=COLORS[3])
            axes[2].grid(False)

        plt.tight_layout()
        path = os.path.join(save_dir, f"training_{env_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Sauvegardé : {path}")


def plot_evaluation_results(experiments, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["success_rate", "fall_rate", "danger_rate"]
    labels  = ["Taux de succès (%)", "Taux de chutes (%)", "Taux d'états dangereux (%)"]
    env_names = sorted({_env_of(exp) for exp in experiments})

    for env_name in env_names:
        # Garde uniquement les métriques avec au moins une valeur non nulle
        active = [
            (m, l) for m, l in zip(metrics, labels)
            if any(
                np.mean([e[m] for e in data["evaluation"]]) > 0.001
                for exp_name, data in experiments.items()
                if _env_of(exp_name) == env_name
            )
        ]
        active_metrics, active_labels = zip(*active)

        fig, axes = plt.subplots(1, len(active_metrics), figsize=(4 * len(active_metrics), 4))
        if len(active_metrics) == 1:
            axes = [axes]
        fig.suptitle(f"Évaluation finale — {env_name}", fontsize=12)

        for ax, metric, label in zip(axes, active_metrics, active_labels):
            algos, means, stds, colors = [], [], [], []
            for exp_name, data in sorted(experiments.items()):
                if _env_of(exp_name) != env_name:
                    continue
                ak = next(k for k in ALGO_COLOR if exp_name.startswith(k))
                vals = np.array([e[metric] for e in data["evaluation"]]) * 100
                algos.append(ALGO_LABEL[ak])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                colors.append(ALGO_COLOR[ak])

            x    = np.arange(len(algos))
            bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                          alpha=0.82, width=0.45, error_kw={"ecolor": COLORS[3]})
            ax.set_xticks(x)
            ax.set_xticklabels(algos, fontsize=10)
            ax.set_ylabel(label)
            ax.set_ylim(0, 115)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{mean:.1f}%", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(save_dir, f"evaluation_{env_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Sauvegardé : {path}")


def plot_danger_boxplot(experiments, save_dir="figures"):
    """Boîtes à moustaches des situations dangereuses par épisode — Q-learning vs SARSA.
    Exclut le cas Facile (déterministe, variance nulle, peu informatif)."""
    os.makedirs(save_dir, exist_ok=True)

    env_names  = [e for e in sorted({_env_of(exp) for exp in experiments})
                  if e != "easy"]
    env_labels = {"medium": "Moyen", "hard": "Difficile"}

    fig, axes = plt.subplots(1, len(env_names), figsize=(4 * len(env_names), 5), sharey=False)
    if len(env_names) == 1:
        axes = [axes]

    fig.suptitle("Distribution des situations dangereuses par épisode (évaluation)",
                 fontsize=12, fontweight="bold")

    for ax, env_name in zip(axes, env_names):
        data, colors, labels = [], [], []
        for algo in ALGO_COLOR:
            key = f"{algo}_{env_name}"
            if key not in experiments:
                continue
            all_steps = []
            for e in experiments[key]["evaluation"]:
                all_steps.extend(e.get("episodes_danger_steps", []))
            if all_steps:
                data.append(all_steps)
                colors.append(ALGO_COLOR[algo])
                labels.append(ALGO_LABEL[algo])

        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set_color(COLORS[3])

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(env_labels.get(env_name, env_name), fontsize=11)
        ax.set_ylabel("Situations dangereuses / épisode")

    plt.tight_layout()
    path = os.path.join(save_dir, "danger_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Sauvegardé : {path}")


def plot_overview(experiments, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    _env_order = ["easy", "medium", "hard"]
    _all_envs  = {_env_of(exp) for exp in experiments}
    env_names  = [e for e in _env_order if e in _all_envs] + sorted(_all_envs - set(_env_order))
    algo_names = list(ALGO_COLOR.keys())
    algo_labels = [ALGO_LABEL[a] for a in algo_names]

    for metric, title, cmap in [
        ("success_rate", "Taux de succès (%)", "Blues"),
        ("fall_rate",    "Taux de chutes (%)", "Reds"),
    ]:
        matrix = np.zeros((len(algo_names), len(env_names)))
        errs   = np.zeros_like(matrix)

        for i, algo in enumerate(algo_names):
            for j, env_name in enumerate(env_names):
                key = f"{algo}_{env_name}"
                if key not in experiments:
                    continue
                vals = np.array([e[metric] for e in experiments[key]["evaluation"]]) * 100
                matrix[i, j] = np.mean(vals)
                errs[i, j]   = np.std(vals)

        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xticks(range(len(env_names)))
        ax.set_xticklabels(env_names)
        ax.set_yticks(range(len(algo_labels)))
        ax.set_yticklabels(algo_labels)
        ax.set_title(title)

        for i in range(len(algo_names)):
            for j in range(len(env_names)):
                ax.text(j, i, f"{matrix[i,j]:.1f}±{errs[i,j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > 50 else "black")

        plt.tight_layout()
        path = os.path.join(save_dir, f"overview_{metric}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Sauvegardé : {path}")


def plot_policy_arrows(experiments, env_names, save_dir="figures"):
    """Visualise la politique greedy (←↓→↑) pour chaque algorithme.
    Le contour vert délimite le chemin de référence calculé par Dijkstra.
    Si les Q-tables ne sont pas dans les résultats, entraîne un run de secours."""
    from src.envs import make_env
    from src.evaluate import safest_path

    TILE_BG   = {"S": COLORS[2], "F": "#f0f4f8", "H": COLORS[0], "G": COLORS[1]}
    TILE_TEXT = {"S": "white",   "F": COLORS[3], "H": "white",   "G": COLORS[3]}

    os.makedirs(save_dir, exist_ok=True)

    for env_name in env_names:
        env = make_env(env_name)
        desc = env.desc
        nrow, ncol = env.nrow, env.ncol

        safe_path = safest_path(env)
        safe_set  = set(safe_path) if safe_path else set()

        algos_present = [a for a in ALGO_COLOR if f"{a}_{env_name}" in experiments]
        ncols_fig = len(algos_present)
        fig_w = max(7.0, 3.5 * ncols_fig)
        fig_h = fig_w / ncols_fig * (nrow / ncol) + 0.8
        fig, axes = plt.subplots(1, ncols_fig, figsize=(fig_w, fig_h))
        if ncols_fig == 1:
            axes = [axes]
        fig.suptitle(f"Politiques greedy — {env_name}", fontsize=12, fontweight="bold")

        for ax, algo in zip(axes, algos_present):
            key = f"{algo}_{env_name}"

            Q_runs = experiments.get(key, {}).get("Q_runs", [])
            if Q_runs:
                Q_mean = np.mean(Q_runs, axis=0)
            else:
                print(f"  [policy] Q non trouvé pour {key} — entraînement de secours (seed=42)…")
                from src.agents import q_learning as _ql, sarsa as _sarsa, shielded_qlearning as _sq
                _fns = {"qlearning": _ql, "sarsa": _sarsa, "shielded_qlearning": _sq}
                _env = make_env(env_name)
                Q_mean, _ = _fns[algo](_env, gamma=0.99, lr=0.5, episodes=5000, seed=42)

            ax.set_xlim(0, ncol)
            ax.set_ylim(0, nrow)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(ALGO_LABEL[algo], fontsize=11, pad=8)

            for r in range(nrow):
                for c in range(ncol):
                    tile  = desc[r][c].decode("utf-8")
                    state = r * ncol + c
                    y     = nrow - r - 1          # ordonnée matplotlib (0 = bas)

                    # Couleur de fond
                    face = TILE_BG.get(tile, "#f0f4f8")
                    # Contour vert si case sur le chemin Dijkstra
                    edge_color = COLORS[5] if state in safe_set else "white"
                    edge_lw    = 3.0     if state in safe_set else 1.0

                    ax.add_patch(mpatches.FancyBboxPatch(
                        (c + 0.05, y + 0.05), 0.90, 0.90,
                        boxstyle="round,pad=0.04",
                        facecolor=face,
                        edgecolor=edge_color,
                        linewidth=edge_lw,
                    ))

                    if tile == "H":
                        ax.text(c + 0.5, y + 0.5, "H",
                                ha="center", va="center",
                                fontsize=16, fontweight="bold", color="white")
                    elif tile in ("S", "G"):
                        ax.text(c + 0.5, y + 0.5, tile,
                                ha="center", va="center",
                                fontsize=16, fontweight="bold", color=TILE_TEXT[tile])
                    else:
                        action = int(np.argmax(Q_mean[state]))
                        ax.text(c + 0.5, y + 0.5, ACTION_ARROW[action],
                                ha="center", va="center",
                                fontsize=18, color=COLORS[3])

            # Légende chemin Dijkstra
            if safe_set:
                legend_patch = mpatches.Patch(
                    facecolor="#f0f4f8", edgecolor=COLORS[5],
                    linewidth=2, label="Chemin Dijkstra (sûr)"
                )
                ax.legend(handles=[legend_patch], loc="upper right",
                          fontsize=7, framealpha=0.85)

        plt.tight_layout()
        path = os.path.join(save_dir, f"policy_{env_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Sauvegardé : {path}")
