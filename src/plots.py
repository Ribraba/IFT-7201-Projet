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

ALGO_COLOR = {"qlearning": COLORS[0], "sarsa": COLORS[2]}
ALGO_LABEL = {"qlearning": "Q-learning", "sarsa": "SARSA"}

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
            if env_name not in exp_name:
                continue

            algo  = "qlearning" if "qlearning" in exp_name else "sarsa"
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
        axes[1].set_ylim(-5, 105)
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
        axes[1].legend(loc="upper right")
        axes[2].set_ylabel("Taux de boucles (%)")
        axes[2].set_xlabel(f"Épisode (fenêtre = {window})")
        axes[2].set_ylim(-5, 105)
        axes[2].yaxis.set_major_formatter(mticker.PercentFormatter())
        axes[2].legend(loc="upper right")

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
    env_names = sorted({exp.split("_", 1)[1] for exp in experiments})

    for env_name in env_names:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Évaluation finale — {env_name}", fontsize=12)

        for ax, metric, label in zip(axes, metrics, labels):
            algos, means, stds, colors = [], [], [], []
            for exp_name, data in sorted(experiments.items()):
                if env_name not in exp_name:
                    continue
                ak = "qlearning" if "qlearning" in exp_name else "sarsa"
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

    env_names  = [e for e in sorted({exp.split("_", 1)[1] for exp in experiments})
                  if e != "easy"]
    env_labels = {"medium": "Moyen", "hard": "Difficile"}

    fig, axes = plt.subplots(1, len(env_names), figsize=(4 * len(env_names), 5), sharey=False)
    if len(env_names) == 1:
        axes = [axes]

    fig.suptitle("Distribution des situations dangereuses par épisode (évaluation)",
                 fontsize=12, fontweight="bold")

    for ax, env_name in zip(axes, env_names):
        data, colors, labels = [], [], []
        for algo in ["qlearning", "sarsa"]:
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

        ax.set_xticks([1, 2])
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

    env_names  = sorted({exp.split("_", 1)[1] for exp in experiments})
    algo_names = ["qlearning", "sarsa"]
    algo_labels = ["Q-learning", "SARSA"]

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

        fig, ax = plt.subplots(figsize=(7, 3))
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

        ncols_fig = 2
        cell_size = max(1.2, 5.0 / max(nrow, ncol))
        fig_w = ncols_fig * ncol * cell_size + 1.0
        fig_h = nrow * cell_size + 1.2
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        fig.suptitle(f"Politiques greedy — {env_name}", fontsize=12, fontweight="bold")

        for ax, algo in zip(axes, ["qlearning", "sarsa"]):
            key = f"{algo}_{env_name}"

            # Récupère Q : depuis les résultats si dispo, sinon entraîne un run de secours
            Q_runs = experiments.get(key, {}).get("Q_runs", [])
            if Q_runs:
                Q_mean = np.mean(Q_runs, axis=0)
            else:
                print(f"  [policy] Q non trouvé pour {key} — entraînement de secours (seed=42)…")
                from src.agents import q_learning as _ql, sarsa as _sarsa
                _fn  = _ql if algo == "qlearning" else _sarsa
                _env = make_env(env_name)
                Q_mean, _ = _fn(_env, gamma=0.99, lr=0.5, episodes=5000, seed=42)

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
                                fontsize=max(8, int(cell_size * 8)),
                                fontweight="bold", color="white")
                    elif tile in ("S", "G"):
                        ax.text(c + 0.5, y + 0.5, tile,
                                ha="center", va="center",
                                fontsize=max(8, int(cell_size * 8)),
                                fontweight="bold", color=TILE_TEXT[tile])
                    else:
                        action = int(np.argmax(Q_mean[state]))
                        ax.text(c + 0.5, y + 0.5, ACTION_ARROW[action],
                                ha="center", va="center",
                                fontsize=max(10, int(cell_size * 10)),
                                color=COLORS[3])

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
