import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Charte graphique Université Laval
COLORS = ['#e30513', '#ffc103', '#0099ff', '#515151',
          '#f7a941', '#51a27e', '#702e78', '#7b003d']

ALGO_COLOR = {"qlearning": COLORS[0], "sarsa": COLORS[2]}
ALGO_LABEL = {"qlearning": "Q-learning", "sarsa": "SARSA"}

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

        training_runs, eval_runs = [], []
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            training_runs.append(data["training"])
            eval_runs.append(data["evaluation"])

        experiments[exp_name] = {"training": training_runs, "evaluation": eval_runs}
        print(f"  {exp_name}  ({len(json_files)} runs)")

    return experiments


def plot_training_curves(experiments, env_names, save_dir="figures", window=500):
    os.makedirs(save_dir, exist_ok=True)

    for env_name in env_names:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"Courbes d'apprentissage — {env_name}", fontsize=13, fontweight="bold")

        for exp_name, data in experiments.items():
            if env_name not in exp_name:
                continue

            algo  = "qlearning" if "qlearning" in exp_name else "sarsa"
            color = ALGO_COLOR[algo]
            label = ALGO_LABEL[algo]

            rewards = np.array([r["rewards"] for r in data["training"]])
            mean_r, std_r = np.mean(rewards, axis=0), np.std(rewards, axis=0)
            if len(mean_r) >= window:
                x = np.arange(len(_smooth(mean_r, window)))
                ms, ss = _smooth(mean_r, window), _smooth(std_r, window)
                axes[0].plot(x, ms, color=color, label=label, linewidth=2)
                axes[0].fill_between(x, ms - ss, ms + ss, color=color, alpha=0.18)

            falls = np.array([r["falls"] for r in data["training"]])
            mean_f, std_f = np.mean(falls, axis=0), np.std(falls, axis=0)
            if len(mean_f) >= window:
                x = np.arange(len(_smooth(mean_f, window)))
                mf, sf = _smooth(mean_f, window) * 100, _smooth(std_f, window) * 100
                axes[1].plot(x, mf, color=color, label=label, linewidth=2)
                axes[1].fill_between(x, mf - sf, mf + sf, color=color, alpha=0.18)

        axes[0].set_ylabel("Récompense par épisode")
        axes[0].legend(loc="lower right")
        axes[1].set_ylabel("Taux de chutes (%)")
        axes[1].set_xlabel(f"Épisode (fenêtre = {window})")
        axes[1].set_ylim(-5, 105)
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter())
        axes[1].legend(loc="upper right")

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
