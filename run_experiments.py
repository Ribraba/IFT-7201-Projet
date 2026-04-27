import argparse
import json
import os
import time

import yaml

from src.envs    import make_env
from src.agents  import q_learning, sarsa, shielded_qlearning
from src.evaluate import evaluate_agent, evaluate_safe_path_rate


ALGO_FN = {"qlearning": q_learning, "sarsa": sarsa}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(exp, run_id):
    seed = exp["seed_base"] + run_id
    env  = make_env(exp["env"])

    Q, history = ALGO_FN[exp["algo"]](
        env       = env,
        gamma     = exp["gamma"],
        lr        = exp["lr"],
        episodes  = exp["episodes"],
        seed      = seed,
        eps_start = exp["eps_start"],
        eps_min   = exp["eps_min"],
        eps_decay = exp["eps_decay"],
        lr_min    = exp["lr_min"],
        lr_decay  = exp["lr_decay"],
    )

    eval_result = evaluate_agent(env, Q, n_episodes=500)
    if "hard" in exp["env"]:
        eval_result["safe_path_rate"] = evaluate_safe_path_rate(env, Q, n_episodes=500)

    return {
        "experiment":  exp["name"],
        "algo":        exp["algo"],
        "env":         exp["env"],
        "run_id":      run_id,
        "seed":        seed,
        "hyperparams": {k: exp[k] for k in
                        ["gamma", "lr", "episodes", "eps_start",
                         "eps_min", "eps_decay", "lr_min", "lr_decay"]},
        "training":    history,
        "evaluation":  eval_result,
        "Q":           Q.tolist(),
    }


def run_all(config_path, results_dir="results"):
    experiments = load_config(config_path)
    print(f"\n{len(experiments)} expériences → {results_dir}/\n")
    t0_total = time.time()

    for exp in experiments:
        exp_dir = os.path.join(results_dir, exp["name"])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[{exp['name']}]")

        for run_id in range(exp["n_runs"]):
            out_path = os.path.join(exp_dir, f"run_{run_id}.json")
            if os.path.exists(out_path):
                continue

            t0 = time.time()
            result = run_single(exp, run_id)

            with open(out_path, "w") as f:
                json.dump(result, f)

            sr = result["evaluation"]["success_rate"] * 100
            fr = result["evaluation"]["fall_rate"]    * 100
            print(f"  run {run_id:2d}  succès={sr:.1f}%  chutes={fr:.1f}%  ({time.time()-t0:.1f}s)")

        print()

    print(f"Terminé en {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config/config_baselines.yaml")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    run_all(args.config, args.results)
