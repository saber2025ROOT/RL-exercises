from glob import glob

import matplotlib.pyplot as plt
import numpy as np

ENVS = ["CartPole-v1", "LunarLander-v3"]
BASELINES = ["none", "avg", "value", "gae"]
SEEDS = [0, 1]


def find_file(baseline, env_name, seed):
    pattern = f"outputs/**/{baseline}_{env_name}_{seed}_results.npz"
    matches = glob(pattern, recursive=True)

    if len(matches) == 0:
        return None

    return matches[0]


def plot_env(env_name):
    plt.figure(figsize=(8, 5))
    found_anything = False

    for baseline in BASELINES:
        runs = []
        steps = None

        for seed in SEEDS:
            path = find_file(baseline, env_name, seed)

            if path is None:
                print(f"Missing: {baseline}_{env_name}_{seed}_results.npz")
                continue

            print(f"Loading: {path}")
            data = np.load(path)

            steps = data["steps"]
            returns = data["returns"]

            print(
                baseline,
                env_name,
                "seed",
                seed,
                "steps shape",
                steps.shape,
                "returns shape",
                returns.shape,
            )

            runs.append(returns)

        if len(runs) == 0:
            continue

        found_anything = True

        runs = np.stack(runs, axis=0)

        mean = runs.mean(axis=0)
        lower = runs.min(axis=0)
        upper = runs.max(axis=0)

        plt.plot(steps, mean, label=baseline)
        plt.fill_between(steps, lower, upper, alpha=0.2)

    if not found_anything:
        print(f"No data found for {env_name}")
        return

    plt.xlabel("Environment steps")
    plt.ylabel("Average evaluation return")
    plt.title(f"{env_name}: Actor-Critic baselines")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = f"{env_name}_baselines.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")


for env in ENVS:
    plot_env(env)
