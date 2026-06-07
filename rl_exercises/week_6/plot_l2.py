from glob import glob

import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1]

METHODS = {
    "Actor-Critic": "value_LunarLander-v3_{seed}_results.npz",
    "PPO Vanilla": "ppo_vanilla_LunarLander-v3_{seed}_results.npz",
    "PPO Enhanced": "ppo_enhanced_LunarLander-v3_{seed}_results.npz",
}


def find_file(filename):
    matches = glob(f"outputs/**/{filename}", recursive=True)
    if not matches:
        print(f"Missing: {filename}")
        return None
    return matches[0]


plt.figure(figsize=(8, 5))

for name, pattern in METHODS.items():
    runs = []
    steps = None

    for seed in SEEDS:
        path = find_file(pattern.format(seed=seed))
        if path is None:
            continue

        data = np.load(path)
        steps = data["steps"]
        runs.append(data["returns"])
        print(f"Loaded {name}: {path}")

    if not runs:
        continue

    runs = np.stack(runs, axis=0)

    mean = runs.mean(axis=0)
    lower = runs.min(axis=0)
    upper = runs.max(axis=0)

    plt.plot(steps, mean, label=name)
    plt.fill_between(steps, lower, upper, alpha=0.2)

plt.xlabel("Environment steps")
plt.ylabel("Average evaluation return")
plt.title("LunarLander-v3: PPO vs Actor-Critic")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("l2_ppo_vs_actor_critic.png", dpi=200)
plt.close()

print("Saved: l2_ppo_vs_actor_critic.png")
