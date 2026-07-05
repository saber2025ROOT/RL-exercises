import math
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np


class StochasticChainEnv:
    """
    A very small stochastic RL environment.

    The agent starts in the middle of a chain.
    Action 0 moves left, action 1 moves right.
    With slip_prob, the action is flipped.

    Reaching the right end gives reward 1.
    Reaching the left end gives reward 0.
    """

    def __init__(self, length=11, max_steps=25, slip_prob=0.20, seed=0):
        self.length = length
        self.max_steps = max_steps
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.length // 2
        self.steps = 0
        return self.state

    def step(self, action):
        # Slip means the environment flips the selected action.
        if self.rng.random() < self.slip_prob:
            action = 1 - action

        if action == 1:
            self.state += 1
        else:
            self.state -= 1

        self.state = int(np.clip(self.state, 0, self.length - 1))
        self.steps += 1

        done = (
            self.state == 0
            or self.state == self.length - 1
            or self.steps >= self.max_steps
        )

        reward = 1.0 if self.state == self.length - 1 else 0.0

        return self.state, reward, done, {}


def sample_config(rng):
    """
    Randomly sample one hyperparameter configuration.

    This is our HPO method: random search.
    """

    return {
        "alpha": float(10 ** rng.uniform(math.log10(0.03), math.log10(0.8))),
        "gamma": float(rng.uniform(0.85, 0.999)),
        "eps_start": float(rng.uniform(0.5, 1.0)),
        "eps_decay": float(10 ** rng.uniform(math.log10(0.002), math.log10(0.08))),
        "eps_min": float(rng.uniform(0.01, 0.25)),
    }


def train_qlearning(
    cfg,
    seed,
    train_episodes=250,
    eval_episodes=50,
    slip_prob=0.20,
):
    """
    Train tabular Q-learning with one hyperparameter configuration.
    Then evaluate the learned greedy policy.

    The returned score is the mean evaluation return.
    Since the environment reward is either 0 or 1, the score can also be
    interpreted as success rate.
    """

    rng = np.random.default_rng(seed)
    env = StochasticChainEnv(seed=seed, slip_prob=slip_prob)

    q_table = np.zeros((env.length, 2), dtype=np.float64)

    for episode in range(train_episodes):
        state = env.reset()

        # Exponential epsilon decay.
        epsilon = max(
            cfg["eps_min"],
            cfg["eps_start"] * math.exp(-cfg["eps_decay"] * episode),
        )

        done = False

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done, _ = env.step(action)

            target = reward
            if not done:
                target += cfg["gamma"] * np.max(q_table[next_state])

            q_table[state, action] += cfg["alpha"] * (
                target - q_table[state, action]
            )

            state = next_state

    # Evaluate greedily with a different environment random seed.
    eval_env = StochasticChainEnv(seed=seed + 10000, slip_prob=slip_prob)
    scores = []

    for episode in range(eval_episodes):
        state = eval_env.reset()
        done = False
        total_return = 0.0

        while not done:
            best_actions = np.flatnonzero(q_table[state] == q_table[state].max())
            action = int(best_actions[(seed + episode) % len(best_actions)])

            next_state, reward, done, _ = eval_env.step(action)
            total_return += reward
            state = next_state

        scores.append(total_return)

    return float(np.mean(scores))


def eval_config(cfg, seeds, train_episodes, eval_episodes=50, slip_prob=0.20):
    """
    Evaluate one configuration over several random seeds.
    """

    scores = []

    for seed in seeds:
        score = train_qlearning(
            cfg=cfg,
            seed=seed,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            slip_prob=slip_prob,
        )
        scores.append(score)

    return scores


def format_config(cfg):
    return ", ".join(f"{key}={value:.5g}" for key, value in cfg.items())


def main():
    rng = np.random.default_rng(123)

    # HPO budget.
    n_random_configs = 30

    # We tune on only a few seeds.
    tuning_seeds = list(range(5))

    # Then we test on unseen seeds.
    test_seeds = list(range(100, 130))

    configs = [sample_config(rng) for _ in range(n_random_configs)]

    # Add one default-like baseline configuration.
    default_config = {
        "alpha": 0.1,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_decay": 0.01,
        "eps_min": 0.05,
    }
    configs.append(default_config)

    results = []

    for idx, cfg in enumerate(configs):
        tuning_scores = eval_config(
            cfg=cfg,
            seeds=tuning_seeds,
            train_episodes=250,
            eval_episodes=50,
        )

        result = {
            "idx": idx,
            "cfg": cfg,
            "tune_scores": tuning_scores,
            "tune_mean": float(np.mean(tuning_scores)),
            "tune_std": float(np.std(tuning_scores)),
        }

        results.append(result)

        print(f"Finished config {idx}: tune_mean={result['tune_mean']:.3f}")

    results_sorted = sorted(results, key=lambda x: x["tune_mean"], reverse=True)

    best = results_sorted[0]
    median_config = results_sorted[len(results_sorted) // 2]
    worst = results_sorted[-1]

    # Evaluate top 5, median, and worst on unseen seeds.
    configs_to_test = results_sorted[:5] + [median_config, worst]

    for result in configs_to_test:
        test_scores = eval_config(
            cfg=result["cfg"],
            seeds=test_seeds,
            train_episodes=250,
            eval_episodes=50,
        )

        result["test_scores"] = test_scores
        result["test_mean"] = float(np.mean(test_scores))
        result["test_std"] = float(np.std(test_scores))

    default_result = next(r for r in results if r["idx"] == n_random_configs)

    if "test_mean" not in default_result:
        test_scores = eval_config(
            cfg=default_result["cfg"],
            seeds=test_seeds,
            train_episodes=250,
            eval_episodes=50,
        )

        default_result["test_scores"] = test_scores
        default_result["test_mean"] = float(np.mean(test_scores))
        default_result["test_std"] = float(np.std(test_scores))


    output_dir = Path(__file__).resolve().parent / "results_l1"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_txt_path = output_dir / "l1_hpo_generalization_results.txt"
    csv_path = output_dir / "l1_hpo_generalization_results.csv"
    plot_path = output_dir / "l1_hpo_generalization_plot.png"

    # -------------------------
    # Save raw TXT results
    # -------------------------
    with open(raw_txt_path, "w", encoding="utf-8") as file:
        file.write("Week 10 Level 1 - Raw HPO Generalization Results\n")
        file.write("=" * 60 + "\n")
        file.write("Generalization type: across random seeds\n")
        file.write("Tuning seeds: 0..4\n")
        file.write("Test seeds: 100..129\n")
        file.write("HPO method: random search\n")
        file.write("Algorithm: tabular Q-learning\n")
        file.write("Environment: stochastic chain environment\n")
        file.write("Metric: mean greedy evaluation return\n")
        file.write("\n")

        file.write("Tested configurations:\n")
        file.write(
            "rank,config_id,tune_mean,tune_std,test_mean,test_std,"
            "alpha,gamma,eps_start,eps_decay,eps_min\n"
        )

        for rank, result in enumerate(configs_to_test, start=1):
            cfg = result["cfg"]
            file.write(
                f"{rank},"
                f"{result['idx']},"
                f"{result['tune_mean']:.3f},"
                f"{result['tune_std']:.3f},"
                f"{result['test_mean']:.3f},"
                f"{result['test_std']:.3f},"
                f"{cfg['alpha']:.6f},"
                f"{cfg['gamma']:.6f},"
                f"{cfg['eps_start']:.6f},"
                f"{cfg['eps_decay']:.6f},"
                f"{cfg['eps_min']:.6f}\n"
            )

        file.write("\nDefault configuration:\n")
        cfg = default_result["cfg"]
        file.write(
            f"config_id={default_result['idx']}, "
            f"tune_mean={default_result['tune_mean']:.3f}, "
            f"tune_std={default_result['tune_std']:.3f}, "
            f"test_mean={default_result['test_mean']:.3f}, "
            f"test_std={default_result['test_std']:.3f}, "
            f"alpha={cfg['alpha']:.6f}, "
            f"gamma={cfg['gamma']:.6f}, "
            f"eps_start={cfg['eps_start']:.6f}, "
            f"eps_decay={cfg['eps_decay']:.6f}, "
            f"eps_min={cfg['eps_min']:.6f}\n"
        )

    # -------------------------
    # Save CSV table
    # -------------------------
    fieldnames = [
        "config_id",
        "rank_by_tuning",
        "tune_mean",
        "tune_std",
        "test_mean",
        "test_std",
        "alpha",
        "gamma",
        "eps_start",
        "eps_decay",
        "eps_min",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for rank, result in enumerate(results_sorted, start=1):
            cfg = result["cfg"]

            writer.writerow(
                {
                    "config_id": result["idx"],
                    "rank_by_tuning": rank,
                    "tune_mean": result["tune_mean"],
                    "tune_std": result["tune_std"],
                    "test_mean": result.get("test_mean", ""),
                    "test_std": result.get("test_std", ""),
                    "alpha": cfg["alpha"],
                    "gamma": cfg["gamma"],
                    "eps_start": cfg["eps_start"],
                    "eps_decay": cfg["eps_decay"],
                    "eps_min": cfg["eps_min"],
                }
            )

    # -------------------------
    # Save plot
    # -------------------------
    tested_results = [result for result in results_sorted if "test_mean" in result]

    labels = [f"cfg {result['idx']}" for result in tested_results]
    tune_means = [result["tune_mean"] for result in tested_results]
    test_means = [result["test_mean"] for result in tested_results]

    x = np.arange(len(tested_results))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, tune_means, width, label="Tuning mean")
    plt.bar(x + width / 2, test_means, width, label="Test mean")

    plt.xticks(x, labels)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Configuration")
    plt.ylabel("Mean return")
    plt.title("HPO generalization across random seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print()
    print(f"Saved raw TXT results to: {raw_txt_path}")
    print(f"Saved CSV results to: {csv_path}")
    print(f"Saved plot to: {plot_path}")




if __name__ == "__main__":
    main()