import math
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
    """

    rng = np.random.default_rng(seed)
    env = StochasticChainEnv(seed=seed, slip_prob=slip_prob)

    q_table = np.zeros((env.length, 2), dtype=np.float64)

    for episode in range(train_episodes):
        state = env.reset()

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
    return ", ".join(f"{k}={v:.5g}" for k, v in cfg.items())
def main():
    rng = np.random.default_rng(123)

    n_random_configs = 30
    seeds = list(range(5))

    configs = [sample_config(rng) for _ in range(n_random_configs)]

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
        low_scores = eval_config(
            cfg=cfg,
            seeds=seeds,
            train_episodes=50,
            eval_episodes=50,
        )

        high_scores = eval_config(
            cfg=cfg,
            seeds=seeds,
            train_episodes=250,
            eval_episodes=50,
        )

        result = {
            "idx": idx,
            "cfg": cfg,
            "low_mean": float(np.mean(low_scores)),
            "low_std": float(np.std(low_scores)),
            "high_mean": float(np.mean(high_scores)),
            "high_std": float(np.std(high_scores)),
        }

        results.append(result)

        print(
            f"Finished config {idx}: "
            f"low_mean={result['low_mean']:.3f}, "
            f"high_mean={result['high_mean']:.3f}"
        )

    ranked_low = sorted(results, key=lambda x: x["low_mean"], reverse=True)
    ranked_high = sorted(results, key=lambda x: x["high_mean"], reverse=True)

    low_rank = {
        result["idx"]: rank
        for rank, result in enumerate(ranked_low, start=1)
    }

    high_rank = {
        result["idx"]: rank
        for rank, result in enumerate(ranked_high, start=1)
    }

    print()
    print("=" * 80)
    print("Week 10 Level 2 - Multi-fidelity Results")
    print("=" * 80)

    print()
    print("Experiment setting:")
    print("Algorithm: tabular Q-learning")
    print("Environment: stochastic chain environment")
    print("Low fidelity: 50 training episodes")
    print("High fidelity: 250 training episodes")
    print("Seeds: 0..4")
    print("Metric: mean greedy evaluation return")

    print()
    print("Top 10 configurations by LOW fidelity:")
    print(
        "low_rank high_rank config_id low_mean low_std "
        "high_mean high_std alpha gamma eps_start eps_decay eps_min"
    )

    for result in ranked_low[:10]:
        cfg = result["cfg"]

        print(
            f"{low_rank[result['idx']]:>8} "
            f"{high_rank[result['idx']]:>9} "
            f"{result['idx']:>9} "
            f"{result['low_mean']:.3f} "
            f"{result['low_std']:.3f} "
            f"{result['high_mean']:.3f} "
            f"{result['high_std']:.3f} "
            f"{cfg['alpha']:.6f} "
            f"{cfg['gamma']:.6f} "
            f"{cfg['eps_start']:.6f} "
            f"{cfg['eps_decay']:.6f} "
            f"{cfg['eps_min']:.6f}"
        )

    print()
    print("Top 10 configurations by HIGH fidelity:")
    print(
        "high_rank low_rank config_id low_mean low_std "
        "high_mean high_std alpha gamma eps_start eps_decay eps_min"
    )

    for result in ranked_high[:10]:
        cfg = result["cfg"]

        print(
            f"{high_rank[result['idx']]:>9} "
            f"{low_rank[result['idx']]:>8} "
            f"{result['idx']:>9} "
            f"{result['low_mean']:.3f} "
            f"{result['low_std']:.3f} "
            f"{result['high_mean']:.3f} "
            f"{result['high_std']:.3f} "
            f"{cfg['alpha']:.6f} "
            f"{cfg['gamma']:.6f} "
            f"{cfg['eps_start']:.6f} "
            f"{cfg['eps_decay']:.6f} "
            f"{cfg['eps_min']:.6f}"
        )

    low_best = ranked_low[0]
    high_best = ranked_high[0]

    print()
    print("Main comparison:")

    print(
        f"Best LOW-fidelity config: config {low_best['idx']} | "
        f"low_mean={low_best['low_mean']:.3f} | "
        f"high_mean={low_best['high_mean']:.3f} | "
        f"high_rank={high_rank[low_best['idx']]}"
    )

    print(
        f"Best HIGH-fidelity config: config {high_best['idx']} | "
        f"low_mean={high_best['low_mean']:.3f} | "
        f"high_mean={high_best['high_mean']:.3f} | "
        f"low_rank={low_rank[high_best['idx']]}"
    )

    print()
    print("Conclusion:")

    if low_best["idx"] != high_best["idx"]:
        print(
            "Early performance was misleading: the best low-fidelity "
            "configuration was not the best high-fidelity configuration."
        )
    else:
        print(
            "The best low-fidelity configuration was also the best "
            "high-fidelity configuration in this run."
        )

    print(
        "This checks whether partial evaluations can influence "
        "multi-fidelity optimization negatively."
    )


if __name__ == "__main__":
    main()