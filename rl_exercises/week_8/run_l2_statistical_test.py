# rl_exercises/week_8/run_l2_statistical_test.py

import os
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from rl_exercises.week_6.actor_critic import ActorCriticAgent
from rl_exercises.week_6.ppo import PPOAgent, set_seed


RESULTS_DIR = "rl_exercises/week_8/results_l2"
ENV_NAME = "CartPole-v1"

TOTAL_STEPS = 50_000
EVAL_EPISODES = 10
SEEDS = list(range(10))

ALPHA = 0.05


def evaluate_ppo(
    agent: PPOAgent,
    env_name: str,
    seed: int,
    num_episodes: int = 10,
) -> float:
    eval_env = gym.make(env_name)
    returns = []

    for episode_idx in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + 10_000 + episode_idx)
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.predict(state)

            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

    eval_env.close()
    return float(np.mean(returns))


def evaluate_actor_critic(
    agent: ActorCriticAgent,
    env_name: str,
    seed: int,
    num_episodes: int = 10,
) -> float:
    eval_env = gym.make(env_name)
    returns = []

    agent.policy.eval()

    with torch.no_grad():
        for episode_idx in range(num_episodes):
            state, _ = eval_env.reset(seed=seed + 20_000 + episode_idx)
            done = False
            total_reward = 0.0

            while not done:
                action, _ = agent.predict_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward

            returns.append(total_reward)

    agent.policy.train()
    eval_env.close()

    return float(np.mean(returns))


def train_ppo_one_seed(seed: int) -> float:
    env = gym.make(ENV_NAME)
    set_seed(env, seed)

    agent = PPOAgent(
        env=env,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=seed,
        hidden_size=128,
    )

    step_count = 0

    while step_count < TOTAL_STEPS:
        state, _ = env.reset(seed=seed + step_count)
        done = False
        trajectory: List[Any] = []

        while not done and step_count < TOTAL_STEPS:
            action, logp, ent, value = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )

            state = next_state
            step_count += 1

        agent.update(trajectory)

    final_return = evaluate_ppo(
        agent=agent,
        env_name=ENV_NAME,
        seed=seed,
        num_episodes=EVAL_EPISODES,
    )

    env.close()
    return final_return


def train_actor_critic_one_seed(seed: int) -> float:
    env = gym.make(ENV_NAME)
    set_seed(env, seed)

    agent = ActorCriticAgent(
        env=env,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        seed=seed,
        hidden_size=128,
        baseline_type="gae",
        baseline_decay=0.9,
    )

    step_count = 0

    while step_count < TOTAL_STEPS:
        state, _ = env.reset(seed=seed + step_count)
        done = False
        trajectory: List[Tuple] = []

        while not done and step_count < TOTAL_STEPS:
            action, logp = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append(
                (state, action, float(reward), next_state, done, logp)
            )

            state = next_state
            step_count += 1

        agent.update_agent(trajectory)

    final_return = evaluate_actor_critic(
        agent=agent,
        env_name=ENV_NAME,
        seed=seed,
        num_episodes=EVAL_EPISODES,
    )

    env.close()
    return final_return


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    for seed in SEEDS:
        print(f"Training PPO with seed {seed}")
        ppo_return = train_ppo_one_seed(seed)

        rows.append(
            {
                "algorithm": "PPO",
                "seed": seed,
                "final_eval_return": ppo_return,
            }
        )

        print(f"PPO seed {seed} final return: {ppo_return:.2f}")

    for seed in SEEDS:
        print(f"Training Actor-Critic GAE with seed {seed}")
        ac_return = train_actor_critic_one_seed(seed)

        rows.append(
            {
                "algorithm": "Actor-Critic GAE",
                "seed": seed,
                "final_eval_return": ac_return,
            }
        )

        print(f"Actor-Critic GAE seed {seed} final return: {ac_return:.2f}")

    df = pd.DataFrame(rows)

    results_path = os.path.join(RESULTS_DIR, "l2_algorithm_comparison_results.csv")
    df.to_csv(results_path, index=False)
    print(f"Saved raw results to {results_path}")

    ppo_values = df[df["algorithm"] == "PPO"]["final_eval_return"].to_numpy()
    ac_values = df[df["algorithm"] == "Actor-Critic GAE"]["final_eval_return"].to_numpy()

    statistic, p_value = mannwhitneyu(
        ppo_values,
        ac_values,
        alternative="two-sided",
    )

    summary_rows = [
        {
            "algorithm": "PPO",
            "mean": np.mean(ppo_values),
            "median": np.median(ppo_values),
            "std": np.std(ppo_values, ddof=1),
            "n": len(ppo_values),
        },
        {
            "algorithm": "Actor-Critic GAE",
            "mean": np.mean(ac_values),
            "median": np.median(ac_values),
            "std": np.std(ac_values, ddof=1),
            "n": len(ac_values),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(RESULTS_DIR, "l2_statistical_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    test_path = os.path.join(RESULTS_DIR, "l2_mann_whitney_test.txt")

    with open(test_path, "w", encoding="utf-8") as f:
        f.write("Level 2 Statistical Test\n")
        f.write("========================\n\n")
        f.write(f"Environment: {ENV_NAME}\n")
        f.write("Algorithms: PPO vs Actor-Critic with GAE\n")
        f.write(f"Seeds per algorithm: {len(SEEDS)}\n")
        f.write(f"Metric: final evaluation return over {EVAL_EPISODES} episodes\n")
        f.write("Test: Mann-Whitney U-test\n")
        f.write(f"Significance level alpha: {ALPHA}\n\n")

        f.write("PPO final returns:\n")
        f.write(str(ppo_values.tolist()) + "\n\n")

        f.write("Actor-Critic GAE final returns:\n")
        f.write(str(ac_values.tolist()) + "\n\n")

        f.write(f"U statistic: {statistic:.4f}\n")
        f.write(f"p-value: {p_value:.6f}\n\n")

        if p_value < ALPHA:
            f.write("Result: Reject the null hypothesis.\n")
            f.write(
                "Interpretation: The observed difference between the two algorithms is statistically significant at alpha = 0.05.\n"
            )
        else:
            f.write("Result: Fail to reject the null hypothesis.\n")
            f.write(
                "Interpretation: The test does not provide enough evidence for a statistically significant difference at alpha = 0.05.\n"
            )

    print(f"Saved test result to {test_path}")


if __name__ == "__main__":
    main()