# rl_exercises/week_8/run_l1_seeding.py

import os
import random
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import pandas as pd

from rl_exercises.week_6.ppo import PPOAgent, set_seed


def evaluate_policy(agent: PPOAgent, env_name: str, seed: int, num_episodes: int = 5) -> float:
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


def train_one_seed(
    seed: int,
    env_name: str = "CartPole-v1",
    total_steps: int = 50_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 5,
) -> List[dict]:
    env = gym.make(env_name)
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

    results = []
    step_count = 0

    while step_count < total_steps:
        state, _ = env.reset(seed=seed + step_count)
        done = False
        trajectory: List[Any] = []

        while not done and step_count < total_steps:
            action, logp, ent, val = agent.predict(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )

            state = next_state
            step_count += 1

            if step_count % eval_interval == 0:
                eval_return = evaluate_policy(
                    agent=agent,
                    env_name=env_name,
                    seed=seed,
                    num_episodes=eval_episodes,
                )

                results.append(
                    {
                        "seed": seed,
                        "step": step_count,
                        "eval_return": eval_return,
                    }
                )

                print(
                    f"Seed {seed:02d} | Step {step_count:6d} | Eval return {eval_return:7.2f}"
                )

        agent.update(trajectory)

    env.close()
    return results


def main() -> None:
    os.makedirs("rl_exercises/week_8/results_l1", exist_ok=True)

    all_results = []

    # You can increase this later if runtime is fine.
    seeds = list(range(30))

    for seed in seeds:
        seed_results = train_one_seed(seed=seed)
        all_results.extend(seed_results)

    df = pd.DataFrame(all_results)
    output_path = "rl_exercises/week_8/results_l1/l1_all_seed_results.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()