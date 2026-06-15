"""
Level 1 comparison: DQN baseline vs RND-DQN on CartPole-v1.

"""

from typing import List, Tuple

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent

OUTPUT_DIR = Path("rl_exercises/week_7/results_l1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    """Compute a simple moving average for smoother learning curves."""
    if len(values) == 0:
        return np.array([])

    values_np = np.array(values, dtype=np.float32)

    if len(values_np) < window:
        return values_np

    kernel = np.ones(window) / window
    return np.convolve(values_np, kernel, mode="valid")


def train_dqn_like_agent(
    agent,
    num_frames: int,
    use_rnd_bonus: bool,
) -> Tuple[List[int], List[float]]:
    """
    Train either DQNAgent or RNDDQNAgent and record extrinsic episode returns.

    For RND-DQN, the replay buffer receives reward + intrinsic_bonus.
    However, the plotted return remains the original environment reward.
    This makes the comparison fair.
    """
    reset_seed = agent.seed if hasattr(agent, "seed") else 0
    state, _ = agent.env.reset(seed=reset_seed)

    episode_return_ext = 0.0
    episode_returns_ext: List[float] = []
    episode_end_frames: List[int] = []

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, reward_ext, done, truncated, _ = agent.env.step(action)

        reward_for_training = float(reward_ext)

        if use_rnd_bonus:
            rnd_bonus = agent.get_rnd_bonus(next_state.astype(np.float32))
            reward_for_training += rnd_bonus

        agent.buffer.add(
            state,
            action,
            reward_for_training,
            next_state,
            done or truncated,
            {},
        )

        state = next_state
        episode_return_ext += float(reward_ext)

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            _ = agent.update_agent(batch)

            if use_rnd_bonus and frame % agent.rnd_update_freq == 0:
                _ = agent.update_rnd(batch)

        if done or truncated:
            episode_returns_ext.append(episode_return_ext)
            episode_end_frames.append(frame)

            state, _ = agent.env.reset()
            episode_return_ext = 0.0

    return episode_end_frames, episode_returns_ext


def run_single_seed(seed: int, dqn_cfg, rnd_cfg) -> pd.DataFrame:
    """Run DQN and RND-DQN for one seed and return a dataframe with results."""
    rows = []

    # -------------------------
    # Baseline DQN
    # -------------------------
    env_dqn = gym.make(dqn_cfg.env.name)
    set_seed(env_dqn, seed)

    dqn_agent = DQNAgent(
        env_dqn,
        buffer_capacity=dqn_cfg.agent.buffer_capacity,
        batch_size=dqn_cfg.agent.batch_size,
        lr=dqn_cfg.agent.learning_rate,
        gamma=dqn_cfg.agent.gamma,
        epsilon_start=dqn_cfg.agent.epsilon_start,
        epsilon_final=dqn_cfg.agent.epsilon_final,
        epsilon_decay=dqn_cfg.agent.epsilon_decay,
        target_update_freq=dqn_cfg.agent.target_update_freq,
        seed=seed,
    )

    frames, returns = train_dqn_like_agent(
        dqn_agent,
        num_frames=dqn_cfg.train.num_frames,
        use_rnd_bonus=False,
    )

    for frame, ret in zip(frames, returns):
        rows.append(
            {
                "algorithm": "DQN",
                "seed": seed,
                "frame": frame,
                "episode_return_ext": ret,
            }
        )

    env_dqn.close()

    # -------------------------
    # RND-DQN
    # -------------------------
    env_rnd = gym.make(rnd_cfg.env.name)
    set_seed(env_rnd, seed)

    rnd_agent = RNDDQNAgent(
        env_rnd,
        buffer_capacity=rnd_cfg.agent.buffer_capacity,
        batch_size=rnd_cfg.agent.batch_size,
        lr=rnd_cfg.agent.learning_rate,
        gamma=rnd_cfg.agent.gamma,
        epsilon_start=rnd_cfg.agent.epsilon_start,
        epsilon_final=rnd_cfg.agent.epsilon_final,
        epsilon_decay=rnd_cfg.agent.epsilon_decay,
        target_update_freq=rnd_cfg.agent.target_update_freq,
        seed=seed,
        rnd_hidden_size=rnd_cfg.rnd.hidden_size,
        rnd_lr=rnd_cfg.rnd.learning_rate,
        rnd_update_freq=rnd_cfg.rnd.update_freq,
        rnd_n_layers=rnd_cfg.rnd.n_layers,
        rnd_reward_weight=rnd_cfg.rnd.reward_weight,
    )

    frames, returns = train_dqn_like_agent(
        rnd_agent,
        num_frames=rnd_cfg.train.num_frames,
        use_rnd_bonus=True,
    )

    for frame, ret in zip(frames, returns):
        rows.append(
            {
                "algorithm": "RND-DQN",
                "seed": seed,
                "frame": frame,
                "episode_return_ext": ret,
            }
        )

    env_rnd.close()

    return pd.DataFrame(rows)


def plot_learning_curves(df: pd.DataFrame) -> None:
    """Create a mean learning curve over seeds."""
    plt.figure(figsize=(9, 5))

    for algorithm in sorted(df["algorithm"].unique()):
        algo_df = df[df["algorithm"] == algorithm]

        curves = []
        max_len = 0

        for seed in sorted(algo_df["seed"].unique()):
            seed_df = algo_df[algo_df["seed"] == seed].sort_values("frame")
            smoothed = moving_average(seed_df["episode_return_ext"].tolist(), window=10)
            curves.append(smoothed)
            max_len = max(max_len, len(smoothed))

        padded = np.full((len(curves), max_len), np.nan)

        for i, curve in enumerate(curves):
            padded[i, : len(curve)] = curve

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        x = np.arange(len(mean_curve))
        plt.plot(x, mean_curve, label=algorithm)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Episode index after smoothing")
    plt.ylabel("Extrinsic episode return")
    plt.title("DQN vs RND-DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "l1_rnd_dqn_comparison_curve.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved comparison plot to: {output_path}")


def main() -> None:
    dqn_cfg = OmegaConf.load("rl_exercises/configs/agent/dqn.yaml")
    rnd_cfg = OmegaConf.load("rl_exercises/configs/agent/rnd_dqn.yaml")

    seeds = [0, 1, 2, 3, 4]

    all_results = []

    for seed in seeds:
        print(f"Running seed {seed}...")
        seed_df = run_single_seed(seed, dqn_cfg, rnd_cfg)
        all_results.append(seed_df)

    results_df = pd.concat(all_results, ignore_index=True)

    csv_path = OUTPUT_DIR / "l1_rnd_dqn_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved raw results to: {csv_path}")

    plot_learning_curves(results_df)


if __name__ == "__main__":
    main()
