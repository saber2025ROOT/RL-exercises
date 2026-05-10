import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def load_latest_runs(num_runs: int = 20):
    """
    Load the latest Hydra output folders and group them
    according to the used DQN configuration.

    We expect 4 configurations:
    - Base DQN
    - DQN + PER
    - DQN + Double
    - DQN + PER + Double

    Each configuration should contain 5 different seeds.
    """

    # Get the newest Hydra run directories
    run_dirs = sorted(glob.glob("outputs/*/*"))[-num_runs:]

    # Dictionary that stores all reward curves for each configuration
    grouped = {
        "Base DQN": [],
        "DQN + PER": [],
        "DQN + Double": [],
        "DQN + PER + Double": [],
    }

    for run_dir in run_dirs:

        # Training rewards produced during training
        data_path = os.path.join(run_dir, "training_data.json")

        # Hydra configuration file of this run
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")

        # Skip invalid runs
        if not os.path.exists(data_path) or not os.path.exists(config_path):
            continue

        # Load training rewards
        with open(data_path, "r") as f:
            data = json.load(f)

        # Load Hydra config
        cfg = OmegaConf.load(config_path)

        # Read which Rainbow components were enabled
        use_per = bool(cfg.agent.use_prioritized_replay)
        use_double = bool(cfg.agent.use_double_dqn)

        # Determine experiment name based on enabled features
        if not use_per and not use_double:
            name = "Base DQN"

        elif use_per and not use_double:
            name = "DQN + PER"

        elif not use_per and use_double:
            name = "DQN + Double"

        else:
            name = "DQN + PER + Double"

        # Store reward curve
        grouped[name].append(data["rewards"])

    return grouped


def align_curves(grouped):
    """
    Different runs may contain slightly different curve lengths.

    To compare them fairly, we truncate all curves to the
    minimum available length.
    """

    # Find shortest curve among all runs
    min_len = min(
        len(curve)
        for curves in grouped.values()
        for curve in curves
    )

    aligned = {}

    for name, curves in grouped.items():

        # Keep only the first min_len values
        aligned[name] = np.array([
            curve[:min_len]
            for curve in curves
        ])

    return aligned


def plot_training_curves(grouped):
    """
    Plot mean training curves with confidence intervals.

    The shaded region represents the variability across seeds.
    """

    plt.figure(figsize=(10, 6))

    for name, curves in grouped.items():

        # Mean performance across seeds
        mean_curve = np.mean(curves, axis=0)

        # 95% confidence interval using percentiles
        lower = np.percentile(curves, 2.5, axis=0)
        upper = np.percentile(curves, 97.5, axis=0)

        # Plot average curve
        plt.plot(mean_curve, label=name)

        # Plot uncertainty region
        plt.fill_between(
            np.arange(len(mean_curve)),
            lower,
            upper,
            alpha=0.15,
        )

    plt.xlabel("Evaluation Points")
    plt.ylabel("Mean Reward")

    plt.title("Level 3 Ablation: DQN Variants Across Seeds")

    plt.legend()

    plt.savefig(
        "l3_training_curves.png",
        bbox_inches="tight",
    )

    plt.close()


def plot_aggregate_metrics(grouped):
    """
    Compute robust aggregate statistics using RLiable.

    Metrics:
    - Mean
    - Median
    - IQM
    - Optimality Gap
    """

    # Use final performance of each run
    final_scores = {
        name: curves[:, -1][:, None]
        for name, curves in grouped.items()
    }

    # Function that computes all aggregate metrics
    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),

            # IQM is more robust against outliers
            metrics.aggregate_iqm(x),

            # Distance from optimal score
            metrics.aggregate_optimality_gap(x, gamma=500),
        ]
    )

    # Compute confidence intervals via bootstrapping
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        final_scores,
        aggregate_func,
        reps=2000,
    )

    # Create RLiable interval plot
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_cis,
        metric_names=[
            "Mean",
            "Median",
            "IQM",
            "Optimality Gap",
        ],
        algorithms=list(grouped.keys()),
    )

    fig.set_size_inches(12, 6)

    fig.tight_layout()

    plt.savefig(
        "l3_aggregate_metrics.png",
        bbox_inches="tight",
    )

    plt.close()


def main():
    """
    Main plotting pipeline for Level 3.
    """

    # Load all experiment runs
    grouped = load_latest_runs(num_runs=20)

    # Align curve lengths
    grouped = align_curves(grouped)

    # Print how many runs were found per configuration
    for name, curves in grouped.items():
        print(f"{name}: {len(curves)} runs")

    # Plot learning curves
    plot_training_curves(grouped)

    # Plot robust aggregate metrics
    plot_aggregate_metrics(grouped)

    print("Saved:")
    print("l3_training_curves.png")
    print("l3_aggregate_metrics.png")


if __name__ == "__main__":
    main()