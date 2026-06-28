# rl_exercises/week_8/plot_l1_seeding.py

from typing import Dict, List

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "rl_exercises/week_8/results_l1"
INPUT_FILE = os.path.join(RESULTS_DIR, "l1_all_seed_results.csv")


SEED_GROUPS: Dict[str, List[int]] = {
    "Low A (5 seeds)": [0, 1, 2, 3, 4],
    "Low B (5 seeds)": [10, 11, 12, 13, 14],
    "Low C (5 seeds)": [25, 26, 27, 28, 29],
    "Medium (10 seeds)": list(range(10)),
    "Large (30 seeds)": list(range(30)),
}


def compute_iqm(values: np.ndarray) -> float:
    """Compute the interquartile mean (IQM)."""
    values = np.sort(values)
    n = len(values)

    lower = int(np.floor(0.25 * n))
    upper = int(np.ceil(0.75 * n))

    trimmed_values = values[lower:upper]
    return float(np.mean(trimmed_values))


def compute_statistics(df: pd.DataFrame, seeds: List[int]) -> pd.DataFrame:
    """Compute mean, median, IQM, std, SE and 95% CI for one seed group."""
    group_df = df[df["seed"].isin(seeds)]

    rows = []

    for step, step_df in group_df.groupby("step"):
        values = step_df["eval_return"].to_numpy(dtype=float)
        n = len(values)

        mean = float(np.mean(values))
        median = float(np.median(values))
        iqm = compute_iqm(values)

        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        ci95 = 1.96 * se

        rows.append(
            {
                "step": step,
                "n": n,
                "mean": mean,
                "median": median,
                "iqm": iqm,
                "std": std,
                "se": se,
                "ci95": ci95,
                "ci95_width": 2 * ci95,
            }
        )

    return pd.DataFrame(rows).sort_values("step")


def plot_metric(
    stats_by_group: Dict[str, pd.DataFrame],
    metric: str,
    ylabel: str,
    title: str,
    output_name: str,
    include_ci: bool = False,
) -> None:
    """Plot one metric over training steps for all seed groups."""
    plt.figure(figsize=(10, 6))

    for group_name, stats_df in stats_by_group.items():
        x = stats_df["step"].to_numpy()
        y = stats_df[metric].to_numpy()

        plt.plot(x, y, label=group_name)

        if include_ci:
            ci = stats_df["ci95"].to_numpy()
            plt.fill_between(x, y - ci, y + ci, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_low_seed_comparison(stats_by_group: Dict[str, pd.DataFrame]) -> None:
    """Plot only the three low-seed groups to show seed-set sensitivity."""
    low_groups = ["Low A (5 seeds)", "Low B (5 seeds)", "Low C (5 seeds)"]

    plt.figure(figsize=(10, 6))

    for group_name in low_groups:
        stats_df = stats_by_group[group_name]
        x = stats_df["step"].to_numpy()
        y = stats_df["mean"].to_numpy()
        ci = stats_df["ci95"].to_numpy()

        plt.plot(x, y, label=group_name)
        plt.fill_between(x, y - ci, y + ci, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Mean evaluation return")
    plt.title("Low-seed comparison: effect of different seed sets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "l1_low_seed_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"seed", "step", "eval_return"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    stats_by_group = {
        group_name: compute_statistics(df, seeds)
        for group_name, seeds in SEED_GROUPS.items()
    }

    combined_stats = []
    for group_name, stats_df in stats_by_group.items():
        temp = stats_df.copy()
        temp["group"] = group_name
        combined_stats.append(temp)

    combined_stats_df = pd.concat(combined_stats, ignore_index=True)
    stats_output_path = os.path.join(RESULTS_DIR, "l1_seed_statistics.csv")
    combined_stats_df.to_csv(stats_output_path, index=False)
    print(f"Saved statistics: {stats_output_path}")

    plot_metric(
        stats_by_group,
        metric="mean",
        ylabel="Mean evaluation return",
        title="Mean reward over time for different numbers of seeds",
        output_name="l1_mean_reward.png",
        include_ci=True,
    )

    plot_metric(
        stats_by_group,
        metric="median",
        ylabel="Median evaluation return",
        title="Median reward over time for different numbers of seeds",
        output_name="l1_median_reward.png",
        include_ci=False,
    )

    plot_metric(
        stats_by_group,
        metric="iqm",
        ylabel="IQM evaluation return",
        title="IQM reward over time for different numbers of seeds",
        output_name="l1_iqm_reward.png",
        include_ci=False,
    )

    plot_metric(
        stats_by_group,
        metric="std",
        ylabel="Standard deviation",
        title="Standard deviation over time",
        output_name="l1_standard_deviation.png",
        include_ci=False,
    )

    plot_metric(
        stats_by_group,
        metric="se",
        ylabel="Standard error",
        title="Standard error over time",
        output_name="l1_standard_error.png",
        include_ci=False,
    )

    plot_metric(
        stats_by_group,
        metric="ci95_width",
        ylabel="95% confidence interval width",
        title="95% confidence interval width over time",
        output_name="l1_ci95_width.png",
        include_ci=False,
    )

    plot_low_seed_comparison(stats_by_group)


if __name__ == "__main__":
    main()
