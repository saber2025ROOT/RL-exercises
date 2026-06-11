"""
Plot SAC training results for Week 6 Level 3.

This script loads the .npz result file saved by sac.py and creates a plot
of average evaluation return over environment steps.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_sac_results(result_path: str, output_path: str) -> None:
    data = np.load(result_path)

    steps = data["steps"]
    returns = data["returns"]
    stds = data["stds"]

    plt.figure(figsize=(8, 5))

    plt.plot(steps, returns, label="SAC")
    plt.fill_between(
        steps,
        returns - stds,
        returns + stds,
        alpha=0.2,
        label="Evaluation std",
    )

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.title("SAC on LunarLander-v3 continuous")
    plt.legend()
    plt.grid(True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SAC result curves.")

    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
        help="Path to the .npz result file saved by sac.py.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="sac_lunarlander_results.png",
        help="Path where the output plot should be saved.",
    )

    args = parser.parse_args()

    plot_sac_results(args.result_path, args.output_path)


if __name__ == "__main__":
    main()
