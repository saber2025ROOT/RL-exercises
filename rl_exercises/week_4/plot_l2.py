import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import plot_utils
from rliable import metrics


def load_training_data():
    files = sorted(glob.glob("outputs/*/*/training_data.json"))[-5:]

    curves = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        curves.append(data["rewards"])

    min_len = min(len(curve) for curve in curves)
    curves = np.array([curve[:min_len] for curve in curves])

    return curves


def main():
    curves = load_training_data()

    # Shape: (num_seeds, num_points)
    frames = np.arange(curves.shape[1])

    # Plot mean training curve with CI
    mean_curve = np.mean(curves, axis=0)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)

    plt.figure()
    plt.plot(frames, mean_curve, label="DQN mean")
    plt.fill_between(frames, lower, upper, alpha=0.2)
    plt.xlabel("Evaluation Points")
    plt.ylabel("Mean Reward")
    plt.title("DQN Training Curve Across 5 Seeds")
    plt.legend()
    plt.savefig("l2_training_curve_ci.png")
    plt.close()

    final_scores = curves[:, -1]
    score_dict = {"DQN": final_scores[:, None]}

    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_mean(x),
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_optimality_gap(x, gamma=500),
        ]
    )

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        score_dict,
        aggregate_func,
        reps=2000,
    )

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_cis,
        metric_names=["Mean", "Median", "IQM", "Optimality Gap"],
        algorithms=["DQN"],
    )

    fig.set_size_inches(14, 4)
    fig.tight_layout()

    plt.savefig("l2_aggregate_metrics.png", bbox_inches="tight")
    plt.close()

    print("Saved:")
    print("l2_training_curve_ci.png")
    print("l2_aggregate_metrics.png")


if __name__ == "__main__":
    main()