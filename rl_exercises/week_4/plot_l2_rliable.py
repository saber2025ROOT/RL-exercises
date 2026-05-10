import os

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

FILES = [
    "outputs/2026-05-10/16-59-48/dqn_seed0.npz",
    "outputs/2026-05-10/17-00-30/dqn_seed1.npz",
    "outputs/2026-05-10/17-02-39/dqn_seed2.npz",
    "outputs/2026-05-10/17-02-53/dqn_seed3.npz",
    "outputs/2026-05-10/17-17-11/dqn_seed4.npz",
]

RESULTS_DIR = "results_l2"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_file(path):
    data = np.load(path)
    print("\nFile:", path)
    print("Keys:", data.files)

    rewards = np.asarray(data["rewards"], dtype=np.float32)

    if "frames" in data.files:
        frames = np.asarray(data["frames"], dtype=np.float32)
    else:
        frames = np.arange(len(rewards), dtype=np.float32)

    return frames, rewards


all_frames = []
all_rewards = []

for path in FILES:
    frames, rewards = load_file(path)
    all_frames.append(frames)
    all_rewards.append(rewards)

min_len = min(len(r) for r in all_rewards)

frames = np.array([f[:min_len] for f in all_frames], dtype=np.float32)
returns = np.array([r[:min_len] for r in all_rewards], dtype=np.float32)

x = np.mean(frames, axis=0)

print("\nreturns shape:", returns.shape)
print("frames shape:", frames.shape)


def moving_average(arr, window=5):
    out = np.zeros_like(arr, dtype=np.float32)

    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])

    return out


smoothed = np.array(
    [moving_average(r, window=5) for r in returns],
    dtype=np.float32,
)


# ============================================================
# Final score per seed
# ============================================================

final_scores = np.mean(returns[:, -5:], axis=1)

score_dict = {"DQN": final_scores[:, None]}

print("\nFinal score per seed:")
for i, score in enumerate(final_scores):
    print(f"Seed {i}: {score:.2f}")

print("Mean:", np.mean(final_scores))
print("Median:", np.median(final_scores))
print("Std:", np.std(final_scores))


# ============================================================
# 1. Clean aggregate metrics in ONE PNG
# ============================================================


def aggregate_func(scores):
    return np.array(
        [
            metrics.aggregate_median(scores),
            metrics.aggregate_iqm(scores),
            metrics.aggregate_mean(scores),
            metrics.aggregate_optimality_gap(scores, gamma=500.0),
        ]
    )


aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
    score_dict,
    aggregate_func,
    reps=5000,
)

metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]

metric_values = aggregate_scores["DQN"]

ci_low = aggregate_score_cis["DQN"][:, 0]
ci_high = aggregate_score_cis["DQN"][:, 1]

x_pos = np.arange(len(metric_names))

yerr = np.vstack(
    [
        metric_values - ci_low,
        ci_high - metric_values,
    ]
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.errorbar(
    x_pos,
    metric_values,
    yerr=yerr,
    fmt="o",
    capsize=6,
    linewidth=2,
)

ax.set_xticks(x_pos)
ax.set_xticklabels(metric_names)
ax.set_ylabel("Metric value")
ax.set_title("DQN CartPole-v1 Aggregate Metrics over 5 Seeds")
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "all_metrics_clean.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 2. IQM training curve
# ============================================================

curve_scores = smoothed[:, None, :]

curve_dict = {"DQN": curve_scores}


def iqm_over_time(scores):
    return np.array(
        [metrics.aggregate_iqm(scores[..., t]) for t in range(scores.shape[-1])]
    )


iqm_scores, iqm_cis = rly.get_interval_estimates(
    curve_dict,
    iqm_over_time,
    reps=5000,
)

fig, ax = plt.subplots(figsize=(9, 5))

plot_utils.plot_sample_efficiency_curve(
    x,
    iqm_scores,
    iqm_cis,
    algorithms=["DQN"],
    xlabel="Frames",
    ylabel="IQM Return",
    ax=ax,
)

ax.set_title("DQN CartPole-v1 IQM Training Curve over 5 Seeds")
fig.subplots_adjust(top=0.88, bottom=0.15)

plt.savefig(
    os.path.join(RESULTS_DIR, "iqm_training_curve.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 3. Performance profile
# ============================================================

thresholds = np.linspace(0, 500, 101)

score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict,
    thresholds,
    reps=5000,
)

fig, ax = plt.subplots(figsize=(9, 5))

plot_utils.plot_performance_profiles(
    score_distributions,
    thresholds,
    performance_profile_cis=score_distributions_cis,
    xlabel=r"CartPole return threshold $(\tau)$",
    ax=ax,
)

ax.set_title("DQN CartPole-v1 Performance Profile over 5 Seeds")
ax.set_ylabel(r"Fraction of runs with score $> \tau$")
fig.subplots_adjust(top=0.88, bottom=0.15)

plt.savefig(
    os.path.join(RESULTS_DIR, "performance_profile.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 4. Plain mean ± std baseline
# ============================================================

mean_curve = np.mean(smoothed, axis=0)
std_curve = np.std(smoothed, axis=0)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x, mean_curve, label="Mean return")
ax.fill_between(
    x,
    mean_curve - std_curve,
    mean_curve + std_curve,
    alpha=0.25,
    label="±1 std",
)

ax.set_xlabel("Frames")
ax.set_ylabel("Return")
ax.set_title("Plain Mean ± Std over 5 Seeds")
ax.legend()
ax.grid(True)

fig.subplots_adjust(top=0.88, bottom=0.15)

plt.savefig(
    os.path.join(RESULTS_DIR, "plain_mean_std.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 5. Save observations_l2.txt
# ============================================================

observations = f"""
What changes when using RLiable vs. plain averages?

Using plain averages only gives one mean curve across the five seeds. This can be misleading because DQN is sensitive to random seeds. In my experiment, one seed performed much worse than the others, so the mean alone does not show the full reliability of the algorithm.

RLiable gives more information. It computes robust metrics such as the Interquartile Mean (IQM), median, mean, and optimality gap. It also gives bootstrap confidence intervals. This makes the uncertainty across seeds visible instead of hiding it inside a single average number.

The performance profile also shows how many runs reach different return thresholds. This is useful because it shows the distribution of performance, not only one summary value.

Do you feel more confident in the results? Why or why not?

Yes, I feel more confident using RLiable than using only plain averages, because RLiable shows both performance and uncertainty. The confidence intervals show that the result is not perfectly stable across seeds. This is important because one seed achieved a much lower final score than the others.

However, I would still be cautious because I only used five seeds and one environment. The confidence intervals are quite wide, so more seeds would make the evaluation more reliable.

Final scores over seeds:
Seed 0: {final_scores[0]:.2f}
Seed 1: {final_scores[1]:.2f}
Seed 2: {final_scores[2]:.2f}
Seed 3: {final_scores[3]:.2f}
Seed 4: {final_scores[4]:.2f}

Mean final score: {np.mean(final_scores):.2f}
Median final score: {np.median(final_scores):.2f}
Standard deviation: {np.std(final_scores):.2f}
"""

with open(os.path.join(RESULTS_DIR, "observations_l2.txt"), "w") as f:
    f.write(observations.strip())


print("\nSaved:")
print(f"{RESULTS_DIR}/all_metrics_clean.png")
print(f"{RESULTS_DIR}/iqm_training_curve.png")
print(f"{RESULTS_DIR}/performance_profile.png")
print(f"{RESULTS_DIR}/plain_mean_std.png")
print(f"{RESULTS_DIR}/observations_l2.txt")
