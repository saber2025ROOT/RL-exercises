import re

import matplotlib.pyplot as plt


def read_ppo(filename):
    steps = []
    returns = []

    with open(filename) as f:
        for line in f:
            if "[Eval" in line:
                m = re.search(r"Step\s+(\d+).*?AvgReturn\s+([0-9.]+)", line)
                if m:
                    steps.append(int(m.group(1)))
                    returns.append(float(m.group(2)))
    return steps, returns


def read_dyna(filename):
    steps = []
    returns = []

    with open(filename) as f:
        for line in f:
            if "[Eval" in line:
                m = re.search(r"Real Steps\s+(\d+).*?AvgReturn\s+([0-9.]+)", line)
                if m:
                    steps.append(int(m.group(1)))
                    returns.append(float(m.group(2)))
    return steps, returns


ppo_steps, ppo_returns = read_ppo("rl_exercises/week_9/ppo_15k.txt")
dyna_steps, dyna_returns = read_dyna("rl_exercises/week_9/dyna_ppo_15k.txt")

plt.figure(figsize=(8, 5))
plt.plot(ppo_steps, ppo_returns, marker="o", label="PPO")
plt.plot(dyna_steps, dyna_returns, marker="s", label="Dyna-PPO")

plt.xlabel("Real Environment Steps")
plt.ylabel("Average Return")
plt.title("Sample Efficiency")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("sample_efficiency.png", dpi=300)
plt.show()
