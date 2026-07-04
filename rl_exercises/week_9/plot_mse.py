import re

import matplotlib.pyplot as plt

steps = []
state_mse = []
reward_mse = []

with open("rl_exercises/week_9/dyna_model.txt", "r") as f:
    for line in f:
        if "[Model]" in line:
            m = re.search(
                r"Step\s+(\d+).*?State MSE:\s*([0-9.eE+-]+),\s*Reward MSE:\s*([0-9.eE+-]+)",
                line,
            )
            if m:
                steps.append(int(m.group(1)))
                state_mse.append(float(m.group(2)))
                reward_mse.append(float(m.group(3)))

plt.figure(figsize=(7, 5))
plt.plot(steps, state_mse, marker="o", linewidth=2, label="State MSE")
plt.plot(steps, reward_mse, marker="s", linewidth=2, label="Reward MSE")

plt.xlabel("Real Environment Steps")
plt.ylabel("One-step MSE")
plt.title("One-step Model Prediction Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("model_mse.png", dpi=300)
plt.show()
