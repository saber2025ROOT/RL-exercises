import re

import matplotlib.pyplot as plt

noise_map = {
    "noise_0.txt": 0.0,
    "noise_001.txt": 0.01,
    "noise_005.txt": 0.05,
    "noise_010.txt": 0.10,
    "noise_020.txt": 0.20,
}

noise_levels = []
final_returns = []

for file, noise in noise_map.items():
    last_return = None

    with open(file) as f:
        for line in f:
            if "[Eval" in line:
                m = re.search(r"AvgReturn\s+([0-9.]+)", line)
                if m:
                    last_return = float(m.group(1))

    noise_levels.append(noise)
    final_returns.append(last_return)

pairs = sorted(zip(noise_levels, final_returns))
noise_levels = [p[0] for p in pairs]
final_returns = [p[1] for p in pairs]

plt.figure(figsize=(7, 5))
plt.plot(noise_levels, final_returns, marker="o", linewidth=2)

plt.xlabel("Model Noise Level")
plt.ylabel("Final Average Return")
plt.title("Failure Mode: Effect of Model Noise")
plt.grid(True)

plt.tight_layout()
plt.savefig("noise_failure.png", dpi=300)
plt.show()
