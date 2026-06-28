import re

import matplotlib.pyplot as plt

files = {
    "Conservative": "conservative.txt",
    "Balanced": "balanced.txt",
    "Aggressive": "aggressive.txt",
}

plt.figure(figsize=(7, 5))

for label, filename in files.items():
    steps = []
    returns = []

    with open(filename) as f:
        for line in f:
            if "[Eval" in line:
                s = re.search(r"Real Steps\s+(\d+)", line)
                r = re.search(r"AvgReturn\s+([0-9.]+)", line)

                if s and r:
                    steps.append(int(s.group(1)))
                    returns.append(float(r.group(1)))

    plt.plot(steps, returns, marker="o", linewidth=2, label=label)

plt.xlabel("Real Environment Steps")
plt.ylabel("Average Return")
plt.title("Model / Imagination Budget")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("budget_comparison.png", dpi=300)
plt.show()
