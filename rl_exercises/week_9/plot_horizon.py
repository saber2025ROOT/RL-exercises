import glob
import re

import matplotlib.pyplot as plt

horizons = []
final_returns = []

for file in sorted(glob.glob("horizon_*.txt")):
    horizon = int(re.search(r"horizon_(\d+)", file).group(1))

    last_return = None
    with open(file) as f:
        for line in f:
            if "[Eval" in line:
                m = re.search(r"AvgReturn\s+([0-9.]+)", line)
                if m:
                    last_return = float(m.group(1))

    horizons.append(horizon)
    final_returns.append(last_return)

pairs = sorted(zip(horizons, final_returns))
horizons = [p[0] for p in pairs]
final_returns = [p[1] for p in pairs]

plt.figure(figsize=(7, 5))
plt.plot(horizons, final_returns, marker="o", linewidth=2)

plt.xlabel("Imagination Horizon")
plt.ylabel("Final Average Return")
plt.title("Final Return vs Imagination Horizon")
plt.grid(True)

plt.tight_layout()
plt.savefig("horizon_return.png", dpi=300)
plt.show()
