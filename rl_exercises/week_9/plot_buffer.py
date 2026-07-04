import glob
import re

import matplotlib.pyplot as plt

buffers = []
final_returns = []
final_state_mse = []

for file in glob.glob("buffer_*.txt"):
    buffer_size = int(re.search(r"buffer_(\d+)", file).group(1))

    last_return = None
    last_mse = None

    with open(file) as f:
        for line in f:
            if "[Eval" in line:
                m = re.search(r"AvgReturn\s+([0-9.]+)", line)
                if m:
                    last_return = float(m.group(1))

            if "[Model]" in line:
                m = re.search(r"State MSE:\s*([0-9.eE+-]+)", line)
                if m:
                    last_mse = float(m.group(1))

    buffers.append(buffer_size)
    final_returns.append(last_return)
    final_state_mse.append(last_mse)

pairs = sorted(zip(buffers, final_returns, final_state_mse))
buffers = [p[0] for p in pairs]
final_returns = [p[1] for p in pairs]
final_state_mse = [p[2] for p in pairs]

plt.figure(figsize=(7, 5))
plt.plot(buffers, final_returns, marker="o")
plt.xlabel("Replay Buffer Size")
plt.ylabel("Final Average Return")
plt.title("Final Return vs Buffer Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("buffer_return.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(buffers, final_state_mse, marker="o")
plt.xlabel("Replay Buffer Size")
plt.ylabel("Final State MSE")
plt.title("Model MSE vs Buffer Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("buffer_mse.png", dpi=300)
plt.show()
