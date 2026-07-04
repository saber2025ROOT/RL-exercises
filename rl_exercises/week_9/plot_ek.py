import re

import matplotlib.pyplot as plt

steps = []
errors = {}

with open("dyna_model_ek.txt", "r") as f:
    for line in f:
        if "[MultiStep]" in line:
            step_match = re.search(r"Step\s+(\d+)", line)
            e_matches = re.findall(r"E(\d+):([0-9.eE+-]+)", line)

            if step_match and e_matches:
                step = int(step_match.group(1))
                steps.append(step)
                errors[step] = [float(v) for _, v in e_matches]

early_step = steps[0]
late_step = steps[-1]

ks = list(range(1, 21))

plt.figure(figsize=(7, 5))
plt.plot(
    ks, errors[early_step], marker="o", label=f"Early checkpoint: {early_step} steps"
)
plt.plot(ks, errors[late_step], marker="s", label=f"Late checkpoint: {late_step} steps")

plt.xlabel("Prediction horizon k")
plt.ylabel("Multi-step error E_k")
plt.title("Multi-step Model Prediction Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("multistep_error.png", dpi=300)
plt.show()
