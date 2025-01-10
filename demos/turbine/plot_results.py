import glob

import matplotlib.pyplot as plt
import numpy as np
from thetis import create_directory

create_directory("plots")

data_dict = {
    "uniform": {
        "label": "Uniform meshing",
    },
    "hessian": {
        "label": "Hessian-based",
    },
    # "go": {
    #     "label": "Goal-oriented",
    # },
}

# Load data from files
variables = ("nc", "J", "t", "m")
for method, data in data_dict.items():
    for variable in variables:
        data[variable] = []
    for fname in glob.glob(f"data/{method}_*.log"):
        ext = "_".join(fname.split("_")[1:])[:-4]
        for variable in variables:
            fname = f"data/{method}_progress_{variable}_{ext}.npy"
            try:
                value = np.load(fname)[-1]
            except (FileNotFoundError, IndexError):
                print(f"Can't load {fname}")
                continue
            if variable == "J":
                data[variable].append(-value / 1000)
            else:
                data[variable].append(value)

metadata = {
    "J": {"label": "qoi", "name": r"Power output ($\mathrm{kW}$)"},
    "nc": {"label": "elements", "name": "Number of mesh elements"},
    "t": {"label": "time", "name": r"CPU time ($\mathrm{s}$)"},
    "m": {"label": "control", "name": r"Control ($\mathrm{m}$)"},
}


def plot(v1, v2):
    fig, axes = plt.subplots(figsize=(5, 3))
    for data in data_dict.values():
        axes.semilogx(data[v1], data[v2], "x", label=data["label"])
    axes.set_xlabel(metadata[v1]["name"])
    axes.set_ylabel(metadata[v2]["name"])
    axes.grid(True, which="both")
    axes.legend()
    plt.tight_layout()
    fname = f"converged_{metadata[v2]['label']}_vs_{metadata[v1]['label']}"
    for ext in ("pdf", "jpg"):
        plt.savefig(f"plots/{fname}.{ext}")


plot("nc", "J")
plot("t", "J")
plot("nc", "m")
plot("t", "m")
