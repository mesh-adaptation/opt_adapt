import glob

import matplotlib.pyplot as plt
import numpy as np
from thetis import create_directory


def plot_progress(axes, method, ext, gradients=True, **kwargs):
    """
    Plot the progress of an optimisation run, in terms
    of control value vs. objective functional value.

    :arg axes: the figure axes to plot on
    :arg method: the optimisation method
    :kwargs: to be passed to matplotlib's plotting functions
    """
    m = np.load(f"data/{method}_progress_m_{ext}.npy")[-1]
    off = np.abs(m - 250.0)
    J = -np.load(f"data/{method}_progress_J_{ext}.npy")[-1] / 1000
    axes.plot(off, J, "x", **kwargs)


fig, axes = plt.subplots(figsize=(5, 3))
labels = {
    "uniform": "Uniform meshing",
    "hessian": "Hessian-based",
    "go": "Goal-based",
}
for method, label in labels.items():
    cpu = []
    J = []
    for fname in glob.glob(f"data/{method}_*.log"):
        ext = "_".join(fname.split("_")[1:])[:-4]
        with open(fname, "r") as f:
            words = f.readline().split()
            if len(words) > 2 and "FAIL" in words[2]:
                continue
            cpu.append(float(words[1]))
        J.append(-np.load(f"data/{method}_progress_J_{ext}.npy")[-1] / 1000)
    axes.semilogx(cpu, J, "x", label=label)
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(r"Power output ($\mathrm{kW}$)")
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
create_directory("plots")
plt.savefig("plots/converged_qoi.pdf")
plt.savefig("plots/converged_qoi.jpg")
