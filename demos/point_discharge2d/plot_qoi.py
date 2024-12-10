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
    J = -np.load(f"data/{method}_progress_J_{ext}.npy")[-1]
    axes.plot(off, J, "x", **kwargs)


labels = {
    "uniform": "Uniform meshing",
    "hessian": "Hessian-based",
    # "go": "Goal-based",
}

# Plot QoI vs #elements
fig, axes = plt.subplots(figsize=(5, 3))
for method, label in labels.items():
    nc = []
    J = []
    for fname in glob.glob(f"data/{method}_*.log"):
        ext = "_".join(fname.split("_")[1:])[:-4]
        nc.append(np.load(f"data/{method}_progress_nc_{ext}.npy")[-1])
        J.append(np.load(f"data/{method}_progress_J_{ext}.npy")[-1])
    axes.semilogx(nc, J, "x", label=label)
axes.set_xlabel("Number of mesh elements")
axes.set_ylabel("QoI (dimensionless)")
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
create_directory("plots")
plt.savefig("plots/converged_qoi_vs_elements.pdf")
plt.savefig("plots/converged_qoi_vs_elements.jpg")

# Plot QoI vs CPU time
fig, axes = plt.subplots(figsize=(5, 3))
for method, label in labels.items():
    try:
        cpu = []
        J = []
        for fname in glob.glob(f"data/{method}_*.log"):
            ext = "_".join(fname.split("_")[1:])[:-4]
            cpu.append(np.load(f"data/{method}_progress_t_{ext}.npy")[-1])
            J.append(np.load(f"data/{method}_progress_J_{ext}.npy")[-1])
        axes.semilogx(cpu, J, "x", label=label)
    except FileNotFoundError:
        continue
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel("QoI (dimensionless)")
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
plt.savefig("plots/converged_qoi_vs_time.pdf")
plt.savefig("plots/converged_qoi_vs_time.jpg")

# Plot control vs #elements
fig, axes = plt.subplots(figsize=(5, 3))
for method, label in labels.items():
    nc = []
    m = []
    for fname in glob.glob(f"data/{method}_*.log"):
        ext = "_".join(fname.split("_")[1:])[:-4]
        nc.append(np.load(f"data/{method}_progress_nc_{ext}.npy")[-1])
        m.append(np.load(f"data/{method}_progress_m_{ext}.npy")[-1])
    axes.semilogx(nc, m, "x", label=label)
axes.set_xlabel("Number of mesh elements")
axes.set_ylabel(r"Control ($\mathrm{m}$)")
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
plt.savefig("plots/converged_control_vs_elements.pdf")
plt.savefig("plots/converged_control_vs_elements.jpg")

# Plot QoI vs CPU time
fig, axes = plt.subplots(figsize=(5, 3))
for method, label in labels.items():
    try:
        cpu = []
        m = []
        for fname in glob.glob(f"data/{method}_*.log"):
            ext = "_".join(fname.split("_")[1:])[:-4]
            cpu.append(np.load(f"data/{method}_progress_t_{ext}.npy")[-1])
            m.append(np.load(f"data/{method}_progress_m_{ext}.npy")[-1])
        axes.semilogx(cpu, m, "x", label=label)
    except FileNotFoundError:
        continue
axes.set_xlabel(r"CPU time ($\mathrm{s}$)")
axes.set_ylabel(r"Control ($\mathrm{m}$)")
axes.grid(True, which="both")
axes.legend()
plt.tight_layout()
plt.savefig("plots/converged_control_vs_time.pdf")
plt.savefig("plots/converged_control_vs_time.jpg")
