import matplotlib.pyplot as plt
import numpy as np


m = np.load("control_space_m.npy")
J = -np.load("control_space_J.npy")/1000

fig, axes = plt.subplots()
axes.plot(m, J, color="C0")
axes.set_xlabel(r"$y$-coordinate of second turbine")
axes.set_ylabel(r"Power output ($\mathrm{kW}$)")
axes.set_xlim([50, 450])
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig("control_space.pdf")
