import matplotlib.pyplot as plt
import numpy as np


m = np.linspace(0, 500, 1001)
H = 40
h = 10
y0, y1 = 100, 400
a = 60
J = H + h * (np.exp(-(((m - y0) / a) ** 2)) + np.exp(-(((m - y1) / a) ** 2)))

fig, axes = plt.subplots()
axes.plot(m, J)
axes.set_xlabel(r"$y$-coordinate")
axes.set_ylabel(r"Bathymetry ($\mathrm{m}$)")
axes.grid(True, which="both")
plt.tight_layout()
plt.savefig("bathymetry.pdf")
