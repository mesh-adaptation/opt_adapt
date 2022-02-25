from setup import *
import numpy as np


mesh = initial_mesh(n=2)
R = FunctionSpace(mesh, "R", 0)

ctrls, qois = [], []
create_directory("data")
for c in np.linspace(50, 450, 101):
    J = forward_run(mesh, Function(R).assign(c))[0]
    print(f"y-coord {c:.2f}: {-J/1000:.2f}kW")
    ctrls.append(c)
    qois.append(J)
    np.save("data/control_space_m", ctrls)
    np.save("data/control_space_J", qois)
