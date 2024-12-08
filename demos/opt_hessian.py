import argparse
import importlib
import os
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from animate.adapt import adapt
from animate.utility import VTKFile
from firedrake import triplot
from goalie.log import pyrint
from goalie.utility import create_directory

from opt_adapt.opt import *

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
pwd = os.path.abspath(os.path.dirname(__file__))
choices = [name for name in os.listdir(pwd) if os.path.isdir(name)]
parser.add_argument("demo", type=str, choices=choices)
parser.add_argument("--method", type=str, default="gradient_descent")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--target", type=float, default=1000.0)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--dtol", type=float, default=10.0)
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lr_min", type=float, default=1e-8)
parser.add_argument("--disp", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
demo = args.demo
method = args.method
n = args.n
target = args.target

# Setup initial mesh
setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)

# Setup parameter class
params = OptAdaptParameters(
    method,
    options={
        "disp": args.disp,
        "lr": args.lr,
        "dtol": args.dtol,
        "lr_min": args.lr_min,
        "gtol": args.gtol,
        "maxiter": args.maxiter,
        "target_base": 0.2 * target,
        "target_inc": 0.1 * target,
        "target_max": target,
        "model_options": {
            "no_exports": True,
            "outfile": VTKFile(
                f"{demo}/outputs_hessian/{method}/solution.pvd", adaptive=True
            ),
        },
    },
    Rspace=setup.initial_control(mesh).ufl_element().family() == "Real",
)
pyrint(f"Using method {method}")


def adapt_hessian_based(mesh, target=1000.0, norm_order=1.0, **kwargs):
    """
    Adapt the mesh w.r.t. the intersection of the Hessians of
    each component of velocity and pressure.
    :kwarg target: Desired metric complexity (continuous
        analogue of mesh vertex count).
    :kwarg norm_order: Normalisation order :math:`p` for the
    :math:`L^p` normalisation routine.
    """
    metric = setup.hessian(mesh)
    metric_parameters = {
        "dm_plex_metric": {
            "target_complexity": target,
            "p": norm_order,
            "dm_plex_metric_h_min": 1.0e-05,
            "dm_plex_metric_h_max": 500.0,
            "dm_plex_metric_a_max": 1000.0,
        }
    }
    metric.set_parameters()
    metric.normalise()
    if args.disp > 2:
        pyrint("Metric construction complete.")
    newmesh = adapt(mesh, metric)
    if args.disp > 2:
        pyrint("Mesh adaptation complete.")
    return newmesh


cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
if args.debug:
    m_opt = minimise(
        setup.forward_run,
        mesh,
        setup.initial_control,
        adapt_fn=adapt_hessian_based,
        method=method,
        params=params,
        op=op,
    )
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation completed in {cpu_time:.2f}s")
else:
    try:
        m_opt = minimise(
            setup.forward_run,
            mesh,
            setup.initial_control,
            adapt_fn=adapt_hessian_based,
            method=method,
            params=params,
            op=op,
        )
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Hessian-based optimisation completed in {cpu_time:.2f}s")
    except Exception as exc:
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Hessian-based optimisation failed after {cpu_time:.2f}s")
        print(f"Reason: {exc}")
        failed = True
create_directory(f"{demo}/data")
t = op.t_progress
m = np.array([m.dat.data[0] for m in op.m_progress]).flatten()
J = op.J_progress
dJ = np.array([dj.dat.data[0] for dj in op.dJ_progress]).flatten()
nc = op.nc_progress
np.save(f"{demo}/data/hessian_progress_t_{n}_{method}", t)
np.save(f"{demo}/data/hessian_progress_m_{n}_{method}", m)
np.save(f"{demo}/data/hessian_progress_J_{n}_{method}", J)
np.save(f"{demo}/data/hessian_progress_dJ_{n}_{method}", dJ)
np.save(f"{demo}/data/hessian_progress_nc_{n}_{method}", nc)
with open(f"{demo}/data/hessian_{target:.0f}_{method}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")

# Plot the final mesh
plot_dir = create_directory(f"{demo}/plots")
fig, axes = plt.subplots()
triplot(op.mesh_progress[-1], axes=axes)
axes.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/mesh_hessian_{method}.png")
