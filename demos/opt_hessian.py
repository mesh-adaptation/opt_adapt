from thetis import create_directory, print_output, File
from firedrake.meshadapt import RiemannianMetric, adapt
from firedrake_adjoint import *
from pyroteus.metric import space_normalise, enforce_element_constraints
from opt_adapt.opt import *
import argparse
import importlib
import numpy as np
import os
from time import perf_counter


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
pwd = os.path.abspath(os.path.dirname(__file__))
choices = [name for name in os.listdir(pwd) if os.path.isdir(name)]
parser.add_argument("demo", type=str, choices=choices)
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--target", type=float, default=1000.0)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--disp", type=int, default=2)
args = parser.parse_args()
demo = args.demo
n = args.n
target = args.target
options = {
    "disp": args.disp,
    "lr": args.lr,
    "gtol": args.gtol,
    "maxiter": args.maxiter,
    "target_base": 0.2 * target,
    "target_inc": 0.1 * target,
    "target_max": target,
    "model_options": {
        "no_exports": True,
        "outfile": File(f"{demo}/outputs_hessian/solution.pvd", adaptive=True),
    },
}


def adapt_hessian_based(mesh, target=1000.0, norm_order=1.0, **kwargs):
    """
    Adapt the mesh w.r.t. the intersection of the Hessians of
    each component of velocity and pressure.

    :kwarg target: Desired metric complexity (continuous
        analogue of mesh vertex count).
    :kwarg norm_order: Normalisation order :math:`p` for the
    :math:`L^p` normalisation routine.
    """
    metric = space_normalise(setup.hessian(mesh), target, norm_order)
    enforce_element_constraints(metric, 1.0e-05, 500.0, 1000.0)
    print_output("Metric construction complete.")
    newmesh = adapt(mesh, RiemannianMetric(mesh).assign(metric))
    print_output("Mesh adaptation complete.")
    return newmesh


setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)
cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
try:
    m_opt = minimise(
        setup.forward_run,
        mesh,
        setup.initial_control,
        adapt_fn=adapt_hessian_based,
        options=options,
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
np.save(
    f"{demo}/data/hessian_progress_m_{target:.0f}",
    np.array([m.dat.data[0] for m in op.m_progress]).flatten(),
)
np.save(f"{demo}/data/hessian_progress_J_{target:.0f}", op.J_progress)
np.save(
    f"{demo}/data/hessian_progress_dJdm_{target:.0f}",
    np.array([dj.dat.data[0] for dj in op.dJdm_progress]).flatten(),
)
with open(f"{demo}/data/hessian_{target:.0f}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")
