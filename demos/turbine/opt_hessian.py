from setup import *
from firedrake.meshadapt import RiemannianMetric, adapt
from firedrake_adjoint import *
from pyroteus.metric import *
from pyroteus.recovery import *
from opt_adapt.opt import *
import argparse
import numpy as np
from time import perf_counter


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--target", type=float, default=1000.0)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--disp", type=int, default=2)
args = parser.parse_args()
n = args.n
target = args.target
options = {
    "disp": args.disp,
    "lr": args.lr,
    "maxiter": args.maxiter,
    "target_base": 0.2*target,
    "target_inc": 0.1*target,
    "target_max": target,
}


def hessian(mesh, **kwargs):
    """
    Recover the Hessian of each component of the state and
    intersect them.

    :kwarg adjoint: If ``True``, recover Hessians from the
    adjoint state, rather than the forward one.
    """
    uv, eta = get_state(**kwargs).split()
    V = FunctionSpace(mesh, uv.ufl_element().family(), uv.ufl_element().degree())
    u = interpolate(uv[0], V)
    v = interpolate(uv[1], V)

    def hessian_component(f):
        """
        Recover the Hessian of a single component and
        scale it so that all of the components are of
        consistent metric complexity.
        """
        return space_normalise(hessian_metric(recover_hessian(f)), 1000.0, "inf")

    metric = metric_intersection(hessian_component(u), hessian_component(v), hessian_component(eta))
    print_output("Hessian intersection complete.")
    return metric


def adapt_hessian_based(mesh, target=1000.0, norm_order=1.0, **kwargs):
    """
    Adapt the mesh w.r.t. the intersection of the Hessians of
    each component of velocity and pressure.

    :kwarg target: Desired metric complexity (continuous
        analogue of mesh vertex count).
    :kwarg norm_order: Normalisation order :math:`p` for the
    :math:`L^p` normalisation routine.
    """
    metric = space_normalise(hessian(mesh), target, norm_order)
    enforce_element_constraints(metric, 1.0e-05, 500.0, 1000.0)
    print_output("Metric construction complete.")
    newmesh = adapt(mesh, RiemannianMetric(mesh).assign(metric))
    print_output("Mesh adaptation complete.")
    return newmesh


mesh = initial_mesh(n=n)
cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
try:
    y2_opt = minimise(forward_run, mesh, initial_control, adapt_fn=adapt_hessian_based, options=options, op=op)
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Hessian-based optimisation completed in {cpu_time:.2f}s")
except Exception as exc:
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Hessian-based optimisation failed after {cpu_time:.2f}s")
    print(f"Reason: {exc}")
    failed = True
create_directory("data")
np.save(f"data/hessian_progress_m_{target:.0f}", np.array([m.dat.data[0] for m in op.m_progress]).flatten())
np.save(f"data/hessian_progress_J_{target:.0f}", op.J_progress)
np.save(f"data/hessian_progress_dJdm_{target:.0f}", np.array([dj.dat.data[0] for dj in op.dJdm_progress]).flatten())
with open(f"data/hessian_{target:.0f}.log", "w+") as f:
    f.write(f"cpu_time: {cpu_time}")
    if failed:
        f.write(f" (FAIL)")
