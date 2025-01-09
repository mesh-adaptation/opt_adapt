import argparse
import importlib
import os
from time import perf_counter

import numpy as np
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from firedrake import *
from firedrake.adjoint import get_solve_blocks, pyadjoint
from goalie.error_estimation import get_dwr_indicator
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
model_options = {
    "no_exports": True,
    "outfile": VTKFile(f"{demo}/outputs_go/{method}/solution.pvd", adaptive=True),
}

# Setup initial mesh
setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)

# Setup parameter class
params = OptAdaptParameters(
    method,
    options={
        "disp": args.disp,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "maxiter": args.maxiter,
        "dtol": args.dtol,
        "gtol": args.gtol,
        "target_base": 0.2 * target,
        "target_inc": 0.1 * target,
        "target_max": target,
        "model_options": model_options,
    },
    Rspace=setup.initial_control(mesh).ufl_element().family() == "Real",
)
pyrint(f"Using method {method}")


def adapt_go(mesh, target=1000.0, alpha=1.0, control=None, **kwargs):
    """
    Adapt the mesh w.r.t. an anisotropic goal-oriented metric.
    :kwarg target: desired target metric complexity
    :kwarg alpha: convergence rate parameter for anisotropic metric
    """
    tape = pyadjoint.get_working_tape()
    mh = MeshHierarchy(mesh, 1)
    q_star = get_state(adjoint=True)
    assert q_star is not None

    # Prolong the adjoint state into an enriched space
    tm = TransferManager()
    V_plus = FunctionSpace(mh[1], q_star.ufl_element())
    q_star_plg = Function(V_plus)
    tm.prolong(q_star, q_star_plg)
    if args.disp > 2:
        pyrint("Base fields prolonged.")

    # Solve the forward and adjoint problem in the enriched space
    # TODO: avoid forward solve
    ref_tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(ref_tape)
    opts = model_options.copy()
    opts.pop("outfile")
    pyadjoint.continue_annotation()
    # TODO: Use Goalie to drive the following
    J_plus, u_plus = setup.forward_run(mh[1], control=control, **opts)
    pyadjoint.pause_annotation()
    pyadjoint.ReducedFunctional(J_plus, pyadjoint.Control(u_plus)).derivative()
    solve_block = get_solve_blocks()[0]
    q_plus = get_state(adjoint=False)
    q_star_plus = get_state(adjoint=True)
    F_plus = replace(solve_block.lhs - solve_block.rhs, {TrialFunction(V_plus): q_plus})
    ref_tape.clear_tape()
    pyadjoint.set_working_tape(tape)
    if args.disp > 2:
        pyrint("Error estimation complete.")

    # Extract an error indicator and project it back down
    q_star_plus -= q_star_plg
    indicator_plus = get_dwr_indicator(F_plus, q_star_plus)
    indicator = project(indicator_plus, FunctionSpace(mesh, "DG", 0))
    indicator.interpolate(abs(indicator))
    if args.disp > 2:
        pyrint("Error estimator projected.")

    # Construct an anisotropic metric
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric.set_parameters(
        {
            "dm_plex_metric": {
                "target_complexity": target,
                "p": np.inf,
                "h_min": 1.0e-05,
                "h_max": 500.0,
                "a_max": 1000.0,
            }
        }
    )
    metric.compute_isotropic_dwr_metric(
        indicator,
        convergence_rate=alpha,
    )
    # TODO: Support switching between metric implementations
    # metric.compute_anisotropic_dwr_metric(
    #     indicator,
    #     hessian=setup.hessian(mesh),
    #     convergence_rate=alpha,
    # )
    metric.normalise(restrict_sizes=True, restrict_anisotropy=True)
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
        adapt_fn=adapt_go,
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
            adapt_fn=adapt_go,
            method=method,
            params=params,
            op=op,
        )
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Goal-oriented optimisation completed in {cpu_time:.2f}s")
    except Exception as exc:
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Goal-oriented optimisation failed after {cpu_time:.2f}s")
        print(f"Reason: {exc}")
        failed = True
create_directory(f"{demo}/data")
t = op.t_progress
m = np.array([m.dat.data[0] for m in op.m_progress]).flatten()
J = op.J_progress
dJ = np.array([dj.dat.data[0] for dj in op.dJ_progress]).flatten()
nc = op.nc_progress
np.save(f"{demo}/data/go_progress_t_{target}_{method}", t)
np.save(f"{demo}/data/go_progress_m_{target}_{method}", m)
np.save(f"{demo}/data/go_progress_J_{target}_{method}", J)
np.save(f"{demo}/data/go_progress_dJ_{target}_{method}", dJ)
np.save(f"{demo}/data/go_progress_nc_{target}_{method}", nc)
with open(f"{demo}/data/go_{target:.0f}_{method}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")
