from setup import *
from firedrake.meshadapt import RiemannianMetric, adapt
from firedrake_adjoint import *
from firedrake.adjoint import get_solve_blocks
from pyroteus.error_estimation import *
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
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--disp", type=int, default=2)
args = parser.parse_args()
n = args.n
target = args.target
model_options = {
    "output_directory": "outputs_go",
}
options = {
    "disp": args.disp,
    "lr": args.lr,
    "maxiter": args.maxiter,
    "gtol": args.gtol,
    "target_base": 0.2 * target,
    "target_inc": 0.1 * target,
    "target_max": target,
    "model_options": model_options,
}


def adapt_go(mesh, target=1000.0, alpha=1.0, control=None, **kwargs):
    """
    Adapt the mesh w.r.t. an anisotropic goal-oriented metric.

    :kwarg target: desired target metric complexity
    :kwarg alpha: convergence rate parameter for anisotropic metric
    """
    tape = get_working_tape()
    mh = MeshHierarchy(mesh, 1)
    q_star = get_state(adjoint=True)
    assert q_star is not None

    # Prolong the adjoint state into an enriched space
    tm = TransferManager()
    V_plus = FunctionSpace(mh[1], q_star.ufl_element())
    q_star_plg = Function(V_plus)
    tm.prolong(q_star, q_star_plg)
    print_output("Base fields prolonged.")

    # Solve the forward and adjoint problem in the enriched space
    # TODO: avoid forward solve
    ref_tape = Tape()
    set_working_tape(ref_tape)
    J_plus, u_plus = forward_run(mh[1], control=control, **model_options)
    ReducedFunctional(J_plus, Control(u_plus)).derivative()
    solve_block = get_solve_blocks()[0]
    q_plus = get_state(adjoint=False)
    q_star_plus = get_state(adjoint=True)
    F_plus = replace(solve_block.lhs - solve_block.rhs, {TrialFunction(V_plus): q_plus})
    ref_tape.clear_tape()
    set_working_tape(tape)
    print_output("Error estimation complete.")

    # Extract an error indicator and project it back down
    q_star_plus -= q_star_plg
    indicator_plus = get_dwr_indicator(F_plus, q_star_plus)
    indicator = project(indicator_plus, FunctionSpace(mesh, "DG", 0))
    indicator.interpolate(abs(indicator))
    print_output("Error estimator projected.")

    # Construct an anisotropic metric
    metric = anisotropic_metric(
        indicator,
        hessian=hessian(mesh),
        target_complexity=target,
        convergence_rate=alpha,
    )
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
    y2_opt = minimise(
        forward_run, mesh, initial_control, adapt_fn=adapt_go, options=options, op=op
    )
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Goal-oriented optimisation completed in {cpu_time:.2f}s")
except Exception as exc:
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Goal-oriented optimisation failed after {cpu_time:.2f}s")
    print(f"Reason: {exc}")
    failed = True
create_directory("data")
np.save(
    f"data/go_progress_m_{target:.0f}",
    np.array([m.dat.data[0] for m in op.m_progress]).flatten(),
)
np.save(f"data/go_progress_J_{target:.0f}", op.J_progress)
np.save(
    f"data/go_progress_dJdm_{target:.0f}",
    np.array([dj.dat.data[0] for dj in op.dJdm_progress]).flatten(),
)
with open(f"data/go_{target:.0f}.log", "w+") as f:
    f.write(f"cpu_time: {cpu_time}\n")
    if failed:
        f.write("(FAIL)\n")
