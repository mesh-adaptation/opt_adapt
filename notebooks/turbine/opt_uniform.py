from setup import *
from firedrake_adjoint import *
from opt_adapt.opt import OptimisationProgress, minimise
import argparse
import numpy as np
from time import perf_counter


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--disp", type=int, default=1)
args = parser.parse_args()
n = args.n
options = {
    "disp": args.disp,
    "lr": args.lr,
    "maxiter": args.maxiter,
}

mesh = initial_mesh(n=n)
cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
try:
    y2_opt = minimise(forward_run, mesh, initial_control, options=options, op=op)
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation completed in {cpu_time:.2f}s")
except Exception as exc:
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation failed after {cpu_time:.2f}s")
    print(f"Reason: {exc}")
    failed = True
np.save(f"uniform_progress_m_{n}", np.array([m.dat.data[0] for m in op.m_progress]).flatten())
np.save(f"uniform_progress_J_{n}", op.J_progress)
np.save(f"uniform_progress_dJdm_{n}", np.array([dj.dat.data[0] for dj in op.dJdm_progress]).flatten())
with open(f"uniform_{n}.log", "w+") as f:
    f.write(f"cpu_time: {cpu_time}")
    if failed:
        f.write(f" (FAIL)")
