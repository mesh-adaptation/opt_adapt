from thetis import create_directory, File
from opt_adapt.opt import OptimisationProgress, minimise
import argparse
import importlib
import numpy as np
from time import perf_counter


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("demo", type=str, choices=["turbine", "point_discharge2d"])
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--disp", type=int, default=1)
args = parser.parse_args()
demo = args.demo
n = args.n
options = {
    "disp": args.disp,
    "lr": args.lr,
    "maxiter": args.maxiter,
    "gtol": args.gtol,
    "model_options": {
        "no_exports": True,
        "outfile": File(f"{demo}/outputs_uniform/solution.pvd", adaptive=True),
    },
}

setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)
cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
try:
    m_opt = minimise(setup.forward_run, mesh, setup.initial_control, options=options, op=op)
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation completed in {cpu_time:.2f}s")
except Exception as exc:
    cpu_time = perf_counter() - cpu_timestamp
    print(f"Uniform optimisation failed after {cpu_time:.2f}s")
    print(f"Reason: {exc}")
    failed = True
create_directory(f"{demo}/data")
np.save(
    f"{demo}/data/uniform_progress_m_{n}",
    np.array([m.dat.data[0] for m in op.m_progress]).flatten(),
)
np.save(f"{demo}/data/uniform_progress_J_{n}", op.J_progress)
np.save(
    f"{demo}/data/uniform_progress_dJdm_{n}",
    np.array([dj.dat.data[0] for dj in op.dJdm_progress]).flatten(),
)
with open(f"{demo}/data/uniform_{n}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")
