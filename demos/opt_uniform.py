from pyroteus.log import pyrint
from pyroteus.utility import create_directory, File
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
parser.add_argument("--method", type=str, default="gradient_descent")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--maxiter", type=int, default=100)
parser.add_argument("--gtol", type=float, default=1.0e-05)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lr_lowerbound", type=float, default=1e-25)
parser.add_argument("--check_lr", type=float, default=False)
parser.add_argument("--disp", type=int, default=1)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
demo = args.demo
method = args.method
n = args.n
params = OptAdaptParameters(
    method,
    options={
        "disp": args.disp,
        "lr": args.lr,
        "lr_lowerbound": args.lr_lowerbound,
        "maxiter": args.maxiter,
        "gtol": args.gtol,
        "model_options": {
            "no_exports": True,
            "outfile": File(f"{demo}/outputs_uniform/solution.pvd", adaptive=True),
        },
    },
)
pyrint(f"Using method {method}")


setup = importlib.import_module(f"{demo}.setup")
mesh = setup.initial_mesh(n=n)
cpu_timestamp = perf_counter()
op = OptimisationProgress()
failed = False
if args.debug:
    m_opt = minimise(
        setup.forward_run,
        mesh,
        setup.initial_control,
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
            method=method,
            params=params,
            op=op,
        )
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Uniform optimisation completed in {cpu_time:.2f}s")
    except Exception as exc:
        cpu_time = perf_counter() - cpu_timestamp
        print(f"Uniform optimisation failed after {cpu_time:.2f}s")
        print(f"Reason: {exc}")
        failed = True
create_directory(f"{demo}/data")
m = np.array([m.dat.data[0] for m in op.m_progress]).flatten()
J = op.J_progress
dJ = np.array([dj.dat.data[0] for dj in op.dJ_progress]).flatten()
np.save(f"{demo}/data/uniform_progress_m_{n}_{method}", m)
np.save(f"{demo}/data/uniform_progress_J_{n}_{method}", J)
np.save(f"{demo}/data/uniform_progress_dJ_{n}_{method}", dJ)
with open(f"{demo}/data/uniform_{n}_{method}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")
