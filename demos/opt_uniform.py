import argparse
import importlib
import os
from time import perf_counter

import numpy as np
from animate.utility import VTKFile
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
        "model_options": {
            "no_exports": not args.debug,
            "outfile": VTKFile(
                f"{demo}/outputs_uniform/{method}/solution.pvd", adaptive=True
            ),
        },
    },
    Rspace=setup.initial_control(mesh).ufl_element().family() == "Real",
)
pyrint(f"Using method {method}")


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
t = op.t_progress
m = np.array([m.dat.data[0] for m in op.m_progress]).flatten()
J = op.J_progress
dJ = np.array([dj.dat.data[0] for dj in op.dJ_progress]).flatten()
nc = op.nc_progress
np.save(f"{demo}/data/uniform_progress_t_{n}_{method}", t)
np.save(f"{demo}/data/uniform_progress_m_{n}_{method}", m)
np.save(f"{demo}/data/uniform_progress_J_{n}_{method}", J)
np.save(f"{demo}/data/uniform_progress_dJ_{n}_{method}", dJ)
np.save(f"{demo}/data/uniform_progress_nc_{n}_{method}", nc)
with open(f"{demo}/data/uniform_{n}_{method}.log", "w+") as f:
    note = " (FAIL)" if failed else ""
    f.write(f"cpu_time: {cpu_time}{note}\n")
