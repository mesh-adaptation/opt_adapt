from animate.metric import RiemannianMetric
from firedrake import *

from opt_adapt.opt import get_state


def initial_mesh(n):
    """
    Construction of initial mesh
    """
    return RectangleMesh(5 * 2**n, 5 * 2**n, (-1, 1), (-1, 1))


def initial_control(mesh):
    """
    The initial of control parameter
    In this example, control parameter is not in Real space
    """
    V = FunctionSpace(mesh, "CG", 1)
    return Function(V, name="Control")


def forward_run(mesh, control, outfile=None, **kwargs):
    """
    Solve the PDEs in the given mesh
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="State")
    v = TestFunction(V)
    m = Function(V).project(control)

    # Run the forward model once to create the simulation record
    F = (inner(grad(u), grad(v)) - m * v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    sp = {
        "mat_type": "aij",  # Matrix type that we need to use a direct solver
        "ksp_type": "preonly",  # Don't use a linear solver, just apply a preconditioner
        "pc_type": "lu",  # Use a full LU decomposition as a preconditioner
        "pc_factor_mat_solver_type": "mumps",  # Use the MUMPS package to compute the LU decomposition
    }
    solve(F == 0, u, bc, solver_parameters=sp)
    if outfile is not None:
        outfile.write(u)

    # The functional of interest is the normed difference between desired
    # and simulated temperature profile
    x = SpatialCoordinate(mesh)
    u_desired = exp(-1 / (1 - x[0] * x[0]) - 1 / (1 - x[1] * x[1]))
    nrm = assemble(abs(u_desired) * dx)
    J = assemble(0.5 * inner(u - u_desired, u - u_desired) / nrm * dx)

    return J, m


def hessian(mesh, **kwargs):
    """
    Recover the Hessian of the state.

    :kwarg adjoint: If ``True``, recover the
        Hessian of the adjoint state, rather
        than the forward one.
    """
    c = get_state(**kwargs)
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(P1_ten)
    metric_parameters = {
        "dm_plex_metric": {
            "p": np.inf,
            "target_complexity": 1000.0,
        }
    }
    metric.set_parameters(metric_parameters)
    metric.compute_hessian(c, method="Clement")
    metric.normalise(restrict_sizes=False, restrict_anisotropy=False)
    return metric
