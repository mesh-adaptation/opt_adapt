from animate.metric import RiemannianMetric
from firedrake import *
from firedrake_adjoint import *

from opt_adapt.opt import get_state


def initial_mesh(n):
    """
    Construction of initial mesh
    """
    mesh = Mesh("stokes/stokes_control.msh")
    return mesh


def initial_control(mesh):
    """
    The initial of control parameter
    In this example, control parameter is not in Real space
    """
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    return Function(V, name="Control")


def forward_run(mesh, control, outfile=None, **kwargs):
    """
    Solve the PDEs in the given mesh
    """

    # Create a mixed function space for u and p
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = V * Q

    # Set up trial and test functions
    v, q = TestFunctions(W)
    u, p = TrialFunctions(W)

    # Build boundary conditions
    nu = Constant(1)  # Viscosity coefficient
    x, y = SpatialCoordinate(mesh)
    u_inflow = Function(W.sub(0)).interpolate(as_vector([y * (10 - y) / 25.0, 0]))
    noslip = DirichletBC(W.sub(0), (0, 0), (3, 5))
    inflow = DirichletBC(W.sub(0), u_inflow, 1)
    static_bcs = [inflow, noslip]
    g = Function(V, name="Control").project(control)
    controlled_bcs = [DirichletBC(W.sub(0), g, 4)]
    bcs = static_bcs + controlled_bcs

    # Define the bilinear and linear forms
    a = (
        nu * inner(grad(u), grad(v)) * dx
        - inner(p, div(v)) * dx
        - inner(q, div(u)) * dx
    )
    L = Constant(0) * q * dx

    # Solve the forward problem
    w = Function(W)
    sp = {
        "mat_type": "aij",  # Matrix type that we need to use a direct solver
        "ksp_type": "preonly",  # Don't use a linear solver, just apply a preconditioner
        "pc_type": "lu",  # Use a full LU decomposition as a preconditioner
        "pc_factor_mat_solver_type": "mumps",  # Use the MUMPS package to compute the LU decomposition
    }
    solve(a == L, w, bcs=bcs, solver_parameters=sp)
    if outfile is not None:
        u, p = w.subfunctions
        outfile.write(u, p)

    # Conpute the objective function value
    u, p = split(w)
    alpha = Constant(10)
    J = assemble(
        1.0 / 2 * inner(grad(u), grad(u)) * dx + alpha / 2 * inner(g, g) * ds(4)
    )

    return J, g


def hessian(mesh, **kwargs):
    """
    Recover the Hessian of each component of the state and
    intersect them.

    :kwarg adjoint: If ``True``, recover Hessians from the
    adjoint state, rather than the forward one.
    """
    up, eta = get_state(**kwargs).subfunctions
    P1_ten = TensorFunctionSpace(mesh, "CG", 1)
    metric_parameters = {
        "dm_plex_metric": {
            "p": np.inf,
            "target_complexity": 1000.0,
        }
    }

    def hessian_component(f):
        """
        Recover the Hessian of a single component and
        scale it so that all of the components are of
        consistent metric complexity.
        """
        component = RiemannianMetric(P1_ten)
        component.set_parameters(metric_parameters)
        component.compute_hessian(f, method="mixed_L2")
        component.normalise(restrict_sizes=False, restrict_anisotropy=False)
        return component

    metric = hessian_component(up[0])
    metric.intersect(hessian_component(up[1]), hessian_component(eta))
    return metric
