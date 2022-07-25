from firedrake import *
from firedrake_adjoint import *
from pyroteus.metric import *
from pyroteus.recovery import *
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


def forward_run(mesh, control, **kwargs):
    """
    Solve the PDEs in the given mesh
    """
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)
    W = V*Q

    v, q = TestFunctions(W)
    u, p = TrialFunctions(W)

    nu = Constant(1)     # Viscosity coefficient

    x, y = SpatialCoordinate(mesh)
    u_inflow = as_vector([y*(10-y)/25.0, 0])

    noslip = DirichletBC(W.sub(0), (0, 0), (3, 5))
    inflow = DirichletBC(W.sub(0), interpolate(u_inflow, V), 1)
    static_bcs = [inflow, noslip]

    g = Function(V, name="Control").assign(control)
    controlled_bcs = [DirichletBC(W.sub(0), g, 4)]
    bcs = static_bcs + controlled_bcs

    a = nu*inner(grad(u), grad(v))*dx - inner(p, div(v))*dx - inner(q, div(u))*dx
    L = Constant(0)*q*dx

    w = Function(W)
    sp = {
    "mat_type": "aij",  # Matrix type that we need to use a direct solver
    "snes_monitor": None, # Print the nonlinear solver progress
    "ksp_type": "preonly",  # Don't use a linear solver, just apply a preconditioner
    "ksp_monitor": None,  # Print the linear solver progress
    "pc_type": "lu",  # Use a full LU decomposition as a preconditioner
    "pc_factor_mat_solver_type": "mumps",  # Use the MUMPS package to compute the LU decomposition
    }
    solve(a == L, w, bcs=bcs, solver_parameters=sp)
    
    u, p = split(w)
    alpha = Constant(10)
    J = assemble(1./2*inner(grad(u), grad(u))*dx + alpha/2*inner(g, g)*ds(4))

    return J, g


def hessian(mesh, **kwargs):
    """
    Recover the Hessian of the state.

    :kwarg adjoint: If ``True``, recover the
        Hessian of the adjoint state, rather
        than the forward one.
    """
    c = get_state(**kwargs)
    H = hessian_metric(recover_hessian(c))
    M = space_normalise(H, 1000.0, "inf")
    return M
