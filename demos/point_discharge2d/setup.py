"""
Problem specification for a simple
advection-diffusion test case with a
point source, from [Riadh et al. 2014].

The 0D control variable is the radius used
in the Gaussian approxmiation of the point
source.

The objective function is the effectively
the :math:`L^2` error between the tracer
solution and the analytical solution, except
we do not integrate near to the point source,
since it blows up.

[Riadh et al. 2014] A. Riadh, G.
    Cedric, M. Jean, "TELEMAC modeling
    system: 2D hydrodynamics TELEMAC-2D
    software release 7.0 user manual."
    Paris: R&D, Electricite de France,
    p. 134 (2014).
"""
from firedrake import *
from pyroteus.math import bessk0
from pyroteus.metric import *
from pyroteus.recovery import *
from opt_adapt.opt import get_state


dx = dx(degree=12)  # Use a high quadrature degree


def initial_mesh(n=1):
    return RectangleMesh(100 * n, 20 * n, 50, 10)


def initial_control(mesh):
    R = FunctionSpace(mesh, "R", 0)
    return Function(R).assign(0.1)


def forward_run(mesh, control=None, outfile=None, **model_options):
    """
    Solve the advection-diffusion point discharge problem
    on a given mesh.

    Optionally, pass an initial value for the control variable
    (radius used in the source model).
    """
    x, y = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    c = Function(P1, name="Tracer concentration")

    # Define source term
    R = FunctionSpace(mesh, "R", 0)
    r = Function(R).assign(control or 0.1)
    scale, xs, ys = Constant(100.0), Constant(2.0), Constant(5.0)
    r2 = (x - xs) ** 2 + (y - ys) ** 2
    d = max_value(sqrt(r2), r)
    source = scale * exp(-r2 / r ** 2)

    # Define physical parameters
    D = Constant(0.1)
    u = Constant(as_vector([1, 0]))

    # Define stabilisation term
    h = CellSize(mesh)
    unorm = sqrt(dot(u, u))
    tau = 0.5 * h / unorm
    tau = min_value(tau, unorm * h / (6 * D))
    psi = TestFunction(P1)
    psi = psi + tau * dot(u, grad(psi))

    # Setup variational problem
    F = (
        source * psi * dx
        - dot(u, grad(c)) * psi * dx
        - inner(D * grad(c), grad(psi)) * dx
    )
    bcs = DirichletBC(P1, 0, 1)

    # Solve PDE
    sp = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solve(F == 0, c, bcs=bcs, solver_parameters=sp)
    if outfile is not None:
        outfile.write(c)

    # Define analytical solution
    Pe = 0.5 * u[0] / D
    q = 1.0
    c_ana = 0.5 * q / (pi * D) * exp(Pe * (x - xs)) * bessk0(Pe * d)

    # Define quantity of interest
    kernel = conditional(r2 > r ** 2, 1, 0)
    area = assemble(Constant(1.0, domain=mesh) * dx)
    J = assemble(kernel * (c - c_ana) ** 2 / area * dx, ad_block_tag="qoi")
    return J, r


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
