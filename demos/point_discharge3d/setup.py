"""
Problem specification for a simple
3D advection-diffusion test case with
a point source. Extended from the 2D
test case in [Riadh et al. 2014].

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

from animate.metric import RiemannianMetric
from firedrake import *

from opt_adapt.opt import get_state

dx = dx(degree=3)
# dx = dx(degree=12)  # TODO: Use a high quadrature degree


def initial_mesh(n=1):
    return BoxMesh(100 * n, 20 * n, 20 * n, 50, 10, 10)


def initial_control(mesh):
    R = FunctionSpace(mesh, "R", 0)
    return Function(R).assign(0.2)


def forward_run(mesh, control=None, outfile=None, **model_options):
    """
    Solve the advection-diffusion point discharge problem
    on a given mesh.

    Optionally, pass an initial value for the control variable
    (radius used in the source model).
    """
    x, y, z = SpatialCoordinate(mesh)
    P1 = FunctionSpace(mesh, "CG", 1)
    c = Function(P1, name="Tracer concentration")

    # Define source term
    R = FunctionSpace(mesh, "R", 0)
    r = Function(R).assign(control or 0.1)
    scale, xs, ys, zs = Constant(100.0), Constant(2.0), Constant(5.0), Constant(5.0)
    r2 = (x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2
    d = max_value(sqrt(r2), r)
    source = scale * exp(-r2 / r**2)

    # Define physical parameters
    D = Constant(0.1)
    u = Constant(as_vector([1, 0, 0]))

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
    c_ana = q / (8 * pi**2 * d * D) * exp(Pe * (x - xs)) * exp(-Pe * d)

    # Define quantity of interest
    kernel = conditional(r2 > r**2, 1, 0)
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
