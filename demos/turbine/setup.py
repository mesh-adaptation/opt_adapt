import logging

import numpy as np
import ufl
from animate.metric import RiemannianMetric
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.utility_meshes import RectangleMesh
from thetis.options import TidalTurbineFarmOptions
from thetis.solver2d import FlowSolver2d
from thetis.utility import get_functionspace

from opt_adapt.opt import get_state

logger = logging.getLogger("thetis_output")
logger.setLevel(logging.ERROR)


def initial_mesh(n=4):
    return RectangleMesh(12 * n, 5 * n, 1200, 500)


def initial_control(mesh):
    R = FunctionSpace(mesh, "R", 0)
    return Function(R).assign(250.0)


def forward_run(mesh, control=None, outfile=None, **model_options):
    """
    Solve the shallow water flow-past-a-turbine problem on a given mesh.

    Optionally, pass an initial value for the control variable
    (y-coordinate of the centre of the second turbine).
    """
    x, y = ufl.SpatialCoordinate(mesh)

    # Setup bathymetry
    H = 40.0
    P1_2d = get_functionspace(mesh, "CG", 1)
    bathymetry = Function(P1_2d)
    bathymetry.assign(H)

    # Setup solver
    solver_obj = FlowSolver2d(mesh, bathymetry)
    options = solver_obj.options
    options.element_family = "dg-cg"
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.swe_timestepper_type = "SteadyState"
    options.swe_timestepper_options.solver_parameters = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-08,
        "snes_max_it": 100,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    options.use_grad_div_viscosity_term = False
    options.horizontal_viscosity = Constant(0.5)
    options.quadratic_drag_coefficient = Constant(0.0025)
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_grad_depth_viscosity_term = False
    options.update(model_options)
    solver_obj.create_equations()

    # Setup boundary conditions
    P1v_2d = solver_obj.function_spaces.P1v_2d
    u_in = Function(P1v_2d).interpolate(ufl.as_vector([5.0, 0.0]))
    bcs = {
        1: {"uv": u_in},
        2: {"elev": Constant(0.0)},
        3: {"un": Constant(0.0)},
    }
    if 4 in mesh.exterior_facets.unique_markers:
        bcs[4] = {"un": Constant(0.0)}
    solver_obj.bnd_functions["shallow_water"] = bcs

    # Define turbine parameters
    R = FunctionSpace(mesh, "R", 0)
    ym = 250.0
    sep = 60.0
    x1 = Constant(456.0)
    x2 = Constant(456.0)
    x3 = Constant(456.0)
    xc = Constant(744.0)
    y1 = Constant(ym)
    y2 = Constant(ym + sep)
    y3 = Constant(ym - sep)
    yc = Function(R).assign(control or 250.0)
    Ct = 0.8
    D = 18.0
    At = D**2

    def bump(x0, y0, scale=1.0):
        r = 18.0 / 2
        qx = ((x - x0) / r) ** 2
        qy = ((y - y0) / r) ** 2
        cond = ufl.And(qx < 1, qy < 1)
        b = ufl.exp(1 - 1 / (1 - qx)) * ufl.exp(1 - 1 / (1 - qy))
        return ufl.conditional(cond, Constant(scale) * b, 0)

    b1 = assemble(bump(x1, y1) * ufl.dx)
    b2 = assemble(bump(x2, y2) * ufl.dx)
    b3 = assemble(bump(x3, y3) * ufl.dx)
    bc = assemble(bump(xc, yc) * ufl.dx)
    assert b1 > 0.0, f"Invalid area for turbine 1: {b1}"
    assert b2 > 0.0, f"Invalid area for turbine 2: {b2}"
    assert b3 > 0.0, f"Invalid area for turbine 3: {b3}"
    assert bc > 0.0, f"Invalid area for control turbine: {bc}"
    bumps = (
        bump(x1, y1, scale=1 / b1)
        + bump(x2, y2, scale=1 / b2)
        + bump(x3, y3, scale=1 / b3)
        + bump(xc, yc, scale=1 / bc)
    )

    # Setup tidal farm
    Ct = Ct * 4.0 / (1.0 + ufl.sqrt(1.0 - Ct * At / (H * D)))  # thrust correction
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = bumps
    farm_options.turbine_options.diameter = 18.0
    farm_options.turbine_options.thrust_coefficient = Ct
    options.tidal_turbine_farms = {"everywhere": [farm_options]}
    rho = Constant(1030.0)

    # Apply initial conditions and solve
    solver_obj.assign_initial_conditions(uv=u_in)
    solver_obj.iterate()
    if outfile is not None:
        u, eta = solver_obj.fields.solution_2d.subfunctions
        outfile.write(u, eta)

    # Define objective function
    u, eta = ufl.split(solver_obj.fields.solution_2d)
    coeff = -rho * 0.5 * Ct * (ufl.pi * D / 2) ** 2 / At * bumps
    J_power = coeff * ufl.dot(u, u) ** 1.5 * ufl.dx
    # NOTE: negative because we want maximum

    # Add a regularisation term for constraining the control
    area = assemble(Constant(1.0) * ufl.dx(domain=mesh))
    alpha = 1.0 / area
    J_reg = (
        alpha
        * ufl.conditional(
            yc < y2, (yc - y2) ** 2, ufl.conditional(yc > y3, (yc - y3) ** 2, 0)
        )
        * ufl.dx
    )

    J = assemble(J_power + J_reg, ad_block_tag="qoi")
    return J, yc


def hessian(mesh, **kwargs):
    """
    Recover the Hessian of each component of the state and
    intersect them.

    :kwarg adjoint: If ``True``, recover Hessians from the
    adjoint state, rather than the forward one.
    """
    uv, eta = get_state(**kwargs).subfunctions
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

    metric = hessian_component(uv[0])
    metric.intersect(hessian_component(uv[1]), hessian_component(eta))
    return metric
