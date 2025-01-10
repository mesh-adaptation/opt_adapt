import logging

import numpy as np
import ufl
from animate.metric import RiemannianMetric
from firedrake.adjoint import pyadjoint
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.utility_meshes import RectangleMesh
from thetis.options import TidalTurbineFarmOptions
from thetis.solver2d import FlowSolver2d
from thetis.utility import domain_constant, get_functionspace

from opt_adapt.opt import get_state

logger = logging.getLogger("thetis_output")
logger.setLevel(logging.ERROR)


def initial_mesh(n=4):
    return RectangleMesh(12 * n, 5 * n, 1200, 500)


def initial_control(mesh):
    R = FunctionSpace(mesh, "R", 0)
    return Function(R, val=250.0)


def forward_run(mesh, control=None, outfile=None, debug=False, **model_options):
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
    solver_obj.create_function_spaces()

    # Setup boundary conditions
    P1v_2d = solver_obj.function_spaces.P1v_2d
    u_in = Function(P1v_2d)
    u_in.interpolate(ufl.as_vector([5.0, 0.0]))
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
    x1 = Function(R, val=456.0)
    x2 = Function(R, val=456.0)
    x3 = Function(R, val=456.0)
    xc = Function(R, val=744.0)
    y1 = Function(R, val=ym)
    y2 = Function(R, val=ym + sep)
    y3 = Function(R, val=ym - sep)
    yc = Function(R)
    yc.assign(control or 250.0)
    thrust_coefficient = 0.8
    turbine_diameter = 18.0
    turbine_footprint = turbine_diameter**2

    def bump(x0, y0, label):
        r = turbine_diameter / 2
        qx = ((x - x0) / r) ** 2
        qy = ((y - y0) / r) ** 2
        cond = ufl.conditional(
            ufl.And(qx < 1, qy < 1),
            ufl.exp(1 - 1 / (1 - qx)) * ufl.exp(1 - 1 / (1 - qy)),
            0,
        )
        integral = assemble(cond * ufl.dx)
        assert integral > 0.0, f"Invalid area for {label}"
        return cond / integral

    P1DG_2d = solver_obj.function_spaces.P1DG_2d
    turbine_density = Function(P1DG_2d)
    turbine_density.project(
        bump(x1, y1, "turbine 1")
        + bump(x2, y2, "turbine 2")
        + bump(x3, y3, "turbine 3")
        + bump(xc, yc, "control turbine")
    )

    # Apply thrust correction
    vertical_slice = H * turbine_diameter
    thrust_coefficient *= 4.0 / (
        1.0 + ufl.sqrt(1.0 - thrust_coefficient * turbine_footprint / vertical_slice)
    )

    # Setup tidal farm
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = turbine_diameter
    farm_options.turbine_options.thrust_coefficient = thrust_coefficient
    options.tidal_turbine_farms["everywhere"] = [farm_options]
    solver_obj.create_equations()
    rho = Constant(1030.0)

    # Apply initial conditions and solve
    solver_obj.assign_initial_conditions(uv=u_in)
    solver_obj.iterate()
    if outfile is not None:
        u, eta = solver_obj.fields.solution_2d.subfunctions
        outfile.write(u, eta)

    # Define objective function
    u, eta = ufl.split(solver_obj.fields.solution_2d)
    swiped_area = (ufl.pi * turbine_diameter / 2) ** 2
    area_frac = swiped_area / turbine_footprint
    coeff = rho * 0.5 * thrust_coefficient * area_frac * turbine_density
    J_power = coeff * ufl.dot(u, u) ** 1.5 * ufl.dx

    # Add a regularisation term for constraining the control
    area = assemble(domain_constant(1.0, mesh) * ufl.dx)
    alpha = domain_constant(1.0 / area, mesh)
    J_reg = (
        alpha
        * ufl.conditional(
            yc < y2, (yc - y2) ** 2, ufl.conditional(yc > y3, (yc - y3) ** 2, 0)
        )
        * ufl.dx
    )

    # Sum the two components
    # NOTE: Scaling should be negative because we want maximum
    J = assemble(-(J_power + J_reg), ad_block_tag="qoi")

    if debug:
        controls = {"q_2d": solver_obj.fields.solution_2d}
        for key, control in controls.items():
            Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(control))
            h = Function(control)
            h.assign(0.1)
            assert pyadjoint.taylor_test(Jhat, control, h) > 1.9
            print(f"Taylor test for {key} passed")

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


if __name__ == "__main__":
    from firedrake.output.vtk_output import VTKFile

    pyadjoint.continue_annotation()
    resolution = 4
    init_mesh = initial_mesh(n=resolution)
    init_control = initial_control(init_mesh)
    forward_run(init_mesh, init_control, outfile=VTKFile("test.pvd"), debug=True)
    pyadjoint.pause_annotation()
