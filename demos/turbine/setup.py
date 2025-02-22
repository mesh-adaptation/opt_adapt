import logging

import numpy as np
import ufl
from animate.metric import RiemannianMetric
from firedrake.adjoint import pyadjoint
from firedrake.assemble import assemble
from firedrake.constant import Constant
from firedrake.function import Function
from firedrake.functionspace import TensorFunctionSpace
from firedrake.utility_meshes import RectangleMesh
from thetis.options import DiscreteTidalTurbineFarmOptions
from thetis.solver2d import FlowSolver2d
from thetis.turbines import TurbineFunctionalCallback
from thetis.utility import domain_constant, get_functionspace

from opt_adapt.opt import get_state

logger = logging.getLogger("thetis_output")
logger.setLevel(logging.ERROR)


def initial_mesh(n=4):
    return RectangleMesh(12 * n, 5 * n, 1200, 500)


def initial_control(mesh):
    return domain_constant(260.0, mesh)


def forward_run(mesh, control=None, outfile=None, debug=False, **model_options):
    """
    Solve the shallow water flow-past-a-turbine problem on a given mesh.

    Optionally, pass an initial value for the control variable
    (y-coordinate of the centre of the second turbine).
    """
    x, y = ufl.SpatialCoordinate(mesh)

    # Specify bathymetry
    channel_depth = domain_constant(40.0, mesh)
    channel_width = domain_constant(500.0, mesh)
    bathymetry_scaling = domain_constant(2.0, mesh)
    P1_2d = get_functionspace(mesh, "CG", 1)
    y_prime = y - channel_width / 2
    bathymetry = Function(P1_2d)
    bathymetry.interpolate(
        channel_depth - (bathymetry_scaling * y_prime / channel_width) ** 2
    )

    # Setup solver
    solver_obj = FlowSolver2d(mesh, Constant(channel_depth))
    options = solver_obj.options
    options.element_family = "dg-dg"
    options.timestep = 1.0
    options.simulation_export_time = 1.0
    options.simulation_end_time = 0.5
    options.swe_timestepper_type = "SteadyState"
    options.swe_timestepper_options.solver_parameters = {
        "snes_rtol": 1.0e-12,
    }
    # options.use_grad_div_viscosity_term = False
    options.horizontal_viscosity = Constant(0.5)
    options.quadratic_drag_coefficient = Constant(0.0025)
    # options.use_grad_depth_viscosity_term = False
    options.update(model_options)

    # Setup boundary conditions
    solver_obj.bnd_functions["shallow_water"] = {
        1: {"uv": Constant((3.0, 0.0))},
        2: {"elev": Constant(0.0)},
        3: {"un": Constant(0.0)},
        4: {"un": Constant(0.0)},
    }
    solver_obj.create_function_spaces()

    # Define the thrust curve of the turbine using a tabulated approach:
    # speeds_AR2000: speeds for corresponding thrust coefficients - thrusts_AR2000
    # thrusts_AR2000: list of idealised thrust coefficients of an AR2000 tidal turbine
    # using a curve fitting technique with:
    #   * cut-in speed = 1 m/s
    #   * rated speed = 3.05 m/s
    #   * cut-out speed = 5 m/s
    # (ramp up and down to cut-in and at cut-out speeds for model stability)
    # NOTE: Taken from Thetis:
    #    https://github.com/thetisproject/thetis/blob/master/examples/discrete_turbines/tidal_array.py
    speeds_AR2000 = [
        0.0,
        0.75,
        0.85,
        0.95,
        1.0,
        3.05,
        3.3,
        3.55,
        3.8,
        4.05,
        4.3,
        4.55,
        4.8,
        5.0,
        5.001,
        5.05,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
    ]
    thrusts_AR2000 = [
        0.010531,
        0.032281,
        0.038951,
        0.119951,
        0.516484,
        0.516484,
        0.387856,
        0.302601,
        0.242037,
        0.197252,
        0.16319,
        0.136716,
        0.115775,
        0.102048,
        0.060513,
        0.005112,
        0.00151,
        0.00089,
        0.000653,
        0.000524,
        0.000442,
        0.000384,
        0.000341,
        0.000308,
    ]

    # Setup tidal farm
    farm_options = DiscreteTidalTurbineFarmOptions()
    turbine_density = Function(solver_obj.function_spaces.P1_2d).assign(1.0)
    farm_options.turbine_type = "table"
    farm_options.turbine_density = turbine_density
    farm_options.turbine_options.diameter = 18.0
    farm_options.turbine_options.thrust_speeds = speeds_AR2000
    farm_options.turbine_options.thrust_coefficients = thrusts_AR2000
    farm_options.upwind_correction = False
    farm_options.turbine_coordinates = [
        [domain_constant(x, mesh=mesh), domain_constant(y, mesh=mesh)]
        for x, y in [[456, 250], [456, 310], [456, 190], [744, control or 250]]
    ]
    y2 = farm_options.turbine_coordinates[1][1]
    y3 = farm_options.turbine_coordinates[2][1]
    yc = farm_options.turbine_coordinates[3][1]
    options.discrete_tidal_turbine_farms["everywhere"] = [farm_options]

    # Add a callback for computing the power output
    cb = TurbineFunctionalCallback(solver_obj)
    solver_obj.add_callback(cb, "timestep")

    # Apply initial conditions and solve
    solver_obj.assign_initial_conditions(uv=(ufl.as_vector((1.0e-03, 0.0))))
    solver_obj.iterate()
    if outfile is not None:
        u, eta = solver_obj.fields.solution_2d.subfunctions
        outfile.write(u, eta)

    J_power = sum(cb.integrated_power)

    # Add a regularisation term for constraining the control
    area = assemble(domain_constant(1.0, mesh) * ufl.dx)
    alpha = domain_constant(1.0 / area, mesh)
    J_reg = assemble(
        alpha
        * ufl.conditional(
            yc < y3, (yc - y3) ** 2, ufl.conditional(yc > y2, (yc - y2) ** 2, 0)
        )
        * ufl.dx
    )

    # Sum the two components
    # NOTE: We rescale the functional such that the gradients are ~ order magnitude 1
    # NOTE: We also multiply by -1 so that if we minimise the functional, we maximise
    #       power (maximize is also available from pyadjoint but currently broken)
    scaling = 10000
    J = scaling * (-J_power + J_reg)

    control_variable = yc
    if debug:
        # Perform a Taylor test
        Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(control_variable))
        h = Function(control_variable)
        np.random.seed(23)
        h.dat.data[:] = np.random.random(h.dat.data.shape)
        assert pyadjoint.taylor_test(Jhat, control, h) > 1.95
        print("Taylor test passed")

    return J, control_variable


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
