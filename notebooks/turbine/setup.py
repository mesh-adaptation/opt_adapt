from thetis import *
import logging


logger = logging.getLogger("thetis_output")
logger.setLevel(ERROR)


def initial_mesh(n=2):
    return RectangleMesh(120 * n, 50 * n, 1200, 500)


def initial_control(mesh):
    R = FunctionSpace(mesh, "R", 0)
    return Function(R).assign(250.0)


def forward_run(mesh, control=None):
    """
    Solve the shallow water flow-past-a-turbine problem on a given mesh.

    Optionally, pass an initial value for the control variable
    (y-coordinate of the centre of the second turbine).
    """
    x, y = SpatialCoordinate(mesh)

    # Setup bathymetry
    H = 40.0
    h = 10.0
    a = 60.0
    P1_2d = get_functionspace(mesh, "CG", 1)
    bathymetry = Function(P1_2d)
    bathymetry.interpolate(
        H + h * (exp(-(((y - 100) / a) ** 2)) + exp(-(((y - 400) / a) ** 2)))
    )

    # Setup solver
    solver_obj = solver2d.FlowSolver2d(mesh, bathymetry)
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
    solver_obj.create_equations()

    # Setup boundary conditions
    P1v_2d = solver_obj.function_spaces.P1v_2d
    u_in = interpolate(as_vector([5.0, 0.0]), P1v_2d)
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
    x1 = Constant(456.0)
    x2 = Constant(744.0)
    y1 = Constant(250.0)
    y2 = Function(R).assign(control or 250.0)
    Ct = 0.8
    D = 18.0
    At = D**2

    def bump(x0, y0, scale=1.0):
        r = 18.0 / 2
        qx = ((x - x0) / r) ** 2
        qy = ((y - y0) / r) ** 2
        cond = And(qx < 1, qy < 1)
        b = exp(1 - 1 / (1 - qx)) * exp(1 - 1 / (1 - qy))
        return conditional(cond, Constant(scale) * b, 0)

    b1 = assemble(bump(x1, y1) * dx)
    b2 = assemble(bump(x2, y2) * dx)
    assert b1 > 0.0, f"Invalid area for turbine 1: {b1}"
    assert b2 > 0.0, f"Invalid area for turbine 2: {b2}"
    bumps = bump(x1, y1, scale=1 / b1) + bump(x2, y2, scale=1 / b2)

    # Setup tidal farm
    Ct = Ct * 4.0 / (1.0 + sqrt(1.0 - Ct * At / (H * D)))  # thrust correction
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = bumps
    farm_options.turbine_options.diameter = 18.0
    farm_options.turbine_options.thrust_coefficient = Ct
    options.tidal_turbine_farms = {"everywhere": farm_options}
    rho = Constant(1030.0)

    # Apply initial conditions and solve
    solver_obj.assign_initial_conditions(uv=u_in)
    solver_obj.iterate()

    # Define objective function
    u, p = split(solver_obj.fields.solution_2d)
    coeff = (
        -rho * 0.5 * Ct * (pi * D / 2) ** 2 / At * bumps
    )  # NOTE: negative because we want maximum
    J = assemble(coeff * dot(u, u) ** 1.5 * dx, ad_block_tag="qoi")
    return J, y2