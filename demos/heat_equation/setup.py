from firedrake import *
from firedrake_adjoint import * 


def initial_mesh(n):
    """
    Construction of initial mesh
    """
    return RectangleMesh(5 * 2 ** n, 5 * 2 ** n, (-1,1), (-1,1))

def initial_control(mesh):
    """
    The initial of control parameter
    In this example, control parameter is not in Real space
    """
    R = FunctionSpace(mesh, "R", 0)
    return Function(R).assign(0.1)

def forward_run(mesh, control, **kwargs):
    """
    Solve the PDEs in the given mesh
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="State")
    R = FunctionSpace(mesh, "R", 0)
    m = Function(R).assign(control)
    v = TestFunction(V)

    # Run the forward model once to create the simulation record
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    # The functional of interest is the normed difference between desired
    # and simulated temperature profile
    x = SpatialCoordinate(mesh)
    u_desired = exp(-1/(1-x[0]*x[0])-1/(1-x[1]*x[1]))
    nrm = assemble(abs(u_desired) * dx)
    J = assemble(0.5 * inner(u - u_desired, u - u_desired) / nrm * dx)
    
    return J, m