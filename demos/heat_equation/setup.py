from firedrake import *
from firedrake_adjoint import * 


def initial_mesh():
    """
    Construction of initial mesh
    """
    return RectangleMesh(5, 5, (-1,1), (-1,1))

def initial_control(mesh):
    """
    The initial of control parameter
    In this example, control parameter is not in Real space
    """
    W = FunctionSpace(mesh, "DG", degree=0)
    return Function(W, name="Control")

def forward_run(mesh, control, **kwargs):
    """
    Solve the PDEs in the given mesh
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="State")
    W = FunctionSpace(mesh, "DG", degree=0)
    m = Function(W).assign(control)
    v = TestFunction(V)

    # Run the forward model once to create the simulation record
    F = (inner(grad(u), grad(v)) - m*v)*dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    # The functional of interest is the normed difference between desired
    # and simulated temperature profile
    x = SpatialCoordinate(mesh)
    u_desired = exp(-1/(1-x[0]*x[0])-1/(1-x[1]*x[1]))
    J = assemble((0.5*inner(u-u_desired, u-u_desired))*dx)
    
    return J, m

def automated_(mesh, control):
    """
    Using the function in Firedrake
    to computed an approximated solution
    """
    J, m = forward_run(mesh, control)
    Jhat = ReducedFunctional(J, Control(m))
    # Make sure you have scipy >= 0.11 installed
    m_opt = minimize(Jhat, method = "L-BFGS-B",
                tol=2e-08, bounds = (0, 0.5), options = {"disp": False})
    
    return Jhat(m_opt), m_opt