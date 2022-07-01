from firedrake import *
from firedrake_adjoint import *
from opt_adapt.opt import compute_full_hessian
import numpy as np
import pytest


@pytest.fixture(params=[True, False])
def gradient(request):
    return request.param


def test_r_space(gradient):
    """
    Test that :func:`compute_full_hessian` works for
    controls in R-space.
    """
    mesh = UnitTriangleMesh()
    R = FunctionSpace(mesh, "R", 0)
    test = TestFunction(R)
    x = Function(R).assign(1.0)
    c = Control(x)

    # Define some functional that depends only on the control
    J = assemble((x ** 2 + x + 1) * dx)

    # Compute its gradient and check the accuracy
    if gradient:
        g = compute_gradient(J, c)
        dJdx = assemble(test * (2 * x + 1) * dx)
        assert np.isclose(errornorm(g, dJdx), 0)

    # Compute its Hessian and check the accuracy
    H = compute_full_hessian(J, c)
    d2Jdx2 = assemble(test * 2 * dx)
    assert np.isclose(errornorm(H, d2Jdx2), 0)
