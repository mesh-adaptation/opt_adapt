"""
Tests for the :class:`Matrix` class.
"""
from firedrake import *
from firedrake_adjoint import *
from opt_adapt.matrix import *
import numpy as np
import pytest


@pytest.fixture(params=[True, False])
def r_space(request):
    return request.param


@pytest.fixture(params=[True, False])
def gradient(request):
    return request.param


def function_space(r_space):
    """
    Construct a function space on a simple one
    element mesh of the unit square.
    """
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    element = ("R", 0) if r_space else ("CG", 1)
    return FunctionSpace(mesh, *element)


def try_compute_full_hessian(*args):
    try:
        return compute_full_hessian(*args)
    except NotImplementedError:
        pytest.xfail("Full Hessian R-space only")  # TODO


def test_matrix_ops(r_space):
    """
    Test that all of the operations associated with
    :class:`Matrix` work correctly.
    """
    fs = function_space(r_space)

    # Create three matrices
    A, B, C = Matrix(fs), Matrix(fs), Matrix(fs)

    # Set them to be scalings of the identity
    I = np.eye(A.n)
    A.set(1.0 * I)
    assert np.allclose(A.array, I)
    B.set(2.0 * I)
    C.set(3.0 * I)

    # Check that subtraction works
    C.subtract(B)
    assert np.allclose(A.array, C.array)

    # Check that addition works
    C.add(A)
    assert np.allclose(B.array, C.array)

    # Check that scaling works
    C.scale(0.5)
    assert np.allclose(A.array, C.array)

    # Check that vector multiplication works
    v = Function(fs).assign(1.0)
    w = B.multiply(v)
    expected = Function(fs).assign(2.0)
    assert np.isclose(errornorm(expected, w), 0.0)

    # Check that solving works
    w = B.solve(v)
    expected.assign(0.5)
    assert np.isclose(errornorm(expected, w), 0.0)


def test_hessian(r_space, gradient):
    """
    Test that :func:`compute_full_hessian` can successfully
    compute the Hessian of a simple expression.
    """
    fs = function_space(r_space)
    test = TestFunction(fs)
    x = Function(fs).assign(1.0)
    c = Control(x)

    # Define some functional that depends only on the control
    J = assemble((x ** 2 + x + 1) * dx)

    # Compute its gradient and check the accuracy
    if gradient:
        g = compute_gradient(J, c)
        dJdx = assemble(test * (2 * x + 1) * dx)
        assert np.isclose(errornorm(g, dJdx), 0)

    # Compute its Hessian and check the accuracy
    H = try_compute_full_hessian(J, c)
    d2Jdx2 = Matrix(fs).set(2.0)
    assert np.allclose(H.array, d2Jdx2.array)
