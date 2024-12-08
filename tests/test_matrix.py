"""
Tests for the :class:`Matrix` class.
"""

import numpy as np
import pytest
import ufl
from animate import errornorm
from firedrake.adjoint import pyadjoint
from firedrake.assemble import assemble
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.utility_meshes import UnitSquareMesh
from pyadjoint.control import Control
from pyadjoint.drivers import compute_gradient

from opt_adapt.matrix import Matrix, OuterProductMatrix, compute_full_hessian


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


def test_matrix_ops(r_space):
    """
    Test that all of the operations associated with
    :class:`Matrix` work correctly.
    """
    fs = function_space(r_space)

    # Create three matrices
    A, B, C = Matrix(fs), Matrix(fs), Matrix(fs)

    # Set them to be scalings of the identity
    identity = np.eye(A.n)
    A.set(1.0 * identity)
    assert np.allclose(A.array, identity)
    B.set(2.0 * identity)
    C.set(3.0 * identity)

    # Check that subtraction works
    C.subtract(B)
    assert np.allclose(A.array, C.array)

    # Check that addition works
    C.add(A)
    assert np.allclose(B.array, C.array)

    # Likewise for += and -=
    C -= A
    assert np.allclose(A.array, C.array)
    C += A
    assert np.allclose(B.array, C.array)

    # Check that scaling works
    C.scale(0.5)
    assert np.allclose(A.array, C.array)

    # Check matrix-vector multiplication works
    v = Function(fs).assign(1.0)
    w = B.multiply(v)
    expected = Function(fs).assign(2.0)
    assert np.isclose(errornorm(expected, w), 0.0)

    # Check that solving works
    w = B.solve(v)
    expected.assign(0.5)
    assert np.isclose(errornorm(expected, w), 0.0)

    # Check that the outer product works
    A = OuterProductMatrix(v, v)
    B.set(1.0)
    assert np.allclose(A.array, B.array)

    # Check vector-matrix multiplication works
    A.set(np.random.rand(*A.shape))
    w = A.multiply(v, side="left")
    A.transpose()
    expected = A.multiply(v)
    assert np.isclose(errornorm(expected, w), 0.0)


def test_hessian(r_space, gradient):
    """
    Test that :func:`compute_full_hessian` can successfully
    compute the Hessian of a simple expression.
    """
    fs = function_space(r_space)
    test = TestFunction(fs)
    x, y = ufl.SpatialCoordinate(fs.mesh())
    X = Function(fs).interpolate(x)
    c = Control(X)

    # Define some functional that depends only on the control
    J = assemble((X**3 + X**2 + X + 1) * ufl.dx)

    # Compute its gradient and check the accuracy
    pyadjoint.pause_annotation()
    lhs = TrialFunction(fs) * test * ufl.dx
    if gradient:
        g = compute_gradient(J, c)
        rhs = test * (3 * X**2 + 2 * X + 1) * ufl.dx
        dJdX = Function(fs)
        solve(lhs == rhs, dJdX)
        assert np.isclose(errornorm(g, dJdX), 0)

    # Compute its Hessian and check the accuracy
    H = compute_full_hessian(J, c)
    rhs = test * (6 * X + 2) * ufl.dx
    d2JdX2 = Function(fs)
    solve(lhs == rhs, d2JdX2)
    expected = Matrix(fs).set(d2JdX2.dat.data)
    assert np.allclose(H.array, expected.array)
    pyadjoint.continue_annotation()
