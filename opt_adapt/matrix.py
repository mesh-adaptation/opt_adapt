import firedrake as fd
import firedrake_adjoint as fd_adj
import numpy as np


__class__ = ["Matrix", "compute_full_hessian"]


class Matrix:
    """
    Class for representing square matrix data.
    """

    # TODO: Use PETSc rather than NumPy

    def __init__(self, function_space):
        """
        :arg function_space: :class:`FunctionSpace` of
            the :class:`Control` variable that the
            matrix is associated with
        """
        if function_space.mesh().comm.size > 1:
            raise ValueError("Matrix class implemented in serial only")
        self.function_space = function_space
        self.n = function_space.dof_count
        self.shape = (self.n, self.n)
        self.array = np.eye(self.n)

    def __repr__(self):
        return self.array.__repr__()

    def set_row(self, row, vals):
        """
        Set the values of a given row.

        :arg row: the row index
        :arg vals: either a :class:`Function` or
            an appropriate data array
        """
        if isinstance(vals, fd.Function):
            self.array[row, :] = vals.dat.data
        elif isinstance(vals, np.ndarray):
            self.array[row, :] = vals
        else:
            self.array[row, :] = vals * np.ones(self.n)

    def set(self, val):
        """
        Set the values of the whole matrix.

        :arg: val: either a NumPy array or a scalar
            number
        """
        if isinstance(val, np.ndarray):
            self.array[:, :] = val
        else:
            self.array[:, :] = val * np.ones(self.shape)
        return self

    def scale(self, alpha):
        """
        Scale by some number.

        :arg alpha: the scalar number
        """
        self.array[:, :] *= alpha
        return self

    def add(self, other):
        """
        Add with another matrix.

        :arg other: the other :class:`Matrix` instance
        """
        assert isinstance(other, Matrix)
        self.array[:, :] += other.array

    def subtract(self, other):
        """
        Subtract another matrix.

        :arg other: the other :class:`Matrix` instance
        """
        assert isinstance(other, Matrix)
        self.array[:, :] -= other.array

    def multiply(self, v):
        """
        Compute a matrix-vector product.

        :arg v: either a :class:`Function` instance
            or an appropriate data array
        """
        if isinstance(v, fd.Function):
            v = v.dat.data
        Av = np.dot(self.array, v)
        out = fd.Function(self.function_space)
        out.dat.data[:] = Av.reshape(v.shape)
        return out

    def solve(self, f):
        """
        Solve a linear system for some right-hand
        side.

        :arg f: the RHS, either as a :class:`Function`
            instance, or an appropriate data array
        """
        if isinstance(f, fd.Function):
            f = f.dat.data.flatten()
        u = np.linalg.solve(self.array, f)
        solution = fd.Function(self.function_space)
        solution.dat.data[:] = u.reshape(f.shape)
        return solution


def compute_full_hessian(J, u):
    """
    Compute the full Hessian of a functional
    w.r.t. a control.

    :arg J: the functional
    :arg u: the :class:`Control`
    """
    if not isinstance(u, fd_adj.Control):
        raise ValueError(f"Second argument should be a Control, not {type(u)}")
    Jhat = fd_adj.ReducedFunctional(J, u)
    fs = u.data().function_space()
    H = Matrix(fs)
    Rspace = fs.ufl_element().family() == "Real"
    if u.block_variable.adj_value is None:
        Jhat.derivative()
    if Rspace:
        h = fd.Function(fs).assign(1.0)
        H.set(Jhat.hessian(h).dat.data)
    else:
        raise NotImplementedError("Full Hessian only supported for R-space")
    return H