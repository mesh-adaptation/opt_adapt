import firedrake as fd
import firedrake_adjoint as fd_adj
from firedrake.adjoint import get_solve_blocks
import ufl

from opt_adapt.utils import pprint

import numpy as np
from time import perf_counter


__all__ = ["OptimisationProgress", "OptAdaptParameters", "identity_mesh", "get_state", "minimise"]


class OptimisationProgress:
    """
    Class for stashing progress of the optimisation
    routine.
    """

    def __init__(self):
        self.J_progress = []
        self.m_progress = []
        self.dJdm_progress = []


class OptAdaptParameters:
    """
    Class for holding parameters associated with the
    combined optimisation-adaptation routine.
    """

    def __init__(self, options={}):
        self.model_options = {}
        self.disp = 0
        self.lr = 0.001  # Step length / learning rate
        self.transfer_fn = fd.project  # Mesh-to-mesh interpolation method

        self.maxiter = 101  # Maximum iteration count
        self.gtol = 1.0e-05  # Gradient relative tolerance
        self.dtol = 1.0001  # Divergence tolerance i.e. 0.01% increase
        self.element_rtol = 0.005  # Element count relative tolerance
        self.qoi_rtol = 0.005  # QoI relative tolerance

        self.target_base = 200.0  # Base target metric complexity
        self.target_inc = 200.0  # Increment for target metric complexity
        self.target_max = 1000.0  # Eventual target metric complexity

        # Apply user-specified values
        for key, value in options.items():
            if not hasattr(self, key):
                raise ValueError(f"Option {key} not recognised")
            self.__setattr__(key, value)


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
    Rspace = fs.ufl_element().family() == "Real"
    if u.block_variable.adj_value is None:
        Jhat.derivative()
    if Rspace:
        h = fd.Function(fs).assign(1.0)
        return Jhat.hessian(h)
    else:
        raise NotImplementedError("Full Hessian only supported for R-space")


def _gradient_descent(it, forward_run, m, options, u, u_, dJ_, Rspace=False):
    """
    Take one gradient descent iteration.

    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :arg options: a dictionary of parameters for the
        optimisation and adaptation routines
    :arg u: the current control value
    :arg u_: the previous control value
    :arg dJ_: the previous gradient value
    :kwarg Rspace: is the prognostic function
        space of type 'Real'?
    """
    model_options = options.get("model_options", {})

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u, **model_options)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield {"J": J, "u": u.copy(deepcopy=True), "dJ": dJ.copy(deepcopy=True)}

    # Choose step length
    if u_ is None or dJ_ is None:
        lr = options.get("lr")
    else:
        if Rspace:
            dJ_ = fd.Function(dJ).assign(dJ_)
            u_ = fd.Function(u).assign(u_)
        else:
            transfer_fn = options.get("transfer_fn")
            dJ_ = transfer_fn(dJ_, dJ.function_space())
            u_ = transfer_fn(u_, u.function_space())
        dJ_diff = fd.assemble(ufl.inner(dJ_ - dJ, dJ_ - dJ) * ufl.dx)
        lr = abs(fd.assemble(ufl.inner(u_ - u, dJ_ - dJ) * ufl.dx) / dJ_diff)

    # Take a step downhill
    u -= lr * dJ
    yield {"lr": lr, "u+": u, "u-": u_, "dJ-": dJ_}


_implemented_methods = {
    "gradient_descent": {"func": _gradient_descent, "order": 1},
}


def identity_mesh(mesh, **kwargs):
    """
    The simplest possible adaptation function: the
    identity function.
    """
    return mesh


def get_state(adjoint=False, tape=None):
    """
    Extract the current state from the tape (velocity and
    elevation).

    :kwarg adjoint: If ``True``, return the corresponding
        adjoint state variables.
    """
    solve_block = get_solve_blocks()[0]
    return solve_block.adj_sol if adjoint else solve_block._outputs[0].saved_output


def minimise(
    forward_run,
    mesh,
    initial_control_fn,
    adapt_fn=identity_mesh,
    options={},
    method="gradient_descent",
    op=None,
):
    """
    Custom minimisation routine, where the tape is
    re-annotated each iteration in order to support
    mesh adaptation.

    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg mesh: the initial mesh
    :arg init_control_fn: a Python function that takes
        a mesh as input and initialises the control
    :kwarg adapt_fn: a Python function that takes a
        mesh as input and adapts it to get a new mesh
    :kwarg options: a dictionary of parameters for the
        optimisation and adaptation routines
    :kwarg method: the optimisation method
    :kwarg op: optional :class:`OptimisationProgress`
        instance
    """
    try:
        step, order = _implemented_methods[method].values()
    except KeyError:
        raise ValueError(f"Method '{method}' not recognised")
    op = op or OptimisationProgress()
    tape = fd_adj.get_working_tape()
    tape.clear_tape()
    u_plus = initial_control_fn(mesh)
    Rspace = u_plus.ufl_element().family() == "Real"
    dJ_init = None

    # Process parameters  # TODO: Create a separate class for these
    options.setdefault("model_options", {})
    options.setdefault("lr", 0.001)
    options.setdefault("transfer_fn", fd.project)  # mesh-to-mesh interpolation method
    maxiter = options.get("maxiter", 101)
    gtol = options.get("gtol", 1.0e-05)
    dtol = options.get("dtol", 1.0001)  # i.e. 0.01% increase in QoI
    element_rtol = options.get("element_rtol", 0.005)
    qoi_rtol = options.get("qoi_rtol", 0.005)
    disp = options.get("disp", 0)
    target = options.get("target_base", 200.0)
    target_inc = options.get("target_inc", 200.0)
    target_max = options.get("target_max", 1000.0)

    # Enter the optimisation loop
    nc_ = mesh.num_cells()
    adaptor = adapt_fn
    mesh_conv_it = []
    for it in range(1, maxiter + 1):
        term_msg = f"Terminated after {it} iterations due to "
        u_ = None if it == 1 else op.m_progress[-1]
        dJ_ = None if it == 1 else op.dJdm_progress[-1]
        if order == 1:
            args = (u_plus, u_, dJ_)
        else:
            raise NotImplementedError(f"Only order 1 methods are supported, not {order}")

        # Take a step
        cpu_timestamp = perf_counter()
        out = {}
        for o in step(it, forward_run, mesh, options, *args, Rspace=Rspace):
            out.update(o)
        J, u, dJ = out["J"], out["u"], out["dJ"]
        lr, u_plus, u_, dJ_ = out["lr"], out["u+"], out["u-"], out["dJ-"]
        if disp > 0:
            t = perf_counter() - cpu_timestamp
            g = dJ.dat.data[0] if Rspace else fd.norm(dJ)
            msgs = [f"{it:3d}:  J = {J:9.4e}"]
            if Rspace:
                msgs.append(f"m = {u_plus.dat.data[0]:9.4e}")
            if Rspace:
                msgs.append(f"dJdm = {g:11.4e}")
            else:
                msgs.append(f"||dJdm|| = {g:9.4e}")
            msgs.append(f"step length = {lr:9.4e}")
            msgs.append(f"#elements = {nc_:5d}")
            msgs.append(f"time = {t:.2f}s")
            pprint(",  ".join(msgs))

        # Stash progress
        op.J_progress.append(J)
        op.m_progress.append(u)
        op.dJdm_progress.append(dJ)

        # Check for QoI divergence
        if it > 1 and np.abs(J / np.min(op.J_progress)) > dtol:
            raise fd.ConvergenceError(term_msg + "dtol divergence")

        # Check for gradient convergence
        if it == 1:
            dJ_init = fd.norm(dJ)
        elif fd.norm(dJ) / dJ_init < gtol:
            if disp > 0:
                pprint(term_msg + "gtol convergence")
            break

        # Check for reaching maximum iteration count
        if it == maxiter:
            raise fd.ConvergenceError(term_msg + "reaching maxiter")

        # Adapt the mesh
        target = min(target + target_inc, target_max)  # Ramp up the target complexity
        mesh = adaptor(mesh, target=target, control=u_plus)
        nc = mesh.num_cells()

        # Check for mesh convergence
        if adaptor != identity_mesh and np.abs(nc - nc_) < element_rtol * nc_:
            conv = np.array([op.J_progress[i] for i in mesh_conv_it])
            if (np.abs(J - conv) < qoi_rtol * np.abs(conv)).any():
                pprint(term_msg + "qoi_rtol convergence")
                break
            mesh_conv_it.append(it)
            if disp > 1:
                pprint("NOTE: turning adaptation off due to element_rtol convergence")
            adaptor = identity_mesh
            continue
        else:
            adaptor = adapt_fn
        nc_ = nc

        # Check for QoI convergence
        if it > 1:
            J_ = op.J_progress[-2]
            if np.abs(J - J_) < qoi_rtol * np.abs(J_):
                if adaptor != identity_mesh:
                    conv = np.array([op.J_progress[i] for i in mesh_conv_it])
                    if (np.abs(J - conv) < qoi_rtol * np.abs(conv)).any():
                        pprint(term_msg + "qoi_rtol convergence")
                        break
                mesh_conv_it.append(it)
                if disp > 1:
                    pprint("NOTE: turning adaptation off due to qoi_rtol convergence")
                adaptor = identity_mesh
                continue
            else:
                adaptor = adapt_fn

        # Clean up
        tape.clear_tape()
    return u_plus
