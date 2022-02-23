import firedrake as fd
import firedrake_adjoint as fd_adj
import ufl

from opt_adapt.utils import pprint

import numpy as np
from time import perf_counter


__all__ = ["OptimisationProgress", "identity_mesh", "minimise"]


class OptimisationProgress(object):
    """
    Class for stashing progress of the optimisation
    routine.
    """

    J_progress = []
    m_progress = []
    dJdm_progress = []


def _gradient_descent(it, forward_run, m, u, u_, dJ_, options, Rspace=False):
    """
    Take one gradient descent iteration.

    :arg it: the current iteration number
    :arg forward_run: a Python function that
        implements the forward model and
        computes the objective functional
    :arg m: the current mesh
    :arg u: the current control value
    :arg u_: the previous control value
    :arg dJ_: the previous gradient value
    :arg options: a dictionary of parameters for the
        optimisation and adaptation routines
    :kwarg Rspace: is the prognostic function
        space of type 'Real'?
    """

    # Annotate the tape and compute the gradient
    J, u = forward_run(m, u)
    dJ = fd_adj.compute_gradient(J, fd_adj.Control(u))
    yield J, u.copy(deepcopy=True), dJ.copy(deepcopy=True)

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
    yield lr, u, u_, dJ_


_implemented_methods = {
    "gradient_descent": _gradient_descent,
}


def identity_mesh(mesh, **kwargs):
    """
    The simplest possible adaptation function: the
    identity function.
    """
    return mesh


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
        step = _implemented_methods[method]
    except KeyError:
        raise ValueError(f"Method '{method}' not recognised")
    op = op or OptimisationProgress()
    tape = fd_adj.get_working_tape()
    tape.clear_tape()
    u = initial_control_fn(mesh)
    Rspace = u.ufl_element().family() == "Real"
    dJ_init = None

    # Process parameters
    options.setdefault("lr", 0.001)
    options.setdefault("transfer_fn", fd.project)  # mesh-to-mesh interpolation method
    maxiter = options.get("maxiter", 101)
    gtol = options.get("gtol", 1.0e-05)
    dtol = options.get("dtol", 1.0001)  # i.e. 0.01 % increase in QoI
    element_rtol = options.get("element_rtol", 0.001)
    qoi_rtol = options.get("qoi_rtol", 0.001)
    disp = options.get("disp", 0)
    target = options.get("target_base", 200.0)
    target_inc = options.get("target_inc", 200.0)
    target_max = options.get("target_max", 1000.0)

    # Enter the optimisation loop
    nc_ = mesh.num_cells()
    for it in range(1, maxiter + 1):
        term_msg = f"Terminated after {it} iterations due to "
        u_ = None if it == 1 else op.m_progress[-1]
        dJ_ = None if it == 1 else op.dJdm_progress[-1]

        # Take a step
        cpu_timestamp = perf_counter()
        out1, out2 = tuple(
            step(it, forward_run, mesh, u, u_, dJ_, options, Rspace=Rspace)
        )
        lr, u, u_, dJ_ = out2
        if disp > 0:
            t = perf_counter() - cpu_timestamp
            g = out1[2].dat.data[0] if Rspace else fd.norm(out1[2])
            msgs = [f"{it:3d}:  J = {out1[0]:9.4e}"]
            if Rspace:
                msgs.append(f"m = {out1[1].dat.data[0]:.2f}")
            if Rspace:
                msgs.append(f"dJdm = {g:11.4e}")
            else:
                msgs.append(f"||dJdm|| = {g:9.4e}")
            msgs.append(f"step length = {lr:9.4e}")
            msgs.append(f"#elements = {nc_:5d}")
            msgs.append(f"time = {t:.2f}s")
            pprint(",  ".join(msgs))

        # Stash progress
        op.J_progress.append(out1[0])
        op.m_progress.append(out1[1])
        op.dJdm_progress.append(out1[2])

        # Check for QoI divergence
        if it > 1 and np.abs(op.J_progress[-1] / np.min(op.J_progress)) > dtol:
            raise fd.ConvergenceError(term_msg + "dtol divergence")

        # Check for gradient convergence
        if it == 1:
            dJ_init = fd.norm(op.dJdm_progress[-1])
        elif fd.norm(op.dJdm_progress[-1]) / dJ_init < gtol:
            if disp > 0:
                pprint(term_msg + "gtol convergence")
            break

        # Check for reaching maximum iteration count
        if it == maxiter:
            raise fd.ConvergenceError(term_msg + "reaching maxiter")

        # Adapt the mesh
        if adapt_fn == identity_mesh:
            continue
        target = min(target + target_inc, target_max)  # Ramp up the target complexity
        mesh = adapt_fn(mesh, target=target, control=u)
        nc = mesh.num_cells()

        # Check for mesh convergence
        if np.abs(nc - nc_) < element_rtol * nc_:
            if disp > 1:
                pprint("NOTE: turning adaptation off due to element_rtol convergence")
            adapt_fn = identity_mesh
        nc_ = nc

        # Check for QoI convergence
        if it > 1:
            qoi = op.J_progress[-1]
            qoi_ = op.J_progress[-2]
            if np.abs(qoi - qoi_) < qoi_rtol * qoi_:
                if disp > 1:
                    pprint("NOTE: turning adaptation off due to qoi_rtol convergence")
                adapt_fn = identity_mesh

        # Clean up
        tape.clear_tape()
    return u
