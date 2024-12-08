import os

from firedrake.petsc import PETSc

pprint = PETSc.Sys.Print


def create_directory(path, comm=PETSc.COMM_WORLD):
    """
    Create a directory on disk.

    Code copied from `Thetis
    <https://thetisproject.org>`__.

    :arg path: path to the directory
    :kwarg comm: MPI communicator
    """
    if comm.rank == 0:
        if not os.path.exists(path):
            os.makedirs(path)
    comm.barrier()
    return path
