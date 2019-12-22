import numpy as np
from scipy.sparse.linalg import spsolve


def solve_linear_boundary_value_problem(L, fs: np.ndarray, B, g: np.ndarray,
                                        N) -> np.ndarray:
    """
    Compute the numerical solution u of the linear differential problem:
    Lu = f
    with boundary condition:
    Bu=g
    @param L: matrix representing the discretized linear operator of size N by N, where N is the number of degrees of freedom
    @param fs: column vector representing the discretized r.h.s. and contributions due non-homogeneous Neumann BCs of size N by 1
    @param B: matrix representing the constraints arising from Dirichlet BCs of size Nc by N
    @param g: column vector representing the non-homogeneous Dirichlet BCs of size Nc by 1
    @param N: matrix representing a orthonormal basis for the null-space of B and of size N by (N-Nc).
    @return: column vector of the numerical_solution of size N by 1
    """

    # TODO implement LU factorization
    # TODO implement Cholesky factorization for SPD matrix
    hpr = spsolve(B * B.transpose(), g)
    Lr = N.transpose() * L * N
    fd = -L * B.transpose() * hpr
    f = fs + fd
    fr = N.transpose() * f
    hor = spsolve(Lr, fr)
    return N * hor + B.transpose() * hpr
