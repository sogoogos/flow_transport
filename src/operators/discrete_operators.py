from typing import List

import numpy as np
from scipy.sparse import diags, eye, kron, hstack, csr_matrix

from grids.staggered_grid import StaggeredGrid
from parameters.parameters import Parameters


class Operators:
    @staticmethod
    def build_discrete_operators(grid: StaggeredGrid):
        """
        Create discrete divergence and gradient operators based on the grids specified
        @param grid: structure containing information about the grid
        @return: D: N by N+1 discrete divergence matrix
                 G: N+1 by N discrete gradient matrix
                 I: N by N identity matrix
        """
        ex = np.ones(grid.n_cell_dofs[0])
        ey = np.ones(grid.n_cell_dofs[1])

        # nx by nx+1 discrete div matrix
        diagonals = np.array([0, 1])
        div_x = 1. / grid.grid_size[0] * diags(diagonals=np.array([-ex, ex]), offsets=diagonals,
                                               shape=(grid.n_cell_dofs[0], grid.n_cell_dofs[0] + 1))
        div_y = 1. / grid.grid_size[1] * diags(diagonals=[-ey, ey], offsets=diagonals,
                                               shape=(grid.n_cell_dofs[1], grid.n_cell_dofs[1] + 1))

        ix = eye(grid.n_cell_dofs[0])
        iy = eye(grid.n_cell_dofs[1])

        # kron: kronecker product of sparse matrices A and B
        div_x = kron(A=div_x, B=iy)
        div_y = kron(A=ix, B=div_y)

        if grid.n_cell_dofs[0] > 1 and grid.n_cell_dofs[1] > 1:
            D = hstack([div_x, div_y], format="csr")
        elif grid.n_cell_dofs[0] > 1:
            D = div_x
        elif grid.n_cell_dofs[1] > 1:
            D = div_y
        else:
            ValueError("Cannot solve one grid problem.")

        G = -D.transpose()
        # this makes index float somehow
        dof_flux_bnd = np.concatenate([grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax, grid.idx_flux_dofs_ymin,
                                       grid.idx_flux_dofs_ymax])
        # remove boundary terms
        G[dof_flux_bnd.astype(int), :] = 0
        I = eye(grid.n_cell_dofs_total, format="csr")
        return D, G, I

    @staticmethod
    # TODO I is sparse matrix
    def build_boundary_operators(grid: StaggeredGrid, param: Parameters, I):
        """
        This function computes the operators and r.h.s vectors for both Dirichlet and Neumann boundary conditions.
        @param grid: structure containing all pertinent information about the grid
        @param param: structure containing all information about the physical problem
        @param I: N by N identity matrix
        @return: B: Nc by N matrix of the Dirichlet constraints
                 N: N by (N-Nc) matrix of the null space of B
                 fn: N by 1 r.h.s. vector of Neumann contributions
        """

        B = I[param.dof_dirichlet, :]
        mask = np.ones(grid.n_cell_dofs_total, dtype=bool)
        mask[param.dof_dirichlet] = False
        N = I[:, mask]
        fn = np.zeros(grid.n_cell_dofs_total)
        if param.qb.size > 0:
            dxy = grid.volume[param.dof_neumann] / grid.area[param.dof_neumann_face]
            fn[param.dof_neumann] = param.qb / dxy
        return csr_matrix(B), csr_matrix(N), fn

    @staticmethod
    def compute_mean(k: List, power: int, grid: StaggeredGrid):
        """
        Takes coefficient field, k, defined at the cell centers and computes the
        mean specified by the power and returns it in a sparse diagonal matrix, Kd.
        @param k:  Ny by Nx column vector of cell centered values
        @param power: power of the generalized mean
                1 (arithmetic mean)
                -1 (harmonic mean)
        @param grid: structure containing information about the grid.
        @return: Nf by Nf diagonal matrix of power means at the cell faces.
        """

        def compute_mean(left, right):
            return np.power(0.5 * (np.power(left, power) + np.power(right, power)), power)

        k = np.asarray(k, dtype=np.float64)
        if power == -1 or power == 1:
            if grid.is_problem_1d():
                idx = 0 if grid.n_cell_dofs[0] == grid.n_cell_dofs_total else 1
                mean = np.zeros(grid.n_cell_dofs[idx] + 1)
                mean[1:-1] = compute_mean(k[:-1], k[1:])
                return diags(diagonals=[mean], offsets=[0], shape=(grid.n_cell_dofs[idx] + 1, grid.n_cell_dofs[idx] + 1))
            elif grid.is_problem_2d():
                mean_x = np.zeros((grid.n_cell_dofs[1], grid.n_cell_dofs[0] + 1))
                mean_x[:, 1:-1] = compute_mean(k[:, :-1], k[:, 1:])

                mean_y = np.zeros((grid.n_cell_dofs[1] + 1, grid.n_cell_dofs[0]))
                mean_y[1:-1, :] = compute_mean(k[:-1, :], k[1:, :])

                mean = np.hstack([mean_x.flatten(), mean_y.flatten()])
                return diags(diagonals=mean, offsets=0, shape=(grid.n_flux_dofs_total, grid.n_flux_dofs_total))
            else:
                raise ValueError("3d coefficient field is not implemented.")
        else:
            raise ValueError("Power has to be either 1 or -1")
