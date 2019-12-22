from typing import List

import numpy as np
from scipy.sparse import diags, block_diag, eye, kron, vstack, hstack, csr_matrix

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

        # nx by nx+1 discrete divergence matrix
        diagonals = np.array([0, 1])
        div_x = 1. / grid.grid_size[0] * diags(diagonals=[-ex, ex], offsets=diagonals,
                                               shape=(grid.n_cell_dofs[0], grid.n_cell_dofs[0] + 1))
        div_y = 1. / grid.grid_size[1] * diags(diagonals=[-ey, ey], offsets=diagonals,
                                               shape=(grid.n_cell_dofs[1], grid.n_cell_dofs[1] + 1))

        ix = eye(grid.n_cell_dofs[0])
        iy = eye(grid.n_cell_dofs[1])

        # kron: kronecker product of sparse matrices A and B
        div_x = kron(A=div_x, B=iy, format='csr')
        div_y = kron(A=ix, B=div_y, format='csr')

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
        # TODO Changing the sparsity structure of a csc_matrix is expensive. Should use lil_matrix?
        G[dof_flux_bnd.astype(int), :] = 0
        I = eye(grid.n_cell_dofs_total, format="csr")
        return D, G, I

    @staticmethod
    def build_boundary_operators(grid: StaggeredGrid, param: Parameters, I):
        """
        This function computes the operators and r.h.s vectors for both Dirichlet and Neumann boundary conditions.
        @param grid: structure containing all pertinent information about the grid
        @param param: structure containing problem parameters and information about BCs
        @param I: N by N identity matrix
        @return: B: Nc by N matrix of the Dirichlet constraints
                 N: N by (N-Nc) matrix of the null space of B
                 fn: N by 1 r.h.s. vector of Neumann contributions
        """

        B = I[param.unique_dof_dirichelt, :]
        mask = np.ones(grid.n_cell_dofs_total, dtype=bool)
        if param.unique_dof_dirichelt.size > 0:
            mask[param.unique_dof_dirichelt] = False
        N = I[:, mask]
        fn = np.zeros(grid.n_cell_dofs_total)
        if param.qb.size > 0:
            dxy = grid.volume[param.dof_neumann] / grid.area[param.dof_neumann_face]
            # Use loop here because param.dof_neumann may have duplicated indices
            for i, j in enumerate(param.dof_neumann):
                fn[j] += param.qb[i] / dxy[i]
        return csr_matrix(B), csr_matrix(N), fn

    @staticmethod
    def compute_flux(D, Kd, G, h, fs, grid: StaggeredGrid, param: Parameters):
        """
        Computes the mass conservative fluxes across all boundaries from the
        residual of the compatibility condition over the boundary cells.
        @param D:  N by Nf discrete divergence matrix
        @param Kd: Nf by Nf conductivity matrix
        @param G: Nf by N discrete gradient matrix
        @param h: N by 1 vector of flow potential in cell centers
        @param fs: N by 1 right hand side vector containing only source terms
        @param grid: structure containing grid information
        @param param: structure containing problem parameters and information about BCs
        @return: fluxes
        """
        q = -Kd * G * h

        if param.dof_dirichlet.size > 0:
            if grid.is_problem_1d() or grid.is_problem_2d():
                # Use mass balance to in boundary cells to attain exact flux
                q[param.dof_dirichlet_face] = (D[param.dof_dirichlet, :] * q - fs[param.dof_dirichlet]) * \
                                              grid.volume[param.dof_dirichlet] / grid.area[param.dof_dirichlet_face]
                q[grid.idx_flux_dofs_xmax] *= -1.
                q[grid.idx_flux_dofs_ymax] *= -1.
            else:
                raise ValueError("3d is not implemented.")

        if param.dof_neumann.size > 0:
            q[param.dof_neumann_face] = param.qb
        return q

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
                k = k.flatten()
                mean[1:-1] = compute_mean(k[:-1], k[1:])
                return diags(diagonals=[mean], offsets=[0],
                             shape=(grid.n_cell_dofs[idx] + 1, grid.n_cell_dofs[idx] + 1))
            elif grid.is_problem_2d():
                mean_x = np.zeros((grid.n_cell_dofs[0] + 1, grid.n_cell_dofs[1]))
                mean_x[1:-1, :] = compute_mean(k[:-1, :], k[1:, :])

                mean_y = np.zeros((grid.n_cell_dofs[0], grid.n_cell_dofs[1] + 1))
                mean_y[:, 1:-1] = compute_mean(k[:, :-1], k[:, 1:])

                mean = np.hstack([mean_x.flatten(), mean_y.flatten()])
                return diags(diagonals=mean, offsets=0, shape=(grid.n_flux_dofs_total, grid.n_flux_dofs_total))
            else:
                raise ValueError("3d coefficient field is not implemented.")
        else:
            raise ValueError("Power has to be either 1 or -1")

    @staticmethod
    def flux_upwind(q, grid: StaggeredGrid):
        """
        This function computes the upwind flux matrix from the flux vector.
        @param q: Nf by 1 flux vector from the flow problem
        @param grid: structure containing all pertinent information about the grid
        @return: Nf by N matrix containing the upwinded fluxes
        """
        if grid.is_problem_1d():
            idx = 0 if grid.n_cell_dofs[0] == grid.n_cell_dofs_total else 1
            qn = np.minimum(q[:grid.n_cell_dofs[idx]], 0)
            qp = np.maximum(q[1:], 0)
            return diags(diagonals=[qp, qn], offsets=[-1, 0], shape=(grid.n_cell_dofs[idx] + 1, grid.n_cell_dofs[idx]))
        elif grid.is_problem_2d():
            qn_x = np.minimum(q[:grid.n_flux_dofs[0] - grid.n_cell_dofs[1]], 0)
            qp_x = np.maximum(q[grid.n_cell_dofs[1]:grid.n_flux_dofs[0]], 0)
            offsets = [-grid.n_cell_dofs[1], 0]
            A_x = diags(diagonals=[qp_x, qn_x], offsets=offsets,
                        shape=(grid.n_flux_dofs[0], grid.n_flux_dofs[0] - grid.n_cell_dofs[1]))
            A_y = []
            for i in range(grid.n_cell_dofs[0]):
                step = grid.n_flux_dofs[0] + i * (grid.n_cell_dofs[1] + 1)
                qn_y = np.minimum(q[step:step + grid.n_cell_dofs[1]], 0)
                qp_y = np.maximum(q[step + 1:step + grid.n_cell_dofs[1] + 1], 0)
                A_y.append(diags(diagonals=[qp_y, qn_y], offsets=[-1, 0],
                                 shape=(grid.n_cell_dofs[1] + 1, grid.n_cell_dofs[1])))
            A_y = block_diag(mats=tuple(A_y), format='csr')
            return vstack([A_x, A_y], format='csr')
        else:
            raise ValueError("3d flux is not implemented.")
