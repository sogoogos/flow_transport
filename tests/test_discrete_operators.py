import unittest

import numpy as np
from scipy.sparse import csr_matrix, diags

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators
from parameters.parameters import Parameters


class TestOperators(unittest.TestCase):
    def test_build_discrete_operators(self):
        # 1D
        grid = StaggeredGrid(size=[3, 1], dimensions=[3, 1])
        D, G, I = Operators.build_discrete_operators(grid)
        D_ans = csr_matrix((grid.n_cell_dofs_total, grid.n_flux_dofs_total))
        G_ans = csr_matrix((grid.n_flux_dofs_total, grid.n_cell_dofs_total))
        I_ans = diags(diagonals=[np.ones(grid.n_cell_dofs_total)], offsets=[0],
                      shape=(grid.n_cell_dofs_total, grid.n_cell_dofs_total))

        D_ans[0, 0] = -1.
        D_ans[1, 1] = -1.
        D_ans[2, 2] = -1.
        D_ans[0, 1] = 1.
        D_ans[1, 2] = 1.
        D_ans[2, 3] = 1.
        G_ans[1, 1] = 1.
        G_ans[2, 2] = 1.
        G_ans[1, 0] = -1.
        G_ans[2, 1] = -1.

        np.testing.assert_array_almost_equal(D.toarray(), D_ans.toarray())
        np.testing.assert_array_almost_equal(G.toarray(), G_ans.toarray())
        np.testing.assert_array_almost_equal(I.toarray(), I_ans.toarray())

        # 2D
        grid = StaggeredGrid(size=[2, 2], dimensions=[2, 2])
        D, G, I = Operators.build_discrete_operators(grid)
        D_ans = csr_matrix((grid.n_cell_dofs_total, grid.n_flux_dofs_total))
        G_ans = csr_matrix((grid.n_flux_dofs_total, grid.n_cell_dofs_total))
        I_ans = diags(diagonals=[np.ones(grid.n_cell_dofs_total)], offsets=[0],
                      shape=(grid.n_cell_dofs_total, grid.n_cell_dofs_total))

        D_ans[0, 0] = -1.
        D_ans[1, 1] = -1.
        D_ans[2, 2] = -1.
        D_ans[3, 3] = -1.
        D_ans[0, 2] = 1.
        D_ans[1, 3] = 1.
        D_ans[2, 4] = 1.
        D_ans[3, 5] = 1.
        D_ans[0, 6] = -1.
        D_ans[1, 7] = -1.
        D_ans[2, 9] = -1.
        D_ans[3, 10] = -1.
        D_ans[0, 7] = 1.
        D_ans[1, 8] = 1.
        D_ans[2, 10] = 1.
        D_ans[3, 11] = 1.

        G_ans[2, 0] = -1.
        G_ans[3, 1] = -1.
        G_ans[2, 2] = 1.
        G_ans[3, 3] = 1.
        G_ans[7, 0] = -1.
        G_ans[7, 1] = 1.
        G_ans[10, 2] = -1.
        G_ans[10, 3] = 1.

        np.testing.assert_array_almost_equal(D.toarray(), D_ans.toarray())
        np.testing.assert_array_almost_equal(G.toarray(), G_ans.toarray())
        np.testing.assert_array_almost_equal(I.toarray(), I_ans.toarray())

    def test_build_boundary_operators(self):
        # 1D
        grid = StaggeredGrid(size=[3, 1], dimensions=[3, 1])
        _, _, I = Operators.build_discrete_operators(grid)

        # No boundary conditions
        param = Parameters()
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.dof_dirichlet), grid.n_cell_dofs_total))
        N_ans = I.copy()
        fn_ans = np.zeros(grid.n_cell_dofs_total)

        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

        # Dirithlet and zero Neumann
        param = Parameters(dir_cell=[0], dir_flux=[0])
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.dof_dirichlet), grid.n_cell_dofs_total))
        N_ans = csr_matrix((grid.n_cell_dofs_total, grid.n_cell_dofs_total - len(param.dof_dirichlet)))
        fn_ans = np.zeros(grid.n_cell_dofs_total)
        B_ans[0, 0] = 1.
        N_ans[1, 0] = 1.
        N_ans[2, 1] = 1.

        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

        # Non-zero Neumann
        param = Parameters(neu_cell=[0], neu_flux=[0], flux_bc=[100])
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.dof_dirichlet), grid.n_cell_dofs_total))
        N_ans = I.copy()
        fn_ans = np.zeros(grid.n_cell_dofs_total)
        fn_ans[param.dof_neumann] = 100
        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

        # 2D
        grid = StaggeredGrid(size=[3, 3], dimensions=[3, 3])
        _, _, I = Operators.build_discrete_operators(grid)

        # No boundary conditions
        param = Parameters()
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.dof_dirichlet), grid.n_cell_dofs_total))
        N_ans = I.copy()
        fn_ans = np.zeros(grid.n_cell_dofs_total)

        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

        # Dirithlet and zero Neumann
        dir_cell = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_ymax]
        dir_flux = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_ymax]
        param = Parameters(dir_cell=dir_cell, dir_flux=dir_flux)
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.unique_dof_dirichelt), grid.n_cell_dofs_total))
        N_ans = csr_matrix((grid.n_cell_dofs_total, grid.n_cell_dofs_total - len(param.unique_dof_dirichelt)))
        fn_ans = np.zeros(grid.n_cell_dofs_total)
        B_ans[0, 0] = 1.
        B_ans[1, 1] = 1.
        B_ans[2, 2] = 1.
        B_ans[3, 5] = 1.
        B_ans[4, 8] = 1.
        N_ans[3, 0] = 1.
        N_ans[4, 1] = 1.
        N_ans[6, 2] = 1.
        N_ans[7, 3] = 1.

        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

        # Non-zero Neumann
        neu_cell = [grid.idx_cell_dofs_xmax, grid.idx_cell_dofs_ymin]
        neu_flux = [grid.idx_flux_dofs_xmax, grid.idx_flux_dofs_ymin]
        flux_bc = np.full((len(np.unique(neu_flux)),), 100)
        flux_bc[:2] = -50
        param = Parameters(neu_cell=neu_cell, neu_flux=neu_flux, flux_bc=flux_bc)
        B, N, fn = Operators.build_boundary_operators(grid, param, I)
        B_ans = csr_matrix((len(param.unique_dof_dirichelt), grid.n_cell_dofs_total))
        N_ans = I.copy()
        fn_ans = np.zeros(grid.n_cell_dofs_total)
        fn_ans[0] = 100
        fn_ans[3] = 100
        fn_ans[6] = 50
        fn_ans[7] = -50
        fn_ans[8] = 100
        np.testing.assert_array_almost_equal(B.toarray(), B_ans.toarray())
        np.testing.assert_array_almost_equal(N.toarray(), N_ans.toarray())
        np.testing.assert_array_almost_equal(fn, fn_ans)

    def test_compute_mean(self):
        # 1D
        grid = StaggeredGrid(size=[3, 1], dimensions=[3, 1])
        k = [1, 3, 5]
        kd_harmonic = Operators.compute_mean(k, -1., grid)
        kd_arithmetic = Operators.compute_mean(k, 1., grid)
        size = (len(k) + 1)
        k_harmonic = csr_matrix((size, size))
        k_harmonic[1, 1] = 1.5
        k_harmonic[2, 2] = 3.75
        k_arithmetic = csr_matrix((size, size))
        k_arithmetic[1, 1] = 2.
        k_arithmetic[2, 2] = 4.
        np.testing.assert_array_almost_equal(kd_harmonic.toarray(), k_harmonic.toarray())
        np.testing.assert_array_almost_equal(kd_arithmetic.toarray(), k_arithmetic.toarray())

        # 2D
        grid = StaggeredGrid(size=[2, 2], dimensions=[2, 2])
        k = [[1, 3], [5, 9]]
        kd_harmonic = Operators.compute_mean(k, -1., grid)
        kd_arithmetic = Operators.compute_mean(k, 1., grid)
        shape = np.array(k).shape
        size = (shape[0] + 1) * shape[1] + (shape[1] + 1) * shape[0]

        k_harmonic = csr_matrix((size, size))
        k_harmonic[2, 2] = 1.6666666666666667
        k_harmonic[3, 3] = 4.5
        k_harmonic[7, 7] = 1.5
        k_harmonic[10, 10] = 6.428571428571429
        k_arithmetic = csr_matrix((size, size))
        k_arithmetic[2, 2] = 3.
        k_arithmetic[3, 3] = 6
        k_arithmetic[7, 7] = 2.
        k_arithmetic[10, 10] = 7.
        np.testing.assert_array_almost_equal(kd_harmonic.toarray(), k_harmonic.toarray())
        np.testing.assert_array_almost_equal(kd_arithmetic.toarray(), k_arithmetic.toarray())

    def test_flux_upwind(self):
        # 1D x direction
        grid = StaggeredGrid(size=[3, 1], dimensions=[3, 1])
        q = [-0.5, -0.5, -0.5, -0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_cell_dofs[0] + 1, grid.n_cell_dofs[0]))
        A_answer[0, 0] = -0.5
        A_answer[1, 1] = -0.5
        A_answer[2, 2] = -0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        q = [-0.5, -0.5, 0.5, 0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_cell_dofs[0] + 1, grid.n_cell_dofs[0]))
        A_answer[0, 0] = -0.5
        A_answer[1, 1] = -0.5
        A_answer[2, 1] = 0.5
        A_answer[3, 2] = 0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        # 1D y direction
        grid = StaggeredGrid(size=[1, 3], dimensions=[1, 3])
        q = [-0.5, -0.5, -0.5, -0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_cell_dofs[1] + 1, grid.n_cell_dofs[1]))
        A_answer[0, 0] = -0.5
        A_answer[1, 1] = -0.5
        A_answer[2, 2] = -0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        q = [-0.5, -0.5, 0.5, 0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_cell_dofs[1] + 1, grid.n_cell_dofs[1]))
        A_answer[0, 0] = -0.5
        A_answer[1, 1] = -0.5
        A_answer[2, 1] = 0.5
        A_answer[3, 2] = 0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        # 2D

        # flow in x direction
        grid = StaggeredGrid(size=[2, 2], dimensions=[2, 2])
        q = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0., 0., 0., 0., 0., 0.]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_flux_dofs_total, grid.n_cell_dofs[0] * grid.n_cell_dofs[0]))
        A_answer[0, 0] = -0.5
        A_answer[1, 1] = -0.5
        A_answer[2, 2] = -0.5
        A_answer[3, 3] = -0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        # flow in y direction
        q = [0., 0., 0., 0., 0., 0., -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_flux_dofs_total, grid.n_cell_dofs[0] * grid.n_cell_dofs[1]))
        A_answer[6, 0] = -0.5
        A_answer[7, 1] = -0.5
        A_answer[9, 2] = -0.5
        A_answer[10, 3] = -0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())

        # flow in both directions
        q = [1., 1., 1., 1., 1., 1., -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        A_computed = Operators.flux_upwind(q, grid)
        A_answer = csr_matrix((grid.n_flux_dofs_total, np.prod(grid.n_cell_dofs)))
        A_answer[2, 0] = 1.
        A_answer[3, 1] = 1.
        A_answer[4, 2] = 1.
        A_answer[5, 3] = 1.
        A_answer[6, 0] = -0.5
        A_answer[7, 1] = -0.5
        A_answer[9, 2] = -0.5
        A_answer[10, 3] = -0.5
        np.testing.assert_array_almost_equal(A_computed.toarray(), A_answer.toarray())


if __name__ == '__main__':
    unittest.main()
