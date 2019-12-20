import unittest

import numpy as np
from scipy.sparse import csr_matrix

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators


class TestOperators(unittest.TestCase):
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
        k_harmonic[1, 1] = 1.5
        k_harmonic[4, 4] = 6.428571428571429
        k_harmonic[8, 8] = 1.6666666666666667
        k_harmonic[9, 9] = 4.5
        k_arithmetic = csr_matrix((size, size))
        k_arithmetic[1, 1] = 2.
        k_arithmetic[4, 4] = 7.
        k_arithmetic[8, 8] = 3.
        k_arithmetic[9, 9] = 6
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
