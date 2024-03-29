import unittest

import numpy as np

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators as op
from parameters.parameters import Parameters
from solvers.solve_lbvp import solve_linear_boundary_value_problem
from visualization.visualization_utils import Visualization


class MyTestCase(unittest.TestCase):

    def test_1d_x_dirichlet(self):
        length_x = 20  # m
        nx = 20
        ny = 1
        h_l = 5
        h_r = 10
        K = 1
        q0 = 1
        show_plot = False

        grid = StaggeredGrid(size=[length_x, 1], dimensions=[nx, ny])
        dirichlet_cell_bc = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_xmax]
        dirichlet_flux_bc = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax]

        g = self.analytical_solution_1d_dirichlet([grid.loc_cell_x[0], grid.loc_cell_x[-1]], length_x, K, q0, h_l, h_r)
        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)

        fs = np.full((grid.n_cell_dofs_total,), q0)
        numerical_solution = self.solve(grid, param, K, fs, g)
        analytical_solution = self.analytical_solution_1d_dirichlet(grid.loc_cell_x, length_x, K, q0, h_l, h_r)
        if show_plot:
            Visualization.plot_two_data(x1=grid.loc_cell_x, y1=numerical_solution, x2=grid.loc_cell_x,
                                        y2=analytical_solution, xlabel="Coordinate")
        np.testing.assert_almost_equal(numerical_solution, analytical_solution)

    def test_1d_y_dirichlet(self):
        length_y = 20  # m
        nx = 1
        ny = 20
        h_l = 5
        h_r = 10
        K = 1
        q0 = 1
        show_plot = False

        grid = StaggeredGrid(size=[1, length_y], dimensions=[nx, ny])
        dirichlet_cell_bc = [grid.idx_cell_dofs_ymin, grid.idx_cell_dofs_ymax]
        dirichlet_flux_bc = [grid.idx_flux_dofs_ymin, grid.idx_flux_dofs_ymax]

        g = self.analytical_solution_1d_dirichlet([grid.loc_cell_y[0], grid.loc_cell_y[-1]], length_y, K, q0, h_l, h_r)
        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)

        fs = np.full((grid.n_cell_dofs_total,), q0)
        numerical_solution = self.solve(grid, param, K, fs, g)
        analytical_solution = self.analytical_solution_1d_dirichlet(grid.loc_cell_y, length_y, K, q0, h_l, h_r)
        if show_plot:
            Visualization.plot_two_data(x1=grid.loc_cell_y, y1=numerical_solution, x2=grid.loc_cell_y,
                                        y2=analytical_solution, xlabel="Coordinate")
        np.testing.assert_almost_equal(numerical_solution, analytical_solution)

    def test_1d_x_neumann(self):
        length_x = 20  # m
        nx = 20
        ny = 1
        h_r = 5
        q_l = np.array([1])
        K = 1
        rhs = 1
        show_plot = False

        grid = StaggeredGrid(size=[length_x, 1], dimensions=[nx, ny])

        # set boundary conditions
        dirichlet_cell_bc = grid.idx_cell_dofs_xmax
        dirichlet_flux_bc = grid.idx_flux_dofs_xmax
        neumann_cell_bc = grid.idx_cell_dofs_xmin
        neumann_flux_bc = grid.idx_flux_dofs_xmin

        g = self.analytical_solution_1d_neumann([grid.loc_cell_x[-1]], length_x, K, rhs, h_r, q_l, neumann_left=True)

        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc, neu_cell=neumann_cell_bc,
                           neu_flux=neumann_flux_bc, flux_bc=q_l)

        fs = np.full((grid.n_cell_dofs_total,), rhs)
        numerical_solution = self.solve(grid, param, K, fs, g)
        analytical_solution = self.analytical_solution_1d_neumann(grid.loc_cell_x, length_x, K, rhs, h_r, q_l,
                                                                  neumann_left=True)

        if show_plot:
            Visualization.plot_two_data(x1=grid.loc_cell_x, y1=numerical_solution, x2=grid.loc_cell_x,
                                        y2=analytical_solution, xlabel="x")
        np.testing.assert_almost_equal(numerical_solution, analytical_solution)

    def test_1d_y_neumann(self):
        length_y = 20  # m
        nx = 1
        ny = 20
        h_l = 5
        q_r = np.array([1])
        K = 1
        rhs = 1
        show_plot = False

        grid = StaggeredGrid(size=[1, length_y], dimensions=[nx, ny])

        # set boundary conditions
        dirichlet_cell_bc = grid.idx_cell_dofs_ymin
        dirichlet_flux_bc = grid.idx_flux_dofs_ymin
        neumann_cell_bc = grid.idx_cell_dofs_ymax
        neumann_flux_bc = grid.idx_flux_dofs_ymax

        g = self.analytical_solution_1d_neumann([grid.loc_cell_y[0]], length_y, K, rhs, h_l, q_r, neumann_left=False)

        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc, neu_cell=neumann_cell_bc,
                           neu_flux=neumann_flux_bc, flux_bc=-q_r)  # this is negative since neumann is right side

        fs = np.full((grid.n_cell_dofs_total,), rhs)
        numerical_solution = self.solve(grid, param, K, fs, g)
        analytical_solution = self.analytical_solution_1d_neumann(grid.loc_cell_y, length_y, K, rhs, h_l, q_r,
                                                                  neumann_left=False)

        if show_plot:
            Visualization.plot_two_data(x1=grid.loc_cell_y, y1=numerical_solution, x2=grid.loc_cell_y,
                                        y2=analytical_solution, xlabel="y")
        np.testing.assert_almost_equal(numerical_solution, analytical_solution)

    @staticmethod
    def solve(grid: StaggeredGrid, param: Parameters, K, fs, g):
        D, G, I = op.build_discrete_operators(grid)
        L = -D * K * G
        B, N, fn = op.build_boundary_operators(grid, param, I)

        return solve_linear_boundary_value_problem(L, fs + fn, B, g, N)

    @staticmethod
    def analytical_solution_1d_dirichlet(x, length, K, rhs, h_l, h_r):
        x = np.array(x)
        return -rhs / 2 / K * np.square(x) + ((h_r - h_l) / length + rhs * length / 2 / K) * x + h_l

    @staticmethod
    def analytical_solution_1d_neumann(x, length, K, rhs, h_bc, q_bc, neumann_left=True):
        if neumann_left:
            x_dir = length
            x_neu = 0
        else:
            x_dir = 0
            x_neu = length
        return -rhs / 2 / K * (np.square(x) - x_dir * x_dir) + (rhs * x_neu - q_bc) / K * (np.array(x) - x_dir) + h_bc


if __name__ == '__main__':
    unittest.main()
