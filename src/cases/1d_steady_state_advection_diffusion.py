"""
    This case numerically solves the following boundary value problems for the one-dimensional steady
    state advection-diffusion equations and compare the result with its analytical solutions.

    d/dx (Pe c - dc/dx) = 0, x in [0, 1]
    c(0) = 0, c(1) = 1.

    where Pe is Peclet number.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators as op
from parameters.parameters import Parameters
from solvers.solve_lbvp import solve_linear_boundary_value_problem


def analytical_solution_steady(x, pe):
    sol = (np.exp(pe * x) - 1.) / (np.exp(pe) - 1.) if pe != 0. else x
    sol[np.exp(pe * x) == float("inf")] = 0.
    return sol


if __name__ == '__main__':
    # Peclet number
    pe = 10
    
    # define grids and corresponding operators
    grid = StaggeredGrid(size=[1, 1], dimensions=[100, 1])
    D, G, I = op.build_discrete_operators(grid)

    # Steady state problem
    # Set boundary conditions
    dirichlet_cell_bc = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_xmax]
    dirichlet_flux_bc = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax]
    param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)
    g = analytical_solution_steady(np.array([grid.loc_cell_x[0], grid.loc_cell_x[-1]]), pe)

    q = np.ones(grid.n_flux_dofs_total)
    A = op.flux_upwind(q, grid)
    L = D * (pe * A - G)

    fs = np.zeros(grid.n_cell_dofs_total)
    B, N, fn = op.build_boundary_operators(grid, param, I)
    num_sol_steady = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)
    ana_sol_steady = analytical_solution_steady(grid.loc_cell_x, pe)

    plt.plot(grid.loc_cell_x, num_sol_steady, '.', label="Numerical")
    plt.plot(grid.loc_cell_x, ana_sol_steady, label="Analytical")
    plt.title("Pe = " + str(pe))
    plt.legend()
    plt.show()
