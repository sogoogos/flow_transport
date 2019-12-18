import matplotlib.pyplot as plt
import numpy as np

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators as op
from parameters.parameters import Parameters
from solvers.solve_lbvp import solve_linear_boundary_value_problem

if __name__ == '__main__':
    # Input parameters
    length_x = 3  # m
    length_y = 4
    dh = 15  # m
    phi = 0.25
    nx = 30
    ny = 30
    tmax = 0.125
    nt = 30
    dt = tmax / nt

    grid = StaggeredGrid(size=[length_x, length_y], dimensions=[nx, ny])

    # Permeability, Assume Gaussian distribution of log(K)
    K = np.exp(np.random.normal(loc=1, scale=0.1, size=grid.n_cell_dofs_total))
    K = np.reshape(K, grid.n_cell_dofs[::-1])
    Kd = op.compute_mean(K, -1, grid)

    D, G, I = op.build_discrete_operators(grid)

    # apply boundary conditions
    dirichlet_cell_bc = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_xmax]
    dirichlet_flux_bc = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax]
    neumann_cell_bc = [grid.idx_cell_dofs_ymin, grid.idx_cell_dofs_ymax]
    neumann_flux_bc = [grid.idx_flux_dofs_ymin, grid.idx_flux_dofs_ymax]
    flux_bc = np.zeros(len(np.hstack(neumann_cell_bc)))
    param = Parameters(dirichlet_cell_bc, neumann_cell_bc, dirichlet_flux_bc, neumann_flux_bc, flux_bc)

    g = np.zeros(len(np.hstack(dirichlet_cell_bc)))
    g[:grid.n_cell_dofs[1]] = 1.

    # Discrete Laplace operator
    # length = -D * Kd * G
    L = -D * G
    fs = np.zeros(grid.n_cell_dofs_total)

    B, N, fn = op.build_boundary_operators(grid, param, I)

    solution = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)
    # numerical_solution = numerical_solution.reshape(grid.n_cell_dofs)[::-1]
    solution = solution.reshape(grid.n_cell_dofs).transpose()

    x, y = np.meshgrid(grid.loc_cell_x, grid.loc_cell_y)

    plt.imshow(solution, origin='lower')
    plt.colorbar()
    plt.show()
