"""
    This case solves the following flow and transport equations in a heterogeneous permeability field, k.

    Flow equation:
        div(-k grad(p) = 0, x in [0, 1],
        p(1) = 0, p(0) = 1.

    Transport equation:
        dc/dt + div(c) = 0, x in [0, 1],
        c(x=0, t) = 1, c(x, t=0) = 0.

    Assume gaussian distribution of log(k).
"""

import numpy as np

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators as op
from parameters.parameters import Parameters
from solvers.solve_lbvp import solve_linear_boundary_value_problem
from visualization.visualization_utils import Visualization

if __name__ == '__main__':
    # Input parameters
    length_x = 1  # m
    length_y = 1
    phi = 0.25
    nx = 150
    ny = 150

    tmax = 0.125
    Nt = 30
    dt = tmax / Nt

    grid = StaggeredGrid(size=[length_x, length_y], dimensions=[nx, ny])

    # Permeability, assume Gaussian distribution of log(k)
    K = np.exp(np.random.normal(loc=-1, scale=1, size=grid.n_cell_dofs_total))
    K = np.reshape(K, grid.n_cell_dofs[::-1])
    Kd = op.compute_mean(K, -1, grid)

    D, G, I = op.build_discrete_operators(grid)

    # Set boundary conditions
    dirichlet_cell_bc = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_xmax]
    dirichlet_flux_bc = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax]
    param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)
    g = np.zeros(len(np.hstack(dirichlet_cell_bc)))
    g[:grid.n_cell_dofs[1]] = 1.

    # Discrete Laplace operator
    L = -D * Kd * G
    fs = np.zeros(grid.n_cell_dofs_total)
    B, N, fn = op.build_boundary_operators(grid, param, I)
    p = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)
    p_reshape = p.reshape(grid.n_cell_dofs).transpose()

    q = op.compute_flux(D, Kd, G, p, fs, grid, param)
    dirichlet_cell_bc = [grid.idx_cell_dofs_xmin]
    dirichlet_flux_bc = [grid.idx_flux_dofs_xmin]
    param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)
    g = np.ones(len(np.hstack(dirichlet_cell_bc)))

    A = op.flux_upwind(q, grid)
    L = dt / phi * D * A + I
    [B, N, fn] = op.build_boundary_operators(grid, param, I)
    c = np.zeros(grid.n_cell_dofs_total)

    for i in range(Nt):
        fs = c
        c = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)

    c_reshape = c.reshape(grid.n_cell_dofs).transpose()

    vis = Visualization
    vis.imshow(data=K, fig_num=1, title="Isotropic permeability field")
    vis.imshow(data=p_reshape, fig_num=2, title="Pressure field")
    vis.imshow(data=c_reshape, fig_num=3, title="Concentration field", show=True)
