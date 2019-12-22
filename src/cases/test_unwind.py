"""
    This solves the following quasi one-dimensional flow problem with four different boundary conditions
    on the unit square to test if the upwind flux is computed correctly.

    Governing equations:
        -div(K*grad(h)) = 0, h in [0, 1]^2,
        phi*dc/dx + div(q*c) = 0, c in [0, 1]^2,
    where K (permeability) = 1, phi (porosity) = 0.25, q (flux) = -K*grad(h).

    Boundary conditions:
    Case 1:
        h(x=0,y) = 1, h(x=1,y) = 0,
        c(x=0,y) = 1

    Case 2:
        h(x=0,y) = 0, h(x=1,y) = 1,
        c(x=1,y) = 1

    Case 3:
        h(x,y=0) = 1, h(x,y=1) = 0,
        c(x,y=0) = 1

    Case 4:
        h(x,y=0) = 0, h(x,y=1) = 1,
        c(x,y=1) = 1

    No flux boundary condition is assigned everywhere else.
"""

import matplotlib.pyplot as plt
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
    nx = 155
    ny = 155
    tmax = 0.125
    Nt = 30
    dt = tmax / Nt
    vis = Visualization

    grid = StaggeredGrid(size=[length_x, length_y], dimensions=[nx, ny])
    D, G, I = op.build_discrete_operators(grid)
    Kd = 1.
    # Discrete Laplace operator
    L = -D * Kd * G
    fs = np.zeros(grid.n_cell_dofs_total)

    for i in range(4):
        # Set boundary conditions
        if i == 0 or i == 1:
            dirichlet_cell_bc = [grid.idx_cell_dofs_xmin, grid.idx_cell_dofs_xmax]
            dirichlet_flux_bc = [grid.idx_flux_dofs_xmin, grid.idx_flux_dofs_xmax]
            g = np.zeros(len(np.hstack(dirichlet_cell_bc)))
            if i == 0:
                g[:grid.n_cell_dofs[1]] = 1.
            elif i == 1:
                g[grid.n_cell_dofs[1]:] = 1.
        elif i == 2 or i == 3:
            dirichlet_cell_bc = [grid.idx_cell_dofs_ymin, grid.idx_cell_dofs_ymax]
            dirichlet_flux_bc = [grid.idx_flux_dofs_ymin, grid.idx_flux_dofs_ymax]
            g = np.zeros(len(np.hstack(dirichlet_cell_bc)))
            if i == 2:
                g[:grid.n_cell_dofs[0]] = 1.
            elif i == 3:
                g[grid.n_cell_dofs[0]:] = 1.

        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)
        B, N, fn = op.build_boundary_operators(grid, param, I)
        h = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)

        q = op.compute_flux(D, Kd, G, h, fs, grid, param)
        if i == 0:
            dirichlet_cell_bc = [grid.idx_cell_dofs_xmin]
            dirichlet_flux_bc = [grid.idx_flux_dofs_xmin]
        elif i == 1:
            dirichlet_cell_bc = [grid.idx_cell_dofs_xmax]
            dirichlet_flux_bc = [grid.idx_flux_dofs_xmax]
        elif i == 2:
            dirichlet_cell_bc = [grid.idx_cell_dofs_ymin]
            dirichlet_flux_bc = [grid.idx_flux_dofs_ymin]
        elif i == 3:
            dirichlet_cell_bc = [grid.idx_cell_dofs_ymax]
            dirichlet_flux_bc = [grid.idx_flux_dofs_ymax]

        param = Parameters(dir_cell=dirichlet_cell_bc, dir_flux=dirichlet_flux_bc)
        g = np.ones(len(np.hstack(dirichlet_cell_bc)))
        A = op.flux_upwind(q, grid)
        L2 = I + dt / phi * D * A
        [B, N, fn] = op.build_boundary_operators(grid, param, I)
        c = np.zeros(grid.n_cell_dofs_total)

        for _ in range(Nt):
            fs2 = c
            c = solve_linear_boundary_value_problem(L2, fs2 + fn, B, g, N)

        c_reshape = h.reshape(grid.n_cell_dofs).transpose()

        plt.subplot(2, 2, i + 1)
        if i == 0:
            title = "Left to right"
        elif i == 1:
            title = "Right to left"
        elif i == 2:
            title = "Bottom to top"
        elif i == 3:
            title = "Top to bottom"
        vis.imshow(data=c_reshape, title=title)
    plt.show()
