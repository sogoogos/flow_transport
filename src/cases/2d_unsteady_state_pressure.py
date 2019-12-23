"""
    This case solves the parabolic equation in a two-dimensional domain with zero flux boundary conditions

    ct*dp/dt + div(-K grad(p)) = q, x in [0, 1]^2
    IC: p(t=0) = 100,
    BC: zero flux
"""

import numpy as np

from grids.staggered_grid import StaggeredGrid
from operators.discrete_operators import Operators as op
from parameters.parameters import Parameters
from solvers.solve_lbvp import solve_linear_boundary_value_problem
from visualization.visualization_utils import Visualization

if __name__ == '__main__':
    # define grids and corresponding operators
    grid = StaggeredGrid(size=[1, 1], dimensions=[10, 10])
    D, G, I = op.build_discrete_operators(grid)
    Kd = 1.
    nt = 1
    ct = 1e-6

    # Initial condition
    p = np.full((grid.n_cell_dofs_total,), 100)
    # Set boundary conditions
    param = Parameters()
    g = np.array([])

    q = np.zeros(grid.n_cell_dofs_total)
    q[0] = 1
    q[-1] = -1
    dt = 0.1

    L = I + dt / ct * D * Kd * G

    # fs = np.zeros(grid.n_cell_dofs_total)
    B, N, fn = op.build_boundary_operators(grid, param, I)

    for i in range(nt):
        fs = q * dt / ct + p
        p = solve_linear_boundary_value_problem(L, fs + fn, B, g, N)
        # p_reshape = p.reshape(grid.n_cell_dofs)
        # Visualization.imshow(data=p_reshape, title="Pressure field", show=True)

    p_reshape = p.reshape(grid.n_cell_dofs)
    Visualization.imshow(data=p_reshape, title="Pressure field", show=True)
