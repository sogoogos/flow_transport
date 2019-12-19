from typing import List

import numpy as np


class StaggeredGrid:
    def __init__(self, size: List, dimensions: List):
        """
        Store all pertinent information about the grid.
         @param size: size of computational domain entered as (dim by 1) float vector
         @param dimensions: dimensions of model entered as (dim by 1) integer vector
        """
        # assume computational domain is 1 or 2D
        self.domain_size = np.array(size)
        self.n_cell_dofs = np.array(dimensions)
        self.n_cell_dofs_total = np.prod(dimensions)

        self.grid_size = self.domain_size / self.n_cell_dofs
        # number of cell face (flux_bc) dofs
        self.n_flux_dofs = (self.n_cell_dofs + 1) * np.roll(self.n_cell_dofs, 1)
        self.n_flux_dofs[self.n_cell_dofs < 2] = 0
        self.n_flux_dofs_total = np.sum(self.n_flux_dofs)

        # nx by 1 column vector of cell center locations
        half_dx = self.grid_size[0] / 2
        half_dy = self.grid_size[1] / 2
        self.loc_cell_x = np.linspace(start=half_dx, stop=size[0] - half_dx, num=dimensions[0])
        self.loc_cell_y = np.linspace(start=half_dy, stop=size[1] - half_dy, num=dimensions[1])

        # nfx by 1 column vector of cell face (flux_bc) locations
        self.loc_flux_x = np.linspace(start=0, stop=size[0], num=dimensions[0] + 1)
        self.loc_flux_y = np.linspace(start=0, stop=size[1], num=dimensions[1] + 1)

        # Assign indices of cell center and face (flux_bc) dofs
        self.idx_dofs = np.arange(self.n_cell_dofs_total)

        # TODO anyways to write this simple way?
        if dimensions[0] > 1:
            self.idx_cell_dofs_xmin = np.arange(start=0, stop=dimensions[1], step=1)
            self.idx_cell_dofs_xmax = np.arange(start=self.n_cell_dofs_total - dimensions[1],
                                                stop=self.n_cell_dofs_total, step=1)
            self.idx_flux_dofs_xmin = np.arange(start=0, stop=dimensions[1], step=1)
            self.idx_flux_dofs_xmax = np.arange(start=self.n_flux_dofs[0] - dimensions[1], stop=self.n_flux_dofs[0],
                                                step=1)
        else:
            self.idx_cell_dofs_xmin = []
            self.idx_cell_dofs_xmax = []
            self.idx_flux_dofs_xmin = []
            self.idx_flux_dofs_xmax = []

        if dimensions[1] > 1:
            self.idx_cell_dofs_ymin = np.arange(start=0, stop=self.n_cell_dofs_total, step=dimensions[1])
            self.idx_cell_dofs_ymax = np.arange(start=dimensions[1] - 1, stop=self.n_cell_dofs_total,
                                                step=dimensions[1])
            self.idx_flux_dofs_ymin = np.arange(start=0, stop=self.n_flux_dofs[1], step=dimensions[1] + 1)
            self.idx_flux_dofs_ymax = np.arange(start=dimensions[1], stop=self.n_flux_dofs[1], step=dimensions[1] + 1)
            self.idx_flux_dofs_ymin += self.n_flux_dofs[0]
            self.idx_flux_dofs_ymax += self.n_flux_dofs[0]
        else:
            self.idx_cell_dofs_ymin = []
            self.idx_cell_dofs_ymax = []
            self.idx_flux_dofs_ymin = []
            self.idx_flux_dofs_ymax = []

        # assume dz is 1
        self.volume = np.full(self.n_cell_dofs_total, np.prod(self.grid_size))
        self.area = np.zeros(self.n_flux_dofs_total)
        self.area[:self.n_flux_dofs[0]] = np.repeat(a=self.grid_size[1], repeats=self.n_flux_dofs[0])
        self.area[self.n_flux_dofs[0]:] = np.repeat(a=self.grid_size[0], repeats=self.n_flux_dofs[1])

    def is_problem_1d(self):
        return self.n_cell_dofs[0] == self.n_cell_dofs_total or self.n_cell_dofs[1] == self.n_cell_dofs_total

    def is_problem_2d(self):
        return self.n_cell_dofs[0] < self.n_cell_dofs_total or self.n_cell_dofs[1] < self.n_cell_dofs_total
