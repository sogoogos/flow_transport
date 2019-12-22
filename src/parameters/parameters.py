from typing import List

import numpy as np


class Parameters:

    def __init__(self, dir_cell: List = None, neu_cell: List = None, dir_flux: List = None, neu_flux: List = None,
                 flux_bc: List = None):
        """
        Store all information about the physical problem including parameters and BCs
        @param dir_cell: indices of cell dofs where Dirichlet bc is assigned
        @param neu_cell: indices of cell dofs where Neumann bc is assigned
        @param dir_flux: indices of flux dofs where Dirichlet bc is assigned
        @param neu_flux: indices of flux dofs where Neumann bc is assigned
        @param flux_bc: flux values at Neumann boundaries
        """
        if dir_cell is None:
            dir_cell = []
        if neu_cell is None:
            neu_cell = []
        if dir_flux is None:
            dir_flux = []
        if neu_flux is None:
            neu_flux = []
        if flux_bc is None:
            flux_bc = []
        self.dof_dirichlet = np.hstack(dir_cell) if len(dir_cell) else np.array([])
        self.dof_neumann = np.hstack(neu_cell) if len(neu_cell) else np.array([])
        self.dof_dirichlet_face = np.hstack(dir_flux) if len(dir_flux) else np.array([])
        self.dof_neumann_face = np.hstack(neu_flux) if len(neu_flux) else np.array([])
        self.qb = np.array(flux_bc)
