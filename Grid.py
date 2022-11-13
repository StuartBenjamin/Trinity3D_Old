import numpy as np

class Grid():

    def __init__(self, inputs):

        # read grid parameters from input file
        grid_parameters = inputs.get('grid', {})
        self.N_radial = grid_parameters.get('N_radial', 10)
        self.rho_edge = grid_parameters.get('rho_edge', 0.8)

        # compute additional parameters
        self.rho_inner = self.rho_edge / (2*self.N_radial - 1)
        self.rho_axis = np.linspace(self.rho_inner, self.rho_edge, self.N_radial) # radial axis, N points
        self.mid_axis = (self.rho_axis[1:] + self.rho_axis[:-1])/2  # midpoints, (N-1) points

        # TODO: consider case where this is non-constant
        self.drho  = (self.rho_edge - self.rho_inner) / (self.N_radial - 1) 
