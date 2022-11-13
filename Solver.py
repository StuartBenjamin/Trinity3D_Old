import numpy as np

class TransportSolver():

    def __init__(self, grid, time, species):

        # number of evolved profiles
        self.N_profiles = species.N_profiles

        # initialize an N_prof x N_prof identity matrix
        self.I_mat = np.identity(self.N_profiles*grid.N_radial)

        y_init = species.nT_vec
        self.y_hist = []
        self.y_hist.append(y_init)
        self.y_error = np.zeros_like(y_init)
        self.chi_error = 0
