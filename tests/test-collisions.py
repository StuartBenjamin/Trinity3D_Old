import numpy as np
import matplotlib.pyplot as plt


import Collisions as clib


svec = clib.Species()

# Dummy profiles
N_radial = 11
radial_grid = np.linspace(0,1,N_radial)
#n_profile_m3  = (1 - radial_grid**2) * 1e20 + 1e18
n_profile_20  = np.ones(N_radial) 
edge_temp_keV = 0.1
T_profile_keV = (1 - radial_grid**2) * 15 + edge_temp_keV
#T_profile_keV = np.ones(N_radial) * 15 ## constant for debugging
T_profile_eV = T_profile_keV*1e3

svec.add_species( n_profile_20, T_profile_keV, mass=2, charge=1, ion=True, name='Deuterium')
svec.add_species( n_profile_20, T_profile_keV, mass=1/1800, charge=-1, ion=False, name='electrons')

svec.compute_collision_matrix()

import pdb
pdb.set_trace()
