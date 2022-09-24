import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb
import os, sys

print("\nWelcome to Trinity3D")
try:
    fin = sys.argv[1]
except:
    fin = 'trinity.in'
print("\n  Loading input file:", fin, "\n")


'''
   ToDo: there should be a section where I set default values

   I think it would be best to put this in the TrinityLib itself
'''


### Run Trinity!

engine = trl.Trinity_Engine(fin)
writer = dgn.ProfileSaver()

### Set up time controls
N_step_print = engine.N_steps // engine.N_prints   # how often to print 

# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (engine.t_idx < engine.N_steps):
#while (engine.gx_idx < engine.N_steps):

    engine.get_flux() # Get fluxes from GX or ReLu model. Electron scale Moose [x]
    engine.normalize_fluxes() # Electron scale Moose [x]
    engine.calc_flux_coefficients() # Electron scale Moose [x]

    # I think collisions and turb. heat exchange should be added here
    engine.calc_collisions()

    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 

    engine.calc_sources()
    engine.calc_y_next()

    engine.update()

    if not ( engine.t_idx % N_step_print):
        
        # load
        density    = engine.density
        pressure_i = engine.pressure_i
        pressure_e = engine.pressure_e
        Fn         = engine.Fn
        Fpi        = engine.Fpi
        Fpe        = engine.Fpe
        Gamma      = engine.Gamma_total
        Q_i        = engine.Qi_total
        Q_e        = engine.Qe_total

        print(f"  Plot: t = {engine.t_idx}")
        #print(f"  Plot: t = {engine.t_idx}, t_gx = {engine.gx_idx}") # unnecessary
        writer.save(engine)

    engine.reset_fluxtubes()

writer.store_system(engine)
writer.export(engine.f_save)

print('TRINITY Complete. Exiting normally')

path = "~tqian/CODE/Trinity3D/"
cmd = f"python {path}tools/profile-plot.py {engine.f_save}.npy"
print('Calling plot function:')
print('  ',cmd)
os.system(cmd)


