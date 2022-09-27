import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb
import os, sys

print("\nWelcome to Trinity3D")

# check that path is set
if os.environ.get("TRINITY_PATH") == None:
    print("\n  Environment Variable $TRINITY_PATH does not appear to be set.")
    print("  Try running (source setup.sh)\n")
    sys.exit()


try:
    fin = sys.argv[1]
except:
    fin = 'trinity.in'
print("\n  Loading input file:", fin, "\n")


### Run Trinity!

engine = trl.Trinity_Engine(fin)

writer = dgn.ProfileSaver()


### Set up time controls
N_step_print = engine.N_steps // engine.N_prints   # how often to print 


# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (engine.t_idx < engine.N_steps):
#while (engine.gx_idx < engine.N_steps):
    '''
    shift from counting time, to counting gx_calls?

    decison: not doing this later, it will change the nature of tests
    and different run modes (electron scale) might use more GX calls than others

    For now, I will just have the Newton method NOT increment this while loop (turtle 9/27)
    '''


    engine.get_flux()
    engine.normalize_fluxes()
    engine.calc_flux_coefficients()

    # I think collisions and turb. heat exchange should be added here
    engine.calc_collisions()

    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 

    engine.calc_sources()

    if not engine.newton_mode:
       engine.calc_y_next()
    else:
       engine.calc_y_iter()

    engine.update()

    if not ( engine.t_idx % N_step_print):
        
        # load
        density    = engine.density
        pressure_i = engine.pressure_i
        pressure_e = engine.pressure_e
        Fn         = engine.Fn
        Fpi        = engine.Fpi
        Fpe        = engine.Fpe
        Gamma      = engine.Gamma
        Q_i        = engine.Qi
        Q_e        = engine.Qe

        print(f"  Saving: t = {engine.t_idx}")
        writer.save(engine)

    engine.reset_fluxtubes()

writer.store_system(engine)
writer.export(engine.f_save)

print('\nTRINITY Complete. Exiting normally')

root = os.environ.get("TRINITY_PATH") 
cmd = f"python {root}/tools/profile-plot.py {engine.f_save}.npy"
print('Calling plot function:')
print('  ',cmd)
os.system(cmd)


