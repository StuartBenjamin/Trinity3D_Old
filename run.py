import numpy as np
import matplotlib.pyplot as plt

#import Trinity_io as t_input ## moved to trinity_lib.py

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb
import os, sys

print("Welcome to Trinity3D")
try:
    fin = sys.argv[1]
except:
    fin = 'trinity.in'
print(f"  Loading input file {fin}")


'''
   ToDo: there should be a section where I set default values

   I think it would be best to put this in the TrinityLib itself
'''



### to be deleted, manipulate this in input file instead
gx_path = 'gx-files/run-dir/'
#gx_path = 'gx-files/JET/'
#gx_path = 'gx-files/JET-QA/'
#gx_path = 'gx-files/ITER-run/'
vmec_path = 'gx-geometry/'
#vmec_wout = 'wout_ITER_15MABURN.nc'
#vmec_wout = 'wout_w7x.nc'
#vmec_wout = 'wout_QA_nfp2-46-hires.nc'
#vmec_wout = 'wout_JET.nc'
vmec_wout = '' # defaults to preloaded flux tubes (can extend this to be user supplied flux tubes)

###


### Set up source



### Run Trinity!


#engine = trl.Trinity_Engine(alpha=alpha,
#                            dtau=dtau,
#                            N_steps=N_steps,
#                            N_prints = N_prints,
#                            ###
#                            N        = N_radial,
#                            n_core   = n_core,
#                            n_edge   = n_edge,
#                            Ti_core   = Ti_core,
#                            Ti_edge   = Ti_edge,
#                            Te_core   = Te_core,
#                            Te_edge   = Te_edge,
#                            R_major  = R_major,
#                            a_minor  = a_minor,
#                            Ba       = Ba,
#                            rho_edge = rho_edge,
#                            ###
#                            Sn_height  = Sn_height,  
#                            Spi_height = Spi_height, 
#                            Spe_height = Spe_height,
#                            Sn_width   = Sn_width,   
#                            Spi_width  = Spi_width, 
#                            Spe_width  = Spe_width,  
#                            Sn_center   = Sn_center,   
#                            Spi_center  = Spi_center, 
#                            Spe_center  = Spe_center,  
#                            ext_source_file = ext_source_file,
#                            ###
#                            model      = model,
#                            D_neo      = D_neo,
#                            gx_path    = gx_path,
#                            vmec_path  = vmec_path,
#                            vmec_wout  = vmec_wout
#                            )

engine = trl.Trinity_Engine(fin)

writer = dgn.ProfileSaver()

#fout = 'gx-files/temp.gx'
#model_gx = mf.GX_Flux_Model(fout)

engine.time = 0

## 7/18 can remove these lines?
#density    = engine.density
#pressure_i = engine.pressure_i
#pressure_e = engine.pressure_e

### Set up time controls
N_step_print = engine.N_steps // engine.N_prints   # how often to print 

j = 0 
# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (j < engine.N_steps):

    ### calculates fluxes from GX
    if   (engine.model == "GX"):
        engine.model_gx.prep_commands(engine, j) # use GX  
        #engine.model_gx.prep_commands(engine, j, Time) # use GX ## replaced 7/18
    elif (engine.model == "diffusive"):
        print('Barnes model')
        engine.barnes_model.compute_Q(engine)
    else:
        engine.compute_flux() # use analytic flux model


    engine.normalize_fluxes()
    engine.calc_flux_coefficients()

    # I think collisions and turb. heat exchange should be added here
#    engine.calc_collisions()
    engine.calc_collisions(zero=True)

    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 

    #engine.calc_sources( )
    engine.calc_sources( alpha_heating=False, brems_radiation=False)
    engine.calc_y_next()

    engine.update()

    if not ( j % N_step_print):
        
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

        print('  Plot: t =',j)
        writer.save(engine)

    j += 1


writer.store_system(engine)
writer.export(engine.f_save)

print('TRINITY Complete. Exiting normally')
cmd = 'python tools/profile-plot.py {}.npy'.format(engine.f_save)
print('Calling plot function:')
print('  ',cmd)
os.system(cmd)


