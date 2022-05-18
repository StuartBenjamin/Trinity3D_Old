import numpy as np
import matplotlib.pyplot as plt

import Trinity_io as t_input

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb
import os, sys

try:
    fin = sys.argv[1]
except:
    fin = 'trinity.in'
tr3d = t_input.Trinity_Input(fin)

### read inputs

N_radial = int   ( tr3d.inputs['grid']['N_radial'] )
rho_edge = float ( tr3d.inputs['grid']['rho_edge'] )
dtau     = float ( tr3d.inputs['grid']['dtau'    ] )
alpha    = float ( tr3d.inputs['grid']['alpha'   ] )
N_steps  = int   ( tr3d.inputs['grid']['N_steps' ] )


model    = tr3d.inputs['model']['model']


n_core  = float ( tr3d.inputs['profiles']['n_core' ] )
n_edge  = float ( tr3d.inputs['profiles']['n_edge' ] )
Ti_core = float ( tr3d.inputs['profiles']['Ti_core'] )
Ti_edge = float ( tr3d.inputs['profiles']['Ti_edge'] )
Te_core = float ( tr3d.inputs['profiles']['Te_core'] )
Te_edge = float ( tr3d.inputs['profiles']['Te_edge'] )

Sn_height  = float ( tr3d.inputs['sources']['Sn_height' ] ) 
Spi_height = float ( tr3d.inputs['sources']['Spi_height'] ) 
Spe_height = float ( tr3d.inputs['sources']['Spe_height'] ) 
Sn_width   = float ( tr3d.inputs['sources']['Sn_width'  ] ) 
Spi_width  = float ( tr3d.inputs['sources']['Spi_width' ] ) 
Spe_width  = float ( tr3d.inputs['sources']['Spe_width' ] ) 
Sn_center  = float ( tr3d.inputs['sources']['Sn_center' ] ) 
Spi_center = float ( tr3d.inputs['sources']['Spi_center'] ) 
Spe_center = float ( tr3d.inputs['sources']['Spe_center'] ) 

R_major   = float ( tr3d.inputs['geometry']['R_major'] ) 
a_minor   = float ( tr3d.inputs['geometry']['a_minor'] ) 
Ba        = float ( tr3d.inputs['geometry']['Ba'     ] ) 


N_prints = int ( tr3d.inputs['log']['N_prints'] )
f_save   = tr3d.inputs['log']['f_save']

'''
   ToDo: there should be a section where I set default values

   I think it would be best to put this in the TrinityLib itself
'''
#vmec_wout = float ( tr3d.inputs['geometry']['Ba'     ] ) 

####

# set up grid
rho_axis = np.linspace(0,rho_edge,N_radial) # radial axis

### Set up time controls
N_step_print = N_steps // N_prints   # how often to print 



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
_debug = False # this knob is being phased out

engine = trl.Trinity_Engine(alpha=alpha,
                            dtau=dtau,
                            N_steps=N_steps,
                            N_prints = N_prints,
                            ###
                            N        = N_radial,
                            n_core   = n_core,
                            n_edge   = n_edge,
                            Ti_core   = Ti_core,
                            Ti_edge   = Ti_edge,
                            Te_core   = Te_core,
                            Te_edge   = Te_edge,
                            R_major  = R_major,
                            a_minor  = a_minor,
                            Ba       = Ba,
                            rho_edge = rho_edge,
                            ###
                            Sn_height  = Sn_height,  
                            Spi_height = Spi_height, 
                            Spe_height = Spe_height,
                            Sn_width   = Sn_width,   
                            Spi_width  = Spi_width, 
                            Spe_width  = Spe_width,  
                            Sn_center   = Sn_center,   
                            Spi_center  = Spi_center, 
                            Spe_center  = Spe_center,  
                            ###
                            model      = model,
                            gx_path    = gx_path,
                            vmec_path  = vmec_path,
                            vmec_wout  = vmec_wout
                            )



writer = dgn.ProfileSaver()

#fout = 'gx-files/temp.gx'
#model_gx = mf.GX_Flux_Model(fout)

Time = 0
density    = engine.density
pressure_i = engine.pressure_i
pressure_e = engine.pressure_e
#d3_prof.plot( density, pressure_i, pressure_e, Time)

j = 0 
# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (j < N_steps):

    if   (engine.model == 'GX'):
        engine.model_gx.prep_commands(engine, j, Time) # use GX
    elif (engine.model == 'diffusive'):
        engine.barnes_model.compute_Q(engine)
    else:
        engine.compute_flux() # use analytic flux model


    engine.normalize_fluxes()
    engine.calc_flux_coefficients()

    # I think collisions and turb. heat exchange should be added here
    engine.calc_collisions()

    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 

    engine.calc_sources( )
    #engine.calc_sources( alpha_heating=False, brems_radiation=False)
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

    Time += dtau
    j += 1


#path = './' # should get path from trinity engine's GX_IO, and if GX is not used?

writer.store_system(engine)
import pdb
pdb.set_trace()
writer.export(f_save)

print('TRINITY Complete. Exiting normally')
cmd = 'python tools/profile-plot.py {}.npy'.format(f_save)
print('Calling plot function:')
print('  ',cmd)
os.system(cmd)


