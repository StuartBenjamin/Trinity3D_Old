import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb

# set up grid
N = 5 # number of radial points (N-1 flux tubes)
rho_edge = 0.85    # rho = r/a : normalized radius
rho_axis = np.linspace(0,rho_edge,N) # radial axis

#model = 'diffusive'   # Barnes test 2
model = 'GX'          # use slurm to call GX
#model = 'ReLU'        # default

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

### Set up time controls
alpha = 1          # explicit to implicit mixer
dtau  = 2         # step size 
N_steps  = 5       # total Time = dtau * N_steps
N_prints = 5 
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
###

## Set initial conditions
n_core  = 3
n_edge  = 3

pi_core = 7
pi_edge = 3

pe_core = 7
pe_edge = 3 

### Set up source
Sn_height  = 0
Spi_height = 0
Spe_height = 0
Sn_width   = 0.2
Spi_width  = 0.2
Spe_width  = 0.2
Sn_center   = 0.3
Spi_center  = 0.5
Spe_center  = 0.3


### will be from VMEC
Ba = 4 # average field on LCFS
R_major = 2.94   # meter
a_minor = 0.94 # meter
#area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


### Run Trinity!
_debug = False # this knob is being phased out

engine = trl.Trinity_Engine(alpha=alpha,
                            dtau=dtau,
                            N_steps=N_steps,
                            N_prints = N_prints,
                            ###
                            N        = N,
                            n_core   = n_core,
                            n_edge   = n_edge,
                            pi_core   = pi_core,
                            pi_edge   = pi_edge,
                            pe_core   = pe_core,
                            pe_edge   = pe_edge,
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


d3_prof  = dgn.diagnostic_3()
d3_flux  = dgn.diagnostic_3()

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
    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 
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
        d3_prof.plot( density, pressure_i, pressure_e, Time)
        d3_flux.plot( Gamma, Q_i, Q_e, Time)


        ### write GX commands
        # later this could be on a separate time scale
        #engine.model_gx.prep_commands(engine, j, Time)

        # is it better for model_gx to live in run scope or in engine?
        writer.save(engine)


    Time += dtau
    j += 1

rlabel = r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau)

d3_prof.label(titles=['n','pi','pe'])
d3_prof.title(rlabel)

d3_flux.label(titles=['Gamma','Qi','Qe'])

#engine.plot_sources()

#path = './' # should get path from trinity engine's GX_IO, and if GX is not used?
#fout = 'trinity_log.npy'
fout = 'log_trinity.npy'

writer.store_system(engine)
writer.export(fout)

print('TRINITY Complete. Exiting normally')
plt.show()

