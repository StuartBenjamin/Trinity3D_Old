import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import pdb

### main

# go into the trinity engine
 
## Set initial conditions
n_core  = 5
n_edge  = 3

pi_core = 5 
pi_edge = 2

pe_core = 5
pe_edge = 2 

# set up grid
N = 7 # number of radial points
rho_edge = 0.8    # rho = r/a : normalized radius
rho_axis = np.linspace(0,rho_edge,N) # radial axis
#drho = 1/N # temp

### Set up time controls
alpha = 1          # explicit to implicit mixer
dtau  = 1         # step size 
N_steps  = 10       # total Time = dtau * N_steps
N_prints = 10
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
###


### Set up source
Sn_height  = 3
Spi_height = 3
Spe_height = 2
Sn_width   = 0.2
Spi_width  = 0.2
Spe_width  = 0.3


# temp fix, pass global param into library
#    this is what should be in the "Trinity Engine"
trl.rho_axis = rho_axis


### will be static > dynamic profile
#pressure = trl.profile(n*T0)
#temperature = trl.profile(T)
### will be from VMEC
Ba = 3 # average field on LCFS
R_major = 5   # meter
a_minor = 0.5 # meter
area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


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
#                            T0       = T0,
                            R_major  = R_major,
                            a_minor  = a_minor,
                            Ba       = Ba,
                            rho_edge = rho_edge,
                            ###
                            Sn_width   = Sn_width,   
                            Sn_height  = Sn_height,  
                            Spi_width  = Spi_width, 
                            Spi_height = Spi_height, 
                            Spe_width  = Spe_width,  
                            Spe_height = Spe_height 
                            )


d3_prof  = dgn.diagnostic_3()
d3_flux  = dgn.diagnostic_3()

#fout = 'gx-files/temp.gx'
#model_gx = mf.GX_Flux_Model(fout)

Time = 0
density    = engine.density
pressure_i = engine.pressure_i
pressure_e = engine.pressure_e
#Gamma     = engine.Gamma
#Q_i       = engine.Qi
#Q_e       = engine.Qe
d3_prof.plot( density, pressure_i, pressure_e, Time)
#d3_flux.plot( Gamma, Q_i, Q_e, Time)
j = 0 
# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (j < N_steps):

    engine.compute_flux()
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
        Fn  = engine.Fn
        Fpi = engine.Fpi
        Fpe = engine.Fpe
        Gamma     = engine.Gamma
        Q_i       = engine.Qi
        Q_e       = engine.Qe

        print('  Plot: t =',j)
        d3_prof.plot( density, pressure_i, pressure_e, Time)
        d3_flux.plot( Gamma, Q_i, Q_e, Time)


        ### write GX commands
        # later this could be on a separate time scale
        engine.model_gx.prep_commands(engine, j, Time)
        #model_gx.prep_commands(engine, j, Time)

        # is it better for model_gx to live in run scope or in engine?



    Time += dtau
    j += 1

rlabel = r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau)

d3_prof.label(titles=['n','pi','pe'])
d3_prof.title(rlabel)

d3_flux.label(titles=['Gamma','Qi','Qe'])

engine.plot_sources()

#plt.show()
