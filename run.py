import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn


### main

# go into the trinity engine
 
## Set initial conditions
n_core = 5
n_edge = 1.5
pi_core = 8
pi_edge = 2
pe_core = 3
pe_edge = .3

# set up grid
N = 10 # number of radial points
rho_edge = 0.8    # rho = r/a : normalized radius
rho_axis = np.linspace(0,rho_edge,N) # radial axis
drho = 1/N # temp
# sample profile initial conditions
n  = (n_core - n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
T0  = 2 # constant temp profile, could be retired
pi = T0*n
pe = T0*n
#pi = (pi_core-pi_edge)*(1 - (rho_axis/rho_edge)**2) + pi_edge
#pe = (pe_core-pe_edge)*(1 - (rho_axis/rho_edge)**2) + pe_edge
T = T0 * np.ones(N)

### Set up time controls
alpha = 0          # explicit to implicit mixer
dtau  = 0.05         # step size 
N_steps  = 500       # total Time = dtau * N_steps
N_prints = 10
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
###


### Set up source
trl.Sn_width = 0.1
trl.Sn_height = 0
trl.Spi_width = 0.1
trl.Spi_height = 0
trl.Spe_width = 0.1
trl.Spe_height = 0


# temp fix, pass global param into library
#    this is what should be in the "Trinity Engine"
#trl.N_radial_points = N
trl.rho_axis = rho_axis


### will be static > dynamic profile
pressure = trl.profile(n*T0)
temperature = trl.profile(T)
### will be from VMEC
Ba = 3 # average field on LCFS
R_major = 4 # meter
a_minor = 1 # meter
area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus



### Run Trinity!
_debug = False # this knob is being phased out

engine = trl.Trinity_Engine(alpha=alpha,
                            dtau=dtau,
                            N_steps=N_steps,
                            N_prints = N_prints,
                            ###
                            n_core   = n_core,
                            n_edge   = n_edge,
                            pi_core   = pi_core,
                            pi_edge   = pi_edge,
                            pe_core   = pe_core,
                            pe_edge   = pe_edge,
                            T0       = T0,
                            R_major  = R_major,
                            a_minor  = a_minor,
                            Ba       = Ba,
                            rho_edge = rho_edge)


d4_n  = dgn.diagnostic_4()
d4_pi = dgn.diagnostic_4()
d4_pe = dgn.diagnostic_4()

j = 0 
Time = 0
# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (j < N_steps):

#    engine.model_flux()
    engine.compute_flux()
    engine.normalize_fluxes()
    engine.calc_flux_coefficients()
    engine.calc_psi_n()
    engine.calc_psi_pi() # new
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
        d4_n.plot(  density, Gamma, Fn, Fn.grad, Time )
        d4_pi.plot( pressure_i, Q_i, Fpi, Fpi.grad, Time )
        d4_pe.plot( pressure_e, Q_e, Fpe, Fpe.grad, Time )


    Time += dtau
    j += 1

rlabel = r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau)
d4_n.label(titles=['density', 'Gamma', 'F', 'grad F'])
d4_pi.label()
d4_n.title(rlabel)
d4_pi.title('Pi')
d4_pe.title('Pe')
d4_pe.legend()

plt.show()
