
import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn


### main

# go into the trinity engine
 
## Set initial conditions
n_core = 4
n_edge = .5
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
alpha = 1          # explicit to implicit mixer
dtau  = 0.5         # step size 
N_steps  = 1000       # total Time = dtau * N_steps
N_prints = 10
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
###


### Set up source
# denisty source
trl.Sn_width = 0.1
trl.Sn_height = 0

# pressure source. These weren't actually created yet in lib
trl.Spi_width = 0.1
trl.Spi_height = 0
trl.Spe_width = 0.1
trl.Spe_height = 0


# temp fix, pass global param into library
#    this is what should be in the "Trinity Engine"
trl.N_radial_points = N
trl.rho_axis = rho_axis


### will be static > dynamic profile
pressure = trl.profile(n*T0)
temperature = trl.profile(T)
### will be from VMEC
Ba = 3 # average field on LCFS
R_major = 4 # meter
a_minor = 1 # meter
area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


#trl.R_major = R_major
#trl.a_minor = a_minor
#trl.pressure = pressure
#trl.temperature = temperature
#trl.area = area
#trl.Ba = Ba
#trl.drho = drho
#trl.dtau = dtau
#trl.alpha = alpha
#trl.n_edge = n_edge
#trl.pi_edge = pi_edge
#trl.pe_edge = pe_edge


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


#density     = trl.init_profile(n,debug=_debug)
#pressure_i  = trl.init_profile(pi,debug=_debug)
#pressure_e  = trl.init_profile(pe,debug=_debug)

#d1 = dgn.diagnostic_1() # init
#d2 = dgn.diagnostic_2() # init
#d3 = dgn.diagnostic_2() # init

d4_n  = dgn.diagnostic_4()
d4_pi = dgn.diagnostic_4()
d4_pe = dgn.diagnostic_4()

j = 0 
Time = 0
while (j < N_steps):

#    Gamma, dlogGamma, Q_i, dlogQ_i, Q_e, dlogQ_e \
#                            = trl.calc_Flux(density,pressure_i,pressure_e, debug=_debug)
#    Fn, Fpi, Fpe = trl.calc_F3(density,pressure_i,pressure_e,Gamma,Q_i,Q_e, debug=_debug)
#    An_pos, An_neg, Bn, Ai_pos, Ai_neg, Bi, Ae_pos, Ae_neg, Be \
#       = trl.calc_AB(density,pressure_i, pressure_e,Fn,Fpi,Fpe,dlogGamma,dlogQ_i,dlogQ_e,debug=_debug)
#    psi_nn, psi_npi, psi_npe = trl.calc_psi(density, pressure_i, pressure_e, \
#                   Fn,Fpi,Fpe,An_pos,An_neg,Bn,Ai_pos,Ai_neg,Bi, \
#                   Ae_pos,Ae_neg,Be)
    #Amat = trl.time_step_LHS3(psi_nn, psi_npi,psi_npe)
    #bvec = trl.time_step_RHS3(density,pressure_i,pressure_e,Fn,Fpi,Fpe,psi_nn,psi_npi,psi_npe)
    #Ainv = np.linalg.inv(Amat) # can also use scipy, or special tridiag method
    #y_next = Ainv @ bvec

    engine.model_flux()
#    Gamma     = engine.Gamma
#    dlogGamma = engine.dlogGamma   
#    Q_i       = engine.Qi
#    Q_e       = engine.Qe
#    dlogQ_i   = engine.dlogQi
#    dlogQ_e   = engine.dlogQe

    engine.normalize_fluxes()
#   Fn  = engine.Fn
#   Fpi = engine.Fpi
#   Fpe = engine.Fpe

    engine.calc_flux_coefficients()
#   An_pos = engine.Cn_n.plus 
#   An_neg = engine.Cn_n.minus 
#   Bn     = engine.Cn_n.zero 
#   Ai_pos = engine.Cn_pi.plus 
#   Ai_neg = engine.Cn_pi.minus 
#   Bi     = engine.Cn_pi.zero   
#   Ae_pos = engine.Cn_pe.plus 
#   Ae_neg = engine.Cn_pe.minus 
#   Be     = engine.Cn_pe.zero 
#
    engine.calc_psi_n()
#   psi_nn  = engine.psi_nn.matrix
#   psi_npi = engine.psi_npi.matrix
#   psi_npe = engine.psi_npe.matrix

    engine.calc_y_next()
#    y_next = engine.y_next

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
