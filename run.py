
import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn



### main


# go into the trinity engine
 
## Set initial conditions
n_core = 4
n_edge = .5 
pi_core = 4
pi_edge = .5
pe_core = 4
pe_edge = .5
# set up grid
N = 10 # number of radial points
rho_edge = 0.8    # rho = r/a : normalized radius
rho_axis = np.linspace(0,rho_edge,N) # radial axis
drho = 1/N # temp
# sample profile initial conditions
n  = (n_core - n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
pi = (pi_core-pi_edge)*(1 - (rho_axis/rho_edge)**2) + pi_edge
pe = (pe_core-pe_edge)*(1 - (rho_axis/rho_edge)**2) + pe_edge
T0  = 2 # constant temp profile, could be retired
T = T0 * np.ones(N)

### Set up time controls
alpha = 1          # explicit to implicit mixer
dtau  = 1e5         # step size 
N_steps  = 1000       # total Time = dtau * N_steps
#alpha = 0          # explicit to implicit mixer
#dtau  = 1e-3         # step size 
#N_steps  = 10000       # total Time = dtau * N_steps
N_prints = 10
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
#N_step_print = 100   # how often to print
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


trl.R_major = R_major
trl.a_minor = a_minor
trl.pressure = pressure
trl.temperature = temperature
trl.area = area
trl.Ba = Ba
trl.drho = drho
trl.dtau = dtau
trl.alpha = alpha
trl.n_edge = n_edge
trl.pi_edge = pi_edge
trl.pe_edge = pe_edge


### Run Trinity!
_debug = False

density            = trl.init_density(n,debug=_debug)
pressure_i  = trl.init_pressure_i(pi,debug=_debug)
pressure_e  = trl.init_pressure_e(pe,debug=_debug)

d1 = dgn.diagnostic_1() # init
d2 = dgn.diagnostic_2() # init

j = 0 
Time = 0
while (j < N_steps):

    # new functions from Sarah
    Gamma, dlogGamma, Q_i, dlogQ_i, Q_e, dlogQ_e \
                             = trl.calc_Flux(density,pressure_i,pressure_e, debug=_debug)
    Fn, Fpi, Fpe = trl.calc_F3(density,pressure_i,pressure_e,Gamma,Q_i,Q_e, debug=_debug)
    An_pos, An_neg, Bn, Ai_pos, Ai_neg, Bi, Ae_pos, Ae_neg, Be \
       = trl.calc_AB(density,pressure_i, pressure_e,Fn,Fpi,Fpe,dlogGamma,dlogQ_i,dlogQ_e,debug=_debug)
    #psi_nn, psi_npi, psi_npe = trl.calc_psi(psi_n_plus,psi_n_minus,psi_n_zero,psi_pi_plus,\
    psi_nn, psi_npi, psi_npe = trl.calc_psi(density, pressure_i, pressure_e,Fn,Fpi,Fpe,An_pos,An_neg,Bn,Ai_pos,Ai_neg,Bi, \
                   Ae_pos,Ae_neg,Be)


#    Gamma, dlogGamma   = trl.calc_Gamma(density           , debug=_debug)
#    F                  = trl.calc_F(density,Gamma         , debug=_debug)
#    An_pos, An_neg, Bn = trl.calc_AB_gen(density,F,dlogGamma, debug=_debug)
#    psi_nn = trl.calc_psi_nn(density,F,An_pos,An_neg,Bn)
#    Amat   = trl.time_step_LHS3( psi_nn )
         # eventually this will take 9 input matrices
         # it would be even better to keep all self contained in a "Trinity-Engine"
#    bvec = trl.time_step_RHS3(density,F,psi_nn)

    Amat = trl.time_step_LHS3(psi_nn, psi_npi,psi_npe)
    bvec = trl.time_step_RHS3(density,pressure_i,pressure_e,Fn,Fpi,Fpe,psi_nn,psi_npi,psi_npe)


    Ainv = np.linalg.inv(Amat) # can also use scipy, or special tridiag method
    y_next = Ainv @ bvec
    n_next, Ti_next, Te_next = np.reshape(y_next,(3,N-1) )
    if not ( j % N_step_print):
        print('  Plot: t =',j)
        d1.plot(density,Gamma,Time)
        d2.plot(Fn.grad, Fn, Time)
        #d2.plot(F.grad, F, Time)
    density = trl.update_density(n_next,debug=_debug)
    Time += dtau
    j += 1

rlabel = r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau)
d1.label(title=rlabel)
d2.label(t0='grad F',t1='F')

plt.show()
