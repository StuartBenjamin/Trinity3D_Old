
import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn



### main


# go into the trinity engine
 
## Set initial conditions
N = 10 # number of radial points
n_core = 4
n_edge = 0.2
rho_edge = 0.8    # rho = r/a : normalized radius
rho_axis = np.linspace(0,rho_edge,N) # radial axis
drho = 1/N # temp
# sample profile initial conditions
n = (n_core-n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
#n = n_core*(1 - rho_axis**2) #+ n_edge  # simple expression
#n = n_edge * np.ones(N)   ## constant init

### Set up time controls
alpha = 0          # explicit to implicit mixer
dtau  = 1e-5         # step size 
N_steps = 5000       # total Time = dtau * N_steps
N_prints = 10
N_step_print = N_steps // N_prints   # how often to print # thanks Sarah!
#N_step_print = 100   # how often to print
###


### Set up source
# denisty source
trl.Sn_width = 0.1
trl.Sn_height = 0


# temp fix, pass global param into library
#    this is what should be in the "Trinity Engine"
trl.rho_axis = rho_axis


### will be static > dynamic profile
T  = 2 # constant temp profile 
pressure = trl.profile(n*T)
### will be from VMEC
Ba = 3 # average field on LCFS
R_major = 4 # meter
a_minor = 1 # meter
area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


trl.R_major = R_major
trl.a_minor = a_minor
trl.pressure = pressure
trl.area = area
trl.Ba = Ba
trl.drho = drho
trl.dtau = dtau
trl.alpha = alpha
trl.n_edge = n_edge


### Run Trinity!
_debug = False

density            = trl.init_density(n,debug=_debug)
d2 = dgn.diagnostic_2()

j = 0 
Time = 0
while (j < N_steps):

    Gamma, dlogGamma   = trl.calc_Gamma(density           , debug=_debug)
    F                  = trl.calc_F(density,Gamma         , debug=_debug)
    An_pos, An_neg, Bn = trl.calc_AB_n(density,F,dlogGamma, debug=_debug)
    psi_n_plus, psi_n_minus, psi_n_zero = trl.calc_psi_n(density,F,An_pos,An_neg,Bn, debug=_debug)
    Amat = trl.time_step_LHS(psi_n_plus,psi_n_minus,psi_n_zero)
    bvec = trl.time_step_RHS(density,F,psi_n_plus)
    
    Ainv = np.linalg.inv(Amat) # can also use scipy, or special tridiag method
    n_next = Ainv @ bvec
    if not ( j % N_step_print):
        print('  Plot: t =',j)
        d2.plot(density,Gamma,Time)
    density = trl.update_density(n_next,debug=_debug)
    Time += dtau
    j += 1

tlabel = r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau)
d2.label(title=tlabel)

plt.show()
