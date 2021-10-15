
import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn



### main




N = 20 # number of radial points
n_core = 4
n_edge = 0.2
rho_edge = 0.8
rho_axis = np.linspace(0,rho_edge,N) # radial axis
drho = 1/N # temp
n = (n_core-n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
#n = n_core*(1 - rho_axis**2) #+ n_edge  # simple expression
#n = n_edge * np.ones(N)   ## constant init

### Set up time controls
alpha = 0.3
dtau  = 1e-4
N_steps = 1000
N_step_print = 100
###

trl.rho_axis = rho_axis

T  = 2 # constant temp profile
Ba = 3 # average field on LCFS
R_major = 4 # meter
a_minor = 1 # meter
pressure = trl.profile(n*T)
area     = trl.profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


# temp fix, pass global param into library
#    we should make this a class object instead
trl.R_major = R_major
trl.a_minor = a_minor
trl.pressure = pressure
trl.area = area
trl.Ba = Ba
trl.drho = drho
trl.dtau = dtau
trl.alpha = alpha
trl.n_edge = n_edge



# run calcs
_debug = False
density            = trl.init_density(n,debug=_debug)

plt.figure()

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
        dgn.diagnostic_1(density,Gamma,Time)
    density = trl.update_density(n_next,debug=_debug)
    Time += dtau
    j += 1

plt.title('Gamma(rho)')
plt.legend()
plt.grid()

plt.subplot(1,2,1)
plt.title(r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau))
plt.legend()
plt.ylim(0,4.2)
plt.grid()
plt.show()
