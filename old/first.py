# need profiles for n, F

# given Gamma (calculated from turbulence), compute F
# given (n,F) compute next time step

# need function for half stepping an array, how to handle boundary?
# for each n-profile, there is an n- and n+, same for F

# simple model for Gamma is gradient length scale + slope


import numpy as np
import matplotlib.pyplot as plt


N = 20 # number of radial points
n_core = 4
n_edge = .2
rho_edge = 0.8
rho_axis = np.linspace(0,rho_edge,N) # radial axis
drho = 1/N # temp
n = (n_core-n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
#n = n_core*(1 - rho_axis**2) #+ n_edge  # simple expression
#n = n_edge * np.ones(N)   ## constant init

### Set up time controls
alpha = 1
dtau  = 1e-3
N_steps = 1000
N_step_print = 100
###

# a general class for handling profiles (n, p, F, gamma, Q, etc)
# with options to evaluate half steps and gradients at init
class profile():
    def __init__(self,arr, grad=False, half=False, full=False):

        # take a 1D array to be density, for example
        self.profile = np.array(arr) 
        self.length  = len(arr)
        #self.axis    = np.linspace(0,1,self.length)
        global rho_axis
        self.axis    = rho_axis
        # assumes fixed radial griding, which (if irregular) could also be a profile, defined as a function of index

        if (grad):
            self.grad     =  profile(self.gradient())
            self.grad_log =  profile(self.log_gradient())

        if (half):
            self.plus  = profile(self.halfstep_pos())
            self.minus = profile(self.halfstep_neg())

        if (full):
            self.plus1  = profile(self.fullstep_pos())
            self.minus1 = profile(self.fullstep_neg())

    def halfstep_neg(self):
        # x_j\pm 1/2 = (x_j + x_j \pm 1) / 2
        xj = self.profile
        x1 = np.roll(xj,1)
        x1[0] = xj[0]
        return (xj + x1) / 2

    def halfstep_pos(self):
        # x_j\pm 1/2 = (x_j + x_j \pm 1) / 2
        xj = self.profile
        x1 = np.roll(xj,-1)
        x1[-1] = xj[-1]
        return (xj + x1) / 2

    def fullstep_pos(self):
        x0 = self.profile
        x1 = np.roll(x0,-1)
        x1[-1] = x0[-1]
        return x1

    def fullstep_neg(self):
        x0 = self.profile
        x1 = np.roll(x0,1)
        x1[0] = x0[0]
        return x1

    def gradient(self):
        # assume equal spacing
        # 3 point - first deriv: u_j+1 - 2u + u_j-1
        xj = self.profile
        xp = np.roll(xj,-1)
        xm = np.roll(xj, 1)


        dx = 1/len(xj) # assumes spacing is from (0,1)
        deriv = (xp - xm) / (2*dx)
        deriv[0]  = deriv[1]
        deriv[-1] = deriv[-2]

        return deriv
        # can recursively make gradient also a profile class
        # need to test this

    def log_gradient(self):
        # this is actually the gradient of the log...
        eps = 1e-8
        return self.gradient() / (self.profile + eps)

    def plot(self,show=False,new_fig=False,label=''):

        if (new_fig):
            plt.figure(figsize=(4,4))

        #ax = np.linspace(0,1,self.length)
        #plt.plot(ax,self.profile,'.-')

        if (label):
            plt.plot(self.axis,self.profile,'.-',label=label)
        else:
            plt.plot(self.axis,self.profile,'.-')

        if (show):
            plt.show()


def ReLU(x,a=0.5,m=1):
    # piecewise-linear function
    # can model Gamma( critical temperature gradient scale length ), for example
    if (x < a):
        return 0
    else:
        return m*(x-a)

def Step(x,a=0.5,m=1):
    # derivative of ReLU (is just step function)
    if (x < a):
        return 0
    else:
        return m


# this should be an init function
def init_density(n,debug=False):
    density = profile(n, grad=True, half=True, full=True)

    if (debug):
        density.plot(new_fig=True, label=r'$n$')
        density.grad.plot(label=r'$ \nabla n$')
        density.grad_log.plot(label=r'$\nabla \log n$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$n(0) = {:.1f} :: n(1) = {:.1f}$'.format(n_core, n_edge) )
        plt.grid()

    return density

T  = 2 # constant temp profile
Ba = 3 # average field on LCFS
R_major = 4 # meter
a_minor = 1 # meter
pressure = profile(n*T)
area     = profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


# compute Gamma as a function of critical density scale length
flux_slope = 1
critical_gradient = 1.5
D_neo = .1 # neoclassical particle diffusion 

def calc_Gamma(density,debug=False):
    Ln_inv     = -density.grad_log.profile # Lninv
    G_turb     = np.vectorize(ReLU)(Ln_inv, a=critical_gradient, m=flux_slope) 
    G_neo      = - D_neo * density.grad.profile
    dlogG_turb = np.vectorize(Step)(Ln_inv, a=critical_gradient, m=flux_slope)
    #dlogG_neo  = D_neo * density.grad.profile # negligible

    gamma = G_turb + G_neo
    
    Gamma     = profile(gamma,grad=True,half=True)
    dlogGamma = profile(dlogG_turb,grad=True,half=True)
    
    if (debug):
        Gamma.plot(new_fig=True,label=r'$\Gamma$')
        dlogGamma.plot(label=r'$\nabla \log \Gamma$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$Tcrit = {:.1f} ::  m = {:.1f}$'.format(critical_gradient, flux_slope) )

    return Gamma,dlogGamma


# compute F profile, given density and Gamma
#     I leave area out, because it is not evolved in time (it could be if VMEC changes)
def calc_F(density,Gamma, debug=False):
    F = area.profile / Ba**2 * Gamma.profile * pressure.profile**(3/2) / density.profile**(1/2)
    F = profile(F,half=True,grad=True)
    
    # set inner boundary condition
    F.minus.profile[0] = 0 # this actually 0 anyways, because F ~ Gamma, which depends on grad n, and grad n is samll near the core
    
    if (debug):
        F.plot(new_fig=True,label=r'$F$')
        density.grad_log.plot(label=r'$\nabla \log n$')
        Gamma.plot(label=r'$\Gamma$')
        plt.xlabel('radius')
        plt.legend()

    return F


# A and B profiles, for density evolution (there are different A,B for pressure)
def calc_AB_n(density,F,dlogGamma, debug=False):
    # original Barnes equations
    An_pos = profile( - (R_major/a_minor) * F.plus.profile / drho \
                         * density.profile / density.plus.profile**2 \
                         * dlogGamma.plus.profile )
    
    An_neg = profile( - (R_major/a_minor) * F.minus.profile / drho \
                         * density.profile / density.minus.profile**2 \
                         * dlogGamma.minus.profile )
    
    Bn     = profile( - (R_major/a_minor/drho) \
                  * (    F.plus.profile  \
                         * density.plus1.profile / density.plus.profile**2 \
                         * dlogGamma.plus.profile \
                      +  F.minus.profile  \
                         * density.minus1.profile / density.minus.profile**2 \
                         * dlogGamma.minus.profile  \
                    ) )

    if (debug):
        An_pos.plot(new_fig=True,label=r'$A_+[n]$')
        An_neg.plot(label=r'$A_-[n]$')
        Bn.plot(label=r'$B[n]$')
        plt.xlabel('radius')
        plt.legend()

    return An_pos, An_neg, Bn

# stub for new A,B coefficients that dont use F explicitly
#An_pos = profile( - (R_major/a_minor / drho) \
#                     * T**(3/2) / Ba**2 \   # need to make T.profile
#                     * Gamma.plus.grad.profile )

# compute psi, the matrix elements for tridiagonal inversion
def calc_psi_n(density,F,An_pos,An_neg,Bn,debug=False):
    # need to implement <|grad rho|>
    psi_n_plus  = profile( - (An_pos.profile - F.plus.profile / density.plus.profile / 4) \
                             / (area.profile * drho ) )
    
    psi_n_minus = profile( - (An_neg.profile + F.minus.profile / density.minus.profile / 4) \
                             / (area.profile * drho ) )
    
    psi_n_zero  = profile( - (Bn.profile \
                            + ( F.minus.profile / density.minus.profile \
                                - F.plus.profile / density.plus.profile )/ 4) \
                            / (area.profile * drho ) )

    if (debug):
        psi_n_plus.plot(new_fig=True, label=r'$\psi^n_+$')
        psi_n_minus.plot(label=r'$\psi^n_-$')
        psi_n_zero.plot(label=r'$\psi^n_0$')
        plt.legend()
        #plt.yscale('log')

    return psi_n_plus, psi_n_minus, psi_n_zero 






##### Evolve Trinity Equations


### Define LHS
# make tri-diagonal matrix

def tri_diagonal(a,b,c):
    N = len(a)
    M = np.diag(a)
    for j in np.arange(N-1):
        M[j,j+1] = b[j] # upper
        M[j+1,j] = c[j+1] # lower
    return M

# 1) should I treat the main equation as the middle of an array
# 2) or append the boundaries as ancillary to the main array?
# the first is more intuitive, but the second may be more efficient
arg_middle = np.s_[:-1]
#arg_middle = np.s_[1:-1]
N_radial_mat = N-1

def time_step_LHS(psi_n_plus,psi_n_minus,psi_n_zero,debug=False):
    M = tri_diagonal(psi_n_zero.profile[arg_middle], 
                    -psi_n_plus.profile[arg_middle], 
                    -psi_n_minus.profile[arg_middle]) # ! something might be wrong here, looks like it should be 1:, unlike the other rows
                    #-psi_n_minus.profile[1:])
    M[0,1] -= psi_n_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row
    I = np.identity(N_radial_mat)
    
    Amat = I + dtau*alpha * M
   
    if (debug):
        plt.figure()
        plt.imshow(Amat)
        #plt.show()

    return Amat


### Define RHS
def time_step_RHS(density,F,psi_n_plus,debug=False):
    n_prev = density.profile[arg_middle]
    force  =  - (1/drho/area.profile[arg_middle]) * F.grad.profile[arg_middle]
    #force  =  - (R_major/drho/area.profile[arg_middle]) * F.grad.profile[arg_middle]
    source = np.zeros(N_radial_mat) # temp, update with step or Gaussian?
    
    boundary = np.zeros(N_radial_mat)
    boundary[-1] =  psi_n_plus.profile[-2] * n_edge # !! which psi_j is this? -1 or -2?
    #boundary[-1] =  psi_n_plus.profile[-1] * n_edge # !! which psi_j is this? -1 or -2?
       # I think it should be -2 of the (full) psi profile, but -1 of the update vector
    
    bvec =  n_prev + dtau*(1 - alpha)*force + dtau*source + dtau*alpha*boundary
    #bvec = n_prev + dtau*(1 - alpha)*force + dtau*source + dtau*boundary # makes problems
    return bvec


def update_density(n_next,debug=False):

    n = np.concatenate([ [n_next[1]], n_next[1:], [n_edge] ])
    #n = np.concatenate([  n_next, [n_edge] ])
    #n = np.concatenate([ [n_next[0]], n_next, [n_edge] ])

    # temp fix to make n positive
    eps = 1e-4
#    n = np.abs(n) + eps # temp fix for (n < 0) 
    density = profile(n, grad=True, half=True, full=True)

    if (debug):
        density.plot(new_fig=True, label=r'$n$')
        density.grad.plot(label=r'$ \nabla n$')
        density.grad_log.plot(label=r'$\nabla \log n$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$n(0) = {:.1f} :: n(1) = {:.1f}$'.format(n_core, n_edge) )
        plt.grid()

    return density

# run calcs
_debug = False
density            = init_density(n,debug=_debug)
#Gamma, dlogGamma   = calc_Gamma(density           , debug=_debug)
#F                  = calc_F(density,Gamma         , debug=_debug)
#An_pos, An_neg, Bn = calc_AB_n(density,F,dlogGamma, debug=_debug)
#psi_n_plus, psi_n_minus, psi_n_zero = calc_psi_n(density,F,An_pos,An_neg,Bn, debug=_debug)

'''
    maybe, rather than passing some many profiles around,
    I should create a "time step" class
    which builds itself, by running each calc() and adding output as memeber objects
'''
#Amat = time_step_LHS(psi_n_plus,psi_n_minus,psi_n_zero)
#bvec = time_step_RHS(density,F,psi_n_plus)

### invert matrix
#Ainv = np.linalg.inv(Amat) # can also use scipy, or special tridiag method
#n_next = Ainv @ bvec

#n_prev = density.profile[arg_middle] # for plotting
#density = update_density(n_next,debug=False)


def diagnostic_1():

    plt.subplot(1,2,1)
    density.plot(label='T = {:.2e}'.format(Time))
    print('  Plot: t =',j)
    plt.subplot(1,2,2)
    #dlogGamma.plot(label='T = {:.2e}'.format(Time))
    Gamma.plot(label='T = {:.2e}'.format(Time))

def diagnostic_2():

    print('  Plot: t =',j)
    plt.subplot(2,3,1)
    density.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(2,3,2)
    Gamma.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(2,3,4)
    psi_n_plus.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(2,3,5)
    psi_n_zero.plot(label='T = {:.2e}'.format(Time))
    plt.subplot(2,3,6)
    psi_n_minus.plot(label='T = {:.2e}'.format(Time))

plt.figure()

j = 0 
Time = 0
while (j < N_steps):

    Gamma, dlogGamma   = calc_Gamma(density           , debug=_debug)
    F                  = calc_F(density,Gamma         , debug=_debug)
    An_pos, An_neg, Bn = calc_AB_n(density,F,dlogGamma, debug=_debug)
    psi_n_plus, psi_n_minus, psi_n_zero = calc_psi_n(density,F,An_pos,An_neg,Bn, debug=_debug)
    Amat = time_step_LHS(psi_n_plus,psi_n_minus,psi_n_zero)
    bvec = time_step_RHS(density,F,psi_n_plus)
    
    Ainv = np.linalg.inv(Amat) # can also use scipy, or special tridiag method
    n_next = Ainv @ bvec


    if not ( j % N_step_print):
#        if not (j < 100 or j > 600):
        #diagnostic_1()
        diagnostic_2()
     
    density = update_density(n_next,debug=_debug)
    #print(j, np.shape(Ainv), np.shape(bvec), np.shape(n_next), np.shape(density.profile))
    Time += dtau
    j += 1


def plot_1():
    plt.subplot(1,2,1)
    plt.title(r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau))
    plt.legend()
    plt.ylim(0,4.2)
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.title('Gamma(rho)')
    plt.legend()
    plt.grid()

def plot_2():
    plt.subplot(2,3,1)
    plt.title(r'$\alpha = {} :: d\tau = {:.3e}$'.format(alpha,dtau))
    #plt.legend()
    plt.ylim(0,4.2)
    plt.grid()
    
    plt.subplot(2,3,2)
    plt.title('Gamma(rho)')
    #plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    
    plt.subplot(2,3,4)
    plt.title('psi_+')
    #plt.legend()
    plt.grid()
    
    plt.subplot(2,3,5)
    plt.title('psi_0')
    #plt.legend()
    plt.grid()
    
    plt.subplot(2,3,6)
    plt.title('psi_-')
    #plt.legend()
    plt.grid()


plot_2()
#plot_1()
plt.show()
