import numpy as np
import matplotlib.pyplot as plt

import models as mf # model functions


# these parameters are set by the run

class Trinity_Engine():
    def __init__(self, N = 40, # number of radial points
                       n_core = 4,
                       n_edge = 0.2,
                       rho_edge = 0.8):

        self.N        = N       # Sarah, nice catch, this was a bug
        self.n_core   = n_core
        self.n_edge   = n_edge
        self.rho_edge = rho_edge
        self.rho_axis = np.linspace(0,rho_edge,N) # radial axis

# they are clumsily pasted for now,
# but should be read into some class

## Actually, ths "parameters" class should be expanded into a "model" class
#  and we can have it take care of all the calculations




# a general class for handling profiles (n, p, F, gamma, Q, etc)
# with options to evaluate half steps and gradients at init
class profile():
    def __init__(self,arr, grad=False, half=False, full=False):

        # take a 1D array to be density, for example
        self.profile = np.array(arr) 
        self.length  = len(arr)
        global rho_axis
        self.axis    = rho_axis
        # assumes fixed radial griding, which (if irregular) could also be a profile, defined as a function of index

        # pre-calculate gradients, half steps, or full steps
        if (grad):
            self.grad     =  profile(self.gradient(), half=half, full=full)
            self.grad_log =  profile(self.log_gradient(), half=half, full=full)

        if (half): # defines half step
            self.plus  = profile(self.halfstep_pos())
            self.minus = profile(self.halfstep_neg())

        if (full): # defines full stup
            self.plus1  = profile(self.fullstep_pos())
            self.minus1 = profile(self.fullstep_neg())

    # pos/neg are forward and backwards
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
        #deriv[0]  = ( -3./2* xj[0] + 2*xj[1] - 1./2* xj[2])  /dx
        #deriv[0]  = (xj[1] - xj[0])  /dx
        #deriv[-1] = (xj[-1] - xj[-2])/dx
        deriv[-1]  = ( 3./2* xj[-1] - 2*xj[-2] + 1./2* xj[-3])  /dx

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


### Calculate Transport Coefficients for Density

# compute Gamma as a function of critical density scale length
flux_slope = 1
critical_gradient = 1.5
D_neo = .1 # neoclassical particle diffusion 

# toy function for calculating Gamma (as ReLU)
# also returns d log ( Gamma ) / d ( 1 / Ln ), with ambiguous normalization
def calc_Gamma(density,debug=False):
    Ln_inv     = -density.grad_log.profile # Lninv
    G_turb     = np.vectorize(mf.ReLU)(Ln_inv, a=critical_gradient, m=flux_slope) 
    G_neo      = - D_neo * density.grad.profile
#    dlogG_turb = np.vectorize(mf.Step)(Ln_inv, a=critical_gradient, m=flux_slope) 
    #dlogG_turb = np.vectorize(mf.Step)(Ln_inv, a=critical_gradient, m=flux_slope) / density.profile
    ### Is this actually dlogGamma? if Gamma is ReLU, dGamma is step
    #dlogG_neo  = D_neo * density.grad.profile # negligible
    #dlogGamma = profile(dlogG_turb,grad=True,half=True)

    # for debugging turublent and neoclassical transport 
        # for debugging turublent and neoclassical transport separately
#    gamma = G_turb
#    gamma = G_neo
    gamma = G_turb + G_neo
    
    Gamma     = profile(gamma,grad=True,half=True)
    dlogGamma = Gamma.grad_log
    
    if (debug):
        Gamma.plot(new_fig=True,label=r'$\Gamma$')
        dlogGamma.plot(label=r'$\nabla \log \Gamma$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$Tcrit = {:.1f} ::  m = {:.1f}$'.format(critical_gradient, flux_slope) )

    return Gamma,dlogGamma


# compute F profile, given density and Gamma
def calc_F(density,Gamma,debug=False):
    F = area.profile / Ba**2 * Gamma.profile * temperature.profile**(3/2) * density.profile
    #F = area.profile / Ba**2 * Gamma.profile * pressure.profile**(3/2) / density.profile**(1/2)
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
# w is one of (density, ion pressure, electron pressure)
# dlogflux is one (Gamma Qi or Qe)
def calc_AB_gen(w,Flux,dlogFlux, debug=False):

    # original Barnes equations (7.60, 7.61)
    # The (R/a) can be removed, if we use (a/L) instead of R/L
    A_pos = profile( - (R_major/a_minor) * Flux.plus.profile / drho \
                         * w.profile / w.plus.profile**2 \
                         * dlogFlux.plus.profile )
    
    A_neg = profile( - (R_major/a_minor) * Flux.minus.profile / drho \
                         * w.profile / w.minus.profile**2 \
                         * dlogFlux.minus.profile )
    
    B     = profile( - (R_major/a_minor/drho) \
                  * (    Flux.plus.profile  \
                         * w.plus1.profile / w.plus.profile**2 \
                         * dlogFlux.plus.profile \
                      +  Flux.minus.profile  \
                         * w.minus1.profile / w.minus.profile**2 \
                         * dlogFlux.minus.profile  \
                    ) )

    if (debug):
        A_pos.plot(new_fig=True,label=r'$A_+[n]$')
        A_neg.plot(label=r'$A_-[n]$')
        B.plot(label=r'$B[n]$')
        plt.xlabel('radius')
        plt.legend()

    return A_pos, A_neg, B


### to be retired
# A and B profiles, for density evolution (there are different A,B for pressure)
def calc_AB_n(density,F,dlogGamma, debug=False):

    # original Barnes equations (7.60, 7.61)
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


# calulates the density contribution to n^m+1 
# returns a tridiagonal matrix
# warning! this returns a (N), but we need the N-1 for the block matrix
def calc_psi_nn(density,F,An_pos,An_neg,Bn,debug=False):
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

    # arg_middle drops the last point, which is fixed by Dirchlet BC
    M = tri_diagonal(psi_n_zero.profile, 
                    -psi_n_plus.profile, 
                    -psi_n_minus.profile)
    M[0,1] -= psi_n_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row
    return M


##### Evolve Trinity Equations

### Define LHS
# make tri-diagonal matrix

def tri_diagonal(a,b,c):
    N = len(a)
    M = np.diag(a)
    for j in np.arange(N-1):
        M[j,j+1] = b[j]   # upper, drop last point
        M[j+1,j] = c[j+1] # lower, drop first 
    return M

# 1) should I treat the main equation as the middle of an array
# 2) or append the boundaries as ancillary to the main array?
# the first is more intuitive, but the second may be more efficient
arg_middle = np.s_[:-1]
#arg_middle = np.s_[1:] # should I drop the Dirchlet boundary or Neumann?

# old
def time_step_LHS(psi_n_plus,psi_n_minus,psi_n_zero,debug=False):
    M = tri_diagonal(psi_n_zero.profile[arg_middle], 
                    -psi_n_plus.profile[arg_middle], 
                    -psi_n_minus.profile[arg_middle])
    M[0,1] -= psi_n_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row
    N = len(psi_n_plus.profile)
    N_radial_mat = N-1
    I = np.identity(N_radial_mat)
    
    Amat = I + dtau*alpha * M
   
    if (debug):
        plt.figure()
        plt.imshow(Amat)
        #plt.show()

    return Amat

def time_step_LHS3(psi_nn,debug=False):
    
    N_block = N_radial_points - 1
    I = np.identity(N_block)
    Z = I*0 # block of 0s

    #psi_nn = tri_diagonal(psi_n_zero.profile[arg_middle], 
    #                -psi_n_plus.profile[arg_middle], 
    #                -psi_n_minus.profile[arg_middle])
    #psi_nn[0,1] -= psi_n_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row

    
    ## build block-diagonal matrices
    b_nn = psi_nn[:-1, :-1]          # drop the last point for Dirchlet boundary
    M = np.block([[ b_nn, Z, Z ],
                  [Z    , I, Z ],
                  [Z    , Z, I ]])
    I3 = np.block([[I, Z, Z ],
                   [Z, I, Z ],
                   [Z, Z, I ]])
    Amat = I3 + dtau*alpha * M
   
    if (debug):
        plt.figure()
        plt.imshow(Amat)
        #plt.show()

    return Amat

### Define RHS
def time_step_RHS3(density,F,psi_nn,debug=False):
    n_prev = density.profile[arg_middle]

    dFdrho = (F.plus.profile - F.minus.profile)/2
    force  =  - (1/drho/area.profile[arg_middle]) * dFdrho[arg_middle]
    N = len(density.profile)
    N_radial_mat = N-1
    source = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Sn_height,sigma=Sn_width)
    
    boundary = np.zeros(N_radial_mat)
    boundary[-1] =  -psi_nn[-2,-1] * n_edge # get last column of second to last row
    
    bvec =  n_prev + dtau*(1 - alpha)*force + dtau*source + dtau*alpha*boundary

    # temp should be the current state (it doesn't change)
    temp = temperature.profile[arg_middle] # assume Te = Ti
    b3 = np.concatenate( [bvec, temp, temp] )
    return b3

# old
def time_step_RHS(density,F,psi_n_plus,debug=False):
    n_prev = density.profile[arg_middle]

    dFdrho = (F.plus.profile - F.minus.profile)/2
    force  =  - (1/drho/area.profile[arg_middle]) * dFdrho[arg_middle]
    #force  =  - (1/drho/area.profile[arg_middle]) * F.grad.profile[arg_middle]
    #force  =  - (R_major/drho/area.profile[arg_middle]) * F.grad.profile[arg_middle]
    N = len(density.profile)
    N_radial_mat = N-1
    source = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Sn_height,sigma=Sn_width)
    #source = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=35,sigma=0.3)
    
    boundary = np.zeros(N_radial_mat)
    boundary[-1] =  psi_n_plus.profile[-2] * n_edge # !! which psi_j is this? -1 or -2?
       # I think it should be -2 of the (full) psi profile, but -1 of the update vector
    
    bvec =  n_prev + dtau*(1 - alpha)*force + dtau*source + dtau*alpha*boundary
    #bvec = n_prev + dtau*(1 - alpha)*force + dtau*source + dtau*boundary # makes problems
    return bvec

# old
def update_density(n_next,debug=False):

    n = np.concatenate([ [n_next[1]], n_next[1:], [n_edge] ]) # check if legit
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

# unused
# updates the state vector (n,Ti,Te)
def update_state(n_next,debug=False):

    n = np.concatenate([ [n_next[1]], n_next[1:], [n_edge] ]) # check if legit
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
