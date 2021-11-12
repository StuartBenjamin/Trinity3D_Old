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


# Initialize Trinity profiles
#def init_profile(n,debug=False):
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
init_profile = init_density

#def init_pressure_i(pi,debug=False):
#    pressure_i = profile(pi, grad=True, half=True, full=True)
#
#    if (debug):
#        pressure_i.plot(new_fig=True, label=r'$p_i$')
#        pressure_i.grad.plot(label=r'$ \nabla p_i$')
#        pressure_i.grad_log.plot(label=r'$\nabla \log p_i$')
#        plt.xlabel('radius')
#        plt.legend()
#        plt.title(r'$p_i(0) = {:.1f} :: p_i(1) = {:.1f}$'.format(pi_core, pi_edge) )
#        plt.grid()
#
#    return pressure_i
#
#def init_pressure_e(pe,debug=False):
#    pressure_e = profile(pe, grad=True, half=True, full=True)
#
#    if (debug):
#        pressure_e.plot(new_fig=True, label=r'$p_e$')
#        pressure_e.grad.plot(label=r'$ \nabla p_e$')
#        pressure_e.grad_log.plot(label=r'$\nabla \log p_e$')
#        plt.xlabel('radius')
#        plt.legend()
#        plt.title(r'$p_e(0) = {:.1f} :: p_e(1) = {:.1f}$'.format(pe_core, pe_edge) )
#        plt.grid()
#
#    return pressure_e


### Calculate Transport Coefficients for Density

# compute Gamma as a function of critical density scale length
flux_slope = 1
critical_gradient = 1.5
D_neo = .1 # neoclassical particle diffusion 

# old
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

# old
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


### from Sarah 11/8
D_neo  = .1 # neoclassical particle diffusion
Pi_neo = .1 # keep neoclassical pressure same as diffusion for now
Pe_neo = .1

n_flux_slope        = 1
n_critical_gradient = 1.5
pi_flux_slope        = 1
pi_critical_gradient = .5
pe_flux_slope        = 1
pe_critical_gradient = 1.5

def calc_Flux(density,pressure_i,pressure_e,debug=False):
    Ln_inv     = -density.grad_log.profile # Lninv
    Lpi_inv    = -pressure_i.grad_log.profile
    Lpe_inv    = -pressure_e.grad_log.profile
    G_turb     = np.vectorize(mf.ReLU)(Ln_inv, a=n_critical_gradient, m=n_flux_slope)
    Qi_turb    = np.vectorize(mf.ReLU)(Lpi_inv, a=pi_critical_gradient, m=pi_flux_slope)
    Qe_turb    = np.vectorize(mf.ReLU)(Lpe_inv, a=pe_critical_gradient, m=pe_flux_slope)
    G_neo      = - D_neo  * density.grad.profile
    Qi_neo     = - Pi_neo * pressure_i.grad.profile
    Qp_neo     = - Pe_neo * pressure_e.grad.profile
    #dlogG_turb = np.vectorize(mf.Step)(Ln_inv, a=critical_gradient, m=flux_slope)
    #dlogQi_turb = np.vectorize(mf.Step)(Lpi_inv, a=critical_gradient, m=flux_slope)
    #dlogQe_turb = np.vectorize(mf.Step)(Lpe_inv, a=critical_gradient, m=flux_slope)
    #dlogG_neo  = D_neo * density.grad.profile # negligible

    #gamma = G_neo
    #gamma = G_turb
    gamma = G_turb + G_neo
    qi   = Qi_turb + Qi_neo
    qe   = Qe_turb + Qi_neo

    Gamma     = profile(gamma,grad=True,half=True)
    Qi       = profile(qi,grad=True,half=True)
    Qe       = profile(qe,grad=True,half=True)
    dlogGamma = Gamma.grad_log
    dlogQi    = Qi.grad_log
    dlogQe    = Qe.grad_log
    #dlogGamma = profile(dlogG_turb,grad=True,half=True)
    #dlogQ_i = profile(dlogQi_turb,grad=True,half=True)
    #dlogQ_e = profile(dlogQe_turb,grad=True,half=True)

    if (debug):
        Gamma.plot(new_fig=True,label=r'$\Gamma$')
        dlogGamma.plot(label=r'$\nabla \log \Gamma$')
        Q_i.plot(new_fig=True,label=r'$\Q_i$')
        dlogQ_i.plot(label=r'$\nabla \log \Q_i$')
        Q_e.plot(new_fig=True,label=r'$\Q_e$')
        dlogQ_e.plot(label=r'$\nabla \log \Q_e$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$Tcrit = {:.1f} ::  m = {:.1f}$'.format(critical_gradient, flux_slope) )

    return Gamma,dlogGamma,   Qi,dlogQi,   Qe,dlogQe


# compute F profile (normalized flux)
#    given density, pressure_i, pressure_e, Gamma,and the Qs
def calc_F3(density,pressure_i,pressure_e,Gamma,Q_i,Q_e,debug=False):
    Fn = area.profile / Ba**2 * Gamma.profile * pressure_i.profile**(1.5) / density.profile**(0.5)
    # these F's for heat flux are not correct (but they are not used right now)
    Fpi = area.profile / Ba**2 * Q_i.profile * temperature.profile**(3/2) * pressure_i.profile
    Fpe = area.profile / Ba**2 * Q_e.profile * temperature.profile**(3/2) * pressure_e.profile
    # Sarah the F's for density should all use the same Gamma
    Fn = profile(Fn,half=True,grad=True)
    Fpi = profile(Fpi,half=True,grad=True)
    Fpe = profile(Fpe,half=True,grad=True)

    # set inner boundary condition
    Fn.minus.profile[0], Fpi.minus.profile[0], Fpe.minus.profile[0] = 0,0,0
    # this actually 0 anyways, because F ~ Gamma, which depends on grad n, and grad n is small near the core

    if (debug):
        Fn.plot(new_fig=True,label=r'$F_n$')
        Fpi.plot(new_fig=True,label=r'$F_pi$')
        Fpe.plot(new_fig=True,label=r'$F_pe$')
        density.grad_log.plot(label=r'$\nabla \log n$')
        pressure_i.grad_log.plot(label=r'$\nabla \log p_i$')
        pressure_e.grad_log.plot(label=r'$\nabla \log p_e$')
        Gamma.plot(label=r'$\Gamma$')
        Q_i.plot(label=r'$\Q_i$')
        Q_e.plot(label=r'$\Q_e$')
        plt.xlabel('radius')
        plt.legend()

    return Fn, Fpi, Fpe


# A and B profiles for density and pressure
def calc_AB(density,pressure_i, pressure_e,Fn,Fpi,Fpe,dlogGamma,dlogQ_i,dlogQ_e,debug=False):

    # original Barnes equations (7.60, 7.61)
    # density
    An_pos = profile( - (R_major/a_minor) * Fn.plus.profile / drho \
                         * density.profile / density.plus.profile**2 \
                         * dlogGamma.plus.profile )

    An_neg = profile( - (R_major/a_minor) * Fn.minus.profile / drho \
                         * density.profile / density.minus.profile**2 \
                         * dlogGamma.minus.profile )

    Bn     = profile( - (R_major/a_minor/drho) \
                  * (    Fn.plus.profile  \
                         * density.plus1.profile / density.plus.profile**2 \
                         * dlogGamma.plus.profile \
                      +  Fn.minus.profile  \
                         * density.minus1.profile / density.minus.profile**2 \
                         * dlogGamma.minus.profile  \
                    ) )

    # pressure_i
    Ai_pos = profile( - (R_major/a_minor) * Fpi.plus.profile / drho \
                         * pressure_i.profile / pressure_i.plus.profile**2 \
                         * dlogQ_i.plus.profile )

    Ai_neg = profile( - (R_major/a_minor) * Fpi.minus.profile / drho \
                         * pressure_i.profile / pressure_i.minus.profile**2 \
                         * dlogQ_i.minus.profile )

    Bi     = profile( - (R_major/a_minor/drho) \
                  * (    Fpi.plus.profile  \
                         * pressure_i.plus1.profile / pressure_i.plus.profile**2 \
                         * dlogQ_i.plus.profile \
                      +  Fpi.minus.profile  \
                         * pressure_i.minus1.profile / pressure_i.minus.profile**2 \
                         * dlogQ_i.minus.profile  \
                    ) )

    # pressure_e
    Ae_pos = profile( - (R_major/a_minor) * Fpe.plus.profile / drho \
                         * pressure_e.profile / pressure_e.plus.profile**2 \
                         * dlogQ_e.plus.profile )

    Ae_neg = profile( - (R_major/a_minor) * Fpe.minus.profile / drho \
                         * pressure_e.profile / pressure_e.minus.profile**2 \
                         * dlogQ_e.minus.profile )

    Be     = profile( - (R_major/a_minor/drho) \
                  * (    Fpe.plus.profile  \
                         * pressure_e.plus1.profile / pressure_e.plus.profile**2 \
                         * dlogQ_e.plus.profile \
                      +  Fpe.minus.profile  \
                         * pressure_e.minus1.profile / pressure_e.minus.profile**2 \
                         * dlogQ_e.minus.profile  \
                    ) )

    if (debug):
        An_pos.plot(new_fig=True,label=r'$An_+$')
        An_neg.plot(label=r'$An_-$')
        Bn.plot(label=r'$B_n$')
        Ai_pos.plot(new_fig=True,label=r'$Ai_+$')
        Ai_neg.plot(label=r'$Ai_-$')
        Bi.plot(label=r'$B_i$')
        Ae_pos.plot(new_fig=True,label=r'$Ae_+$')
        Ae_neg.plot(label=r'$Ae_-$')
        Be.plot(label=r'$B_e$')
        plt.xlabel('radius')
        plt.legend()

    return An_pos, An_neg, Bn, Ai_pos, Ai_neg, Bi, Ae_pos, Ae_neg, Be


# stub for new A,B coefficients that dont use F explicitly
#An_pos = profile( - (R_major/a_minor / drho) \
#                     * T**(3/2) / Ba**2 \   # need to make T.profile
#                     * Gamma.plus.grad.profile )

# compute psi, the matrix elements for tridiagonal inversion
def calc_psi(density, pressure_i, pressure_e,Fn,Fpi,Fpe, \
                   An_pos,An_neg,Bn, Ai_pos,Ai_neg,Bi, \
                   Ae_pos,Ae_neg,Be,debug=False):

    # geometric factor
    grho = 1 # need to implement <|grad rho|>, by reading surface area from VMEC
    geometry_factor = - grho / area.profile * drho

    # tri diagonal matrix elements
    psi_nn_plus  = profile( geometry_factor * (An_pos.profile - Fn.plus.profile  / density.plus.profile / 4) )
    psi_nn_minus = profile( geometry_factor * (An_neg.profile + Fn.minus.profile / density.minus.profile / 4) )
    psi_nn_zero  = profile( geometry_factor * (Bn.profile \
                            + ( Fn.minus.profile / density.minus.profile \
                               -Fn.plus.profile  / density.plus.profile )/ 4) )

    psi_npi_plus  = profile( geometry_factor * (Ai_pos.profile + 3*Fn.plus.profile / pressure_i.plus.profile / 4) )
    psi_npi_minus = profile( geometry_factor * (Ai_neg.profile - 3*Fn.minus.profile / pressure_i.minus.profile / 4) )
    psi_npi_zero  = profile( geometry_factor * (Bi.profile \
                             - 3./4*(Fn.minus.profile / pressure_i.minus.profile \
                                   - Fn.plus.profile  / pressure_i.plus.profile )   ) )

    psi_npe_plus  = profile( geometry_factor * Ae_pos.profile )
    psi_npe_minus = profile( geometry_factor * Ae_neg.profile )
    psi_npe_zero  = profile( geometry_factor * Be.profile )

    if (debug):
        psi_nn_plus.plot(new_fig=True, label=r'$\psi^n_+$')
        psi_nn_minus.plot(label=r'$\psi^n_-$')
        psi_nn_zero.plot(label=r'$\psi^n_0$')
        psi_npi_plus.plot(new_fig=True, label=r'$\psi^p_i_+$')
        psi_npi_minus.plot(label=r'$\psi^p_i_-$')
        psi_npi_zero.plot(label=r'$\psi^p_i_0$')
        psi_npe_plus.plot(new_fig=True, label=r'$\psi^p_e_+$')
        psi_npe_minus.plot(label=r'$\psi^p_e_-$')
        psi_npe_zero.plot(label=r'$\psi^p_e_0$')
        plt.legend()
        #plt.yscale('log')


    # calulates the density contribution to n^m+1
    # returns a tridiagonal matrix
    # warning! this returns a (N), but we need the N-1 for the block matrix
    # arg_middle drops the last point, which is fixed by Dirchlet BC
    M_nn = tri_diagonal(psi_nn_zero.profile,
                       -psi_nn_plus.profile,
                       -psi_nn_minus.profile)
    M_nn[0,1] -= psi_nn_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row

    # Not sure of the n-p relationship here so this'll need to be changed
    M_npi = tri_diagonal(psi_npi_zero.profile,
                        -psi_npi_zero.profile,
                        -psi_npi_minus.profile)
    M_npi[0,1] -= (psi_npi_minus.profile[0])  # for boundary condition, add the second value of psi, to matrix element in second column of first row

    M_npe = tri_diagonal(psi_npe_zero.profile,
                        -psi_npe_zero.profile,
                        -psi_npe_minus.profile)
    M_npi[0,1] -= (psi_npe_minus.profile[0])  # for boundary condition, add the second value of psi, to matrix element in second column of first row

    return M_nn, M_npi, M_npe


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
arg_middle = np.s_[:-1] # the purpose of this expression is to remove "magic numbers" where we drop the last point due to Dirchlet boundary condition

# the "3" appended to LHS and RHS represents the statevector y = (n,pi,pe)
#    this could be removed later

# new code from Sarah
def time_step_LHS3(psi_nn,psi_npi,psi_npe,debug=False):
    
    N_block = N_radial_points - 1
    I = np.identity(N_block)
    Z = I*0 # block of 0s
    
    ## build block-diagonal matrices
    b_nn  = psi_nn[:-1, :-1]          # drop the last point for Dirchlet boundary
    b_npi = psi_npi[:-1, :-1]
    b_npe = psi_npe[:-1, :-1]
    #M = np.block([[ b_nn, b_npi, b_npe ],
    M = np.block([[ b_nn, Z    , Z     ],
                  [Z    , I    , Z     ],
                  [Z    , Z    , I     ]])
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
def time_step_RHS3(density,pressure_i,pressure_e,Fn,Fpi,Fpe,psi_nn,psi_npi,psi_npe,debug=False):
    n_prev  = density.profile[arg_middle]
    pi_prev = pressure_i.profile[arg_middle]
    pe_prev = pressure_e.profile[arg_middle]

    dFdrho_n  = (Fn.plus.profile - Fn.minus.profile)/2
    dFdrho_pi = (Fpi.plus.profile - Fpi.minus.profile)/2
    dFdrho_pe = (Fpe.plus.profile - Fpe.minus.profile)/2
    force_n   =  - (1/drho/area.profile[arg_middle]) * dFdrho_n[arg_middle]
#    force_pi  =  - (1/drho/area.profile[arg_middle]) * dFdrho_pi[arg_middle]
#    force_pe  =  - (1/drho/area.profile[arg_middle]) * dFdrho_pe[arg_middle]
    force_pi = 0
    force_pe = 0
    N = len(density.profile)
    N_radial_mat = N-1

    source_n  = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Sn_height,sigma=Sn_width)
    source_pi = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Spi_height,sigma=Spi_width)
    source_pe = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Spe_height,sigma=Spe_width)

    # init boundary condition
    boundary_n  = np.zeros(N_radial_mat)
    boundary_pi = np.zeros(N_radial_mat)
    boundary_pe = np.zeros(N_radial_mat)
    # add information from Dirchlet point
    boundary_n[-1]   = -psi_nn[-2,-1] * n_edge # get last column of second to last row
    boundary_pi[-1] = -psi_npi[-2,-1]*pi_edge
    boundary_pe[-1] = -psi_npe[-2,-1]*pi_edge

    # should each psi have its own bvec? rename bvec to bvec_n if so
    bvec_n  =  n_prev + dtau*(1 - alpha)*force_n + dtau*source_n + dtau*alpha*boundary_n
    bvec_pi =  pi_prev + dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
    bvec_pe =  pe_prev + dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe

    bvec3 = np.concatenate( [bvec_n, bvec_pi, bvec_pe] )
    return bvec3

# old
def time_step_LHS(psi_nn,debug=False):
    
    N_block = N_radial_points - 1
    I = np.identity(N_block)
    Z = I*0 # block of 0s

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
#old
def time_step_RHS(density,F,psi_nn,debug=False):
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

# updates the state vector (n,Ti,Te)
def update_state(y_next,debug=False):

    N = N_radial_points
    n_next, pi_next, pe_next = np.reshape(y_next,(3,N-1) )
    n = np.concatenate([ [n_next[1]], n_next[1:], [n_edge] ]) # check if legit
    pi = np.concatenate([ [pi_next[1]], pi_next[1:], [pi_edge] ]) # check if legit
    pe = np.concatenate([ [pe_next[1]], pe_next[1:], [pe_edge] ]) # check if legit
    density = profile(n, grad=True, half=True, full=True)
    pressure_i = profile(pi, grad=True, half=True, full=True)
    pressure_e = profile(pe, grad=True, half=True, full=True)

    if (debug):
        density.plot(new_fig=True, label=r'$n$')
        density.grad.plot(label=r'$ \nabla n$')
        density.grad_log.plot(label=r'$\nabla \log n$')
        plt.xlabel('radius')
        plt.legend()
        plt.title(r'$n(0) = {:.1f} :: n(1) = {:.1f}$'.format(n_core, n_edge) )
        plt.grid()

    return density, pressure_i, pressure_e
