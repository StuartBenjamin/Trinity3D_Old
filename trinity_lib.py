import numpy as np
import matplotlib.pyplot as plt

import models as mf # model functions


# This class contains TRINITY calculations and stores partial results as member objects
# There is a sub class for fluxes of each (n, pi, pe) evolution

class Trinity_Engine():
    def __init__(self, N = 10, # number of radial points
                       n_core = 4,
                       n_edge = 0.5,
                       T0 = 2,
                       R_major = 4,
                       a_minor = 1,
                       Ba = 3,
                       alpha = 1,          # explicit to implicit mixer
                       dtau  = 0.5,        # step size 
                       N_steps  = 1000,    # total Time = dtau * N_steps
                       N_prints = 10,
                       rho_edge = 0.8):

        self.N_radial = N  
        self.n_core   = n_core
        self.n_edge   = n_edge
        self.rho_edge = rho_edge
        self.rho_axis = np.linspace(0,rho_edge,N) # radial axis
        self.drho     = 1/N # for now assume equal spacing, 
                            #    could be computed in general

        ### will be from VMEC
        self.Ba      = Ba # average field on LCFS
        self.R_major = R_major # meter
        self.a_minor = a_minor # meter
        self.area     = profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


        ### init profiles
        #     temporary profiles
        n  = (n_core - n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
        self.n  = n
        self.T0 = T0 # constant temp profile, could be retired
        pi = T0*n
        pe = T0*n
        # save
        self.density     = init_profile(n)
        self.pressure_i  = init_profile(pi)
        self.pressure_e  = init_profile(pe)
        # should I split this out? decide later

        ### init transport variables
        zeros =  profile( np.zeros(N) )
        self.Gamma     = zeros 
        self.Qi        = zeros
        self.Qe        = zeros
        self.dlogGamma = zeros
        self.dlogQi    = zeros
        self.dlogQe    = zeros

        ### init flux coefficients
        self.Cn_n  = 0
        self.Cn_pi = 0 
        self.Cn_pe = 0


    # unused, archaic, can be delated
    def evolve_time(self):
        pass

    def update(self):

        density, foo, bar = update_state(self.y_next)
        #density, pressure_i, pressure_e = trl.update_state(y_next)
        pressure_i = init_profile( density.profile * self.T0 )
        pressure_e = init_profile( density.profile * self.T0 )

        self.density    = density
        self.pressure_i = pressure_i
        self.pressure_e = pressure_e


    # this is a toy model of Flux based on ReLU + neoclassical
    #     to be replaced by GX or STELLA import module
    def model_flux(self,
                   # neoclassical diffusion coefficient
                   D_n  = .1, 
                   D_pi = .1, 
                   D_pe = .1,
                   # critical gradient
                   cg_n  = 1.5, 
                   cg_pi = 0.5,
                   cg_pe = 1.5,
                   # slope of flux(Ln) after onset
                   s_n  = 1, 
                   s_pi = 1,
                   s_pe = 1 ):

        ### calc gradients
        grad_n  = self.density.grad.profile 
        grad_pi = self.pressure_i.grad.profile
        grad_pe = self.pressure_e.grad.profile
        Ln_inv  = - self.density.grad_log.profile 
        Lpi_inv = - self.pressure_i.grad_log.profile
        Lpe_inv = - self.pressure_e.grad_log.profile
        # should add a or R for normalization (fix later)

        ### model functions
        relu = np.vectorize(mf.ReLU)
        G_turb  = relu(Ln_inv,  a=n_critical_gradient,  m=n_flux_slope)
        Qi_turb = relu(Lpi_inv, a=pi_critical_gradient, m=pi_flux_slope)
        Qe_turb = relu(Lpe_inv, a=pe_critical_gradient, m=pe_flux_slope)
        G_neo   = - D_neo  * grad_n
        Qi_neo  = - Pi_neo * grad_pi
        Qe_neo  = - Pe_neo * grad_pe
    
        gamma = G_turb + G_neo
        qi    = Qi_turb + Qi_neo
        qe    = Qe_turb + Qi_neo
   
        # this block could be merged into the 'save'
        Gamma     = profile(gamma,grad=True,half=True)
        Qi        = profile(qi,   grad=True,half=True)
        Qe        = profile(qe,   grad=True,half=True)
        # this is incorrect, it takes grad log wrt to x instead of Ln(x) 
        dlogGamma = Gamma.grad_log
        dlogQi    = Qi.grad_log
        dlogQe    = Qe.grad_log
    
        # save
        self.Gamma     = Gamma
        self.dlogGamma = dlogGamma   
        self.Qi        = Qi
        self.Qe        = Qe
        self.dlogQi    = dlogQi
        self.dlogQe    = dlogQe

    def normalize_fluxes(self):

        # load
        density    = self.density
        pressure_i = self.pressure_i
        pressure_e = self.pressure_e
        Gamma = self.Gamma
        Qi    = self.Qi
        Qe    = self.Qe

        # calc
        Fn, Fpi, Fpe = calc_F3(density,pressure_i,pressure_e,Gamma,Qi,Qe)

        # save
        self.Fn  = Fn
        self.Fpi = Fpi
        self.Fpe = Fpe


    # A and B profiles for density and pressure
    def calc_AB(self):
        
        # load
        n         = self.density
        pi        = self.pressure_i
        pe        = self.pressure_e
        Fn        = self.Fn
        dlogGamma = self.dlogGamma
        #Fpi       = self.Fpi
        #Fpe       = self.Fpe
        #dlogQi    = self.dlogQi
        #dlogQe    = self.dlogQe

        # normalization
        norm = (self.R_major / self.a_minor) / self.drho 

        # calculate and save
        self.Cn_n  = flux_coefficients(n,  Fn, dlogGamma, norm)
        self.Cn_pi = flux_coefficients(pi, Fn, dlogGamma, norm)
        self.Cn_pe = flux_coefficients(pe, Fn, dlogGamma, norm)
    

# the class computes and stores normalized flux F, AB coefficients, and psi for the tridiagonal matrix
# it will need a flux Q, and profiles nT
# it should know whether ions or electrons are being computed, or density...
class flux_coefficients():

    # x is state vector (n, pi, pe)
    # Y is normalized flux (F,I)
    # Z is dlog flux (d log Gamma / d L_x ), evaluated at +- half step
    def __init__(self,x,Y,Z,norm):

        self.state   = x
        self.flux    = Y # this is normalized flux F,I
        self.logFlux = Z # this is Gamma,Q
        self.norm    = norm # normalizlation constant (R/a)/drho

        # plus,minus,zero : these are the A,B coefficients
        self.plus  = self.C_plus()
        self.minus = self.C_minus()
        self.zero  = self.C_zero()


    def C_plus(self):

        x  = self.state.profile
        xp = self.state.plus.profile
        Yp = self.flux.plus.profile
        Zp = self.logFlux.plus.profile
       
        norm = self.norm
        Cp = - x / xp**2 * Yp * Zp * norm
        return profile(Cp)

    def C_minus(self):

        x  = self.state.profile
        xm = self.state.minus.profile
        Ym = self.flux.minus.profile
        Zm = self.logFlux.minus.profile
        
        norm = self.norm
        Cm = - x / xm**2 * Ym * Zm * norm
        #Cm = x / xm**2 * Ym * Zm / dRho # typo in notes?
        return profile(Cm)

    def C_zero(self):

        x  = self.state.profile
        xp = self.state.plus.profile
        xm = self.state.minus.profile
        xp1 = self.state.plus1.profile
        xm1 = self.state.minus1.profile
        
        Yp = self.flux.plus.profile
        Zp = self.logFlux.plus.profile
        Ym = self.flux.minus.profile
        Zm = self.logFlux.minus.profile
        
        norm = self.norm
        cp = xp1 / xp**2 * Yp * Zp
        cm = xm1 / xm**2 * Ym * Zm
        Cz = ( cp + cm ) * norm
        return profile(Cz)


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
#     with default gradients, half steps, and full steps
def init_profile(x,debug=False):

    X = profile(x, grad=True, half=True, full=True)
    return X



### Calculate Transport Coefficients for Density

# compute Gamma as a function of critical density scale length
flux_slope = 1
critical_gradient = 1.5
D_neo = .1 # neoclassical particle diffusion 

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
    B     = profile(  (R_major/a_minor/drho) \
    # extra (-) was here 
    #B     = profile( - (R_major/a_minor/drho) \
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


# stub for new A,B coefficients that dont use F explicitly
#An_pos = profile( - (R_major/a_minor / drho) \
#                     * T**(3/2) / Ba**2 \   # need to make T.profile
#                     * Gamma.plus.grad.profile )



### from Sarah 11/8
D_neo  = .1 # neoclassical particle diffusion
Pi_neo = .1 # keep neoclassical pressure same as diffusion for now
Pe_neo = .1

n_flux_slope        = 1
pi_flux_slope        = 1
pe_flux_slope        = 1
n_critical_gradient = 1.5
pi_critical_gradient = .5
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
def calc_AB(density,pressure_i, pressure_e,Fn,Fpi,Fpe,dlogGamma,dlogQ_i,dlogQ_e, debug=False):

    # original Barnes equations (7.60, 7.61)

    An_pos , An_neg , Bn  = calc_AB_gen(density,    Fn,dlogGamma)
    Api_pos, Api_neg, Bpi = calc_AB_gen(pressure_i, Fn,dlogGamma) # note: this deriv is wrt to LT not Ln (!)
    Ape_pos, Ape_neg, Bpe = calc_AB_gen(pressure_e, Fn,dlogGamma)

    return An_pos, An_neg, Bn, Api_pos, Api_neg, Bpi, Ape_pos, Ape_neg, Bpe


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
            # this (-) is a surprise. It disagrees with Barnes thesis
                            - ( Fn.minus.profile / density.minus.profile \
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
# bug with (-) sign??
#    M_nn = tri_diagonal(psi_nn_zero.profile,
#                       -psi_nn_plus.profile,
#                       -psi_nn_minus.profile)
#    M_nn[0,1] -= psi_nn_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row
    M_nn = tri_diagonal(psi_nn_zero.profile,
                       psi_nn_plus.profile,
                       psi_nn_minus.profile)
## (!) I'm not enitrely sure which sign is correct.
#  My brain says the (+), but empirically (-) seems more smooth
#  overall the distinction isn't huge for now
#    M_nn[0,1] += psi_nn_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row
    M_nn[0,1] -= psi_nn_minus.profile[0]  # for boundary condition, add the second value of psi, to matrix element in second column of first row

    # Not sure of the n-p relationship here so this'll need to be changed
    M_npi = tri_diagonal(psi_npi_zero.profile,
                         psi_npi_plus.profile,
                         psi_npi_minus.profile)

    M_npe = tri_diagonal(psi_npe_zero.profile,
                         psi_npe_plus.profile,
                         psi_npe_minus.profile)

# I'm not sure if these need to be here, since they don't multiply n
#    (!!!) LOOK HERE, if hunting for bugs
#    M_npi[0,1] -= (psi_npi_minus.profile[0])  
#    M_npe[0,1] -= (psi_npe_minus.profile[0]) 

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
    p_nn  = psi_nn[:-1, :-1]          # drop the last point for Dirchlet boundary
    p_npi = psi_npi[:-1, :-1]
    p_npe = psi_npe[:-1, :-1]
    M = np.block([[ p_nn, p_npi, p_npe ],
                  [Z    , I    , Z     ],
                  [Z    , Z    , I     ]])
    I3 = np.block([[I, Z, Z ],
                   [Z, I, Z ],
                   [Z, Z, I ]])
    Amat = I3 - dtau*alpha * M
   
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
    force_n   =  - (1/drho/area.profile[arg_middle]) * dFdrho_n[arg_middle]
    source_n  = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Sn_height,sigma=Sn_width)

    # unused for now
    dFdrho_pi = (Fpi.plus.profile - Fpi.minus.profile)/2
    dFdrho_pe = (Fpe.plus.profile - Fpe.minus.profile)/2
#    force_pi  =  - (1/drho/area.profile[arg_middle]) * dFdrho_pi[arg_middle]
#    force_pe  =  - (1/drho/area.profile[arg_middle]) * dFdrho_pe[arg_middle]
    force_pi = 0
    force_pe = 0
    source_pi = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Spi_height,sigma=Spi_width)
    source_pe = np.vectorize(mf.Gaussian)(rho_axis[:-1], A=Spe_height,sigma=Spe_width)

    # init boundary condition
    N = len(density.profile)
    N_radial_mat = N-1
    boundary_n  = np.zeros(N_radial_mat)
    boundary_pi = np.zeros(N_radial_mat)
    boundary_pe = np.zeros(N_radial_mat)
    # add information from Dirchlet point
    boundary_n[-1]   = psi_nn[-2,-1] * n_edge # get last column of second to last row
    # here this is a (-) from flipping the psi
    boundary_pi[-1] = -psi_npi[-2,-1]*pi_edge
    boundary_pe[-1] = -psi_npe[-2,-1]*pi_edge

    # should each psi have its own bvec? rename bvec to bvec_n if so
    bvec_n  =  n_prev + dtau*(1 - alpha)*force_n + dtau*source_n + dtau*alpha*boundary_n
    bvec_pi =  pi_prev + dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
    bvec_pe =  pe_prev + dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe
    
    # there was a major bug here with the pressure parts of RHS state vector
    bvec3 = np.concatenate( [bvec_n, 0*bvec_pi, 0*bvec_pe] )
    return bvec3


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
