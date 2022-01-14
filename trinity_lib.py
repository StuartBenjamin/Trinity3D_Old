import numpy as np
import matplotlib.pyplot as plt

import models as mf # model functions


# This class contains TRINITY calculations and stores partial results as member objects
# There is a sub class for fluxes of each (n, pi, pe) evolution

class Trinity_Engine():
    def __init__(self, N = 10, # number of radial points
                       n_core = 4,
                       n_edge = 0.5,
                       pi_core = 8,
                       pi_edge = 2,
                       pe_core = 3,
                       pe_edge = .3,
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
        self.pi_core   = pi_core
        self.pi_edge   = pi_edge
        self.pe_core   = pe_core
        self.pe_edge   = pe_edge
        self.rho_edge = rho_edge
        self.drho     = 1/N # for now assume equal spacing, 
                            #    could be computed in general
        self.dtau     = dtau
        self.alpha    = alpha

        ### will be from VMEC
        self.Ba      = Ba # average field on LCFS
        self.R_major = R_major # meter
        self.a_minor = a_minor # meter
        self.area     = profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


        ### init profiles
        #     temporary profiles
        rho_axis = np.linspace(0,rho_edge,N) # radial axis
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
        self.rho_axis = rho_axis

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

        ### init psi profiles
        self.psi_nn  = 0
        self.psi_npi = 0
        self.psi_npe = 0

        ### init flux models
        self.model_G  = mf.Flux_model()
        self.model_Qi = mf.Flux_model()
        self.model_Qe = mf.Flux_model()

    # this is a toy model of Flux based on ReLU + neoclassical
    #     to be replaced by GX or STELLA import module
    def compute_flux(self):

        ### calc gradients
        grad_n  = self.density.grad.profile 
        grad_pi = self.pressure_i.grad.profile
        grad_pe = self.pressure_e.grad.profile
        Ln_inv  = - self.density.grad_log.profile 
        Lpi_inv = - self.pressure_i.grad_log.profile
        Lpe_inv = - self.pressure_e.grad_log.profile

        # run model (opportunity for parallelization)
        Lx = np.array( [Ln_inv, Lpi_inv, Lpe_inv] )

        G_neo  = - self.model_G.neo  * grad_n
        Qi_neo = - self.model_Qi.neo * grad_pi
        Qe_neo = - self.model_Qe.neo * grad_pe
        
        s   = self
        vec = np.vectorize
        G  = vec(s.model_G .flux)(*Lx) + G_neo 
        Qi = vec(s.model_Qi.flux)(*Lx) + Qi_neo
        Qe = vec(s.model_Qe.flux)(*Lx) + Qe_neo

        # derivatives
        G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients)(*Lx)
        Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(*Lx)
        Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(*Lx)

        # save
        self.Gamma     = profile(G, half=True)
        self.Qi        = profile(Qi, half=True) 
        self.Qe        = profile(Qe, half=True) 
        
        self.G_n   = profile(G_n , half=True)
        self.G_pi  = profile(G_pi, half=True)
        self.G_pe  = profile(G_pe, half=True)
        self.Qi_n   = profile(Qi_n , half=True)
        self.Qi_pi  = profile(Qi_pi, half=True)
        self.Qi_pe  = profile(Qi_pe, half=True)
        self.Qe_n   = profile(Qe_n , half=True)
        self.Qe_pi  = profile(Qe_pi, half=True)
        self.Qe_pe  = profile(Qe_pe, half=True)

        # temporary (backward compatibility)
        self.dlogGamma = self.G_n
        self.dlogQi    = self.G_pi
        self.dlogQe    = self.G_pe




    # this is a toy model of Flux based on ReLU + neoclassical
    #     to be replaced by GX or STELLA import module
    def model_flux(self,
                   # neoclassical diffusion coefficient
                   D_n  = .1, 
                   D_pi = .1, 
                   D_pe = .1,
                   # critical gradient
                   n_critical_gradient  = 1.5, 
                   pi_critical_gradient = 0.5,
                   pe_critical_gradient = 1.5,
                   # slope of flux(Ln) after onset
                   n_flux_slope  = 1, 
                   pi_flux_slope = 1,
                   pe_flux_slope = 1 ):

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
        G_neo   = - D_n  * grad_n
        Qi_neo  = - D_pi * grad_pi
        Qe_neo  = - D_pe * grad_pe
    
        gamma = G_turb + G_neo
        qi    = Qi_turb + Qi_neo
        qe    = Qe_turb + Qi_neo
   
        # this block could be merged into the 'save'
        Gamma     = profile(gamma,grad=True,half=True)
        Qi        = profile(qi,   grad=True,half=True)
        Qe        = profile(qe,   grad=True,half=True)
        # this is incorrect, it takes grad log wrt to x instead of Ln(x) 
        #   also, in general there will be a Jacobian of derivatives with respect to Ln, LTi, and LTe
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
        n     = self.density.profile
        pi    = self.pressure_i.profile
        pe    = self.pressure_e.profile
        Gamma = self.Gamma.profile
        Qi    = self.Qi.profile
        Qe    = self.Qe.profile
        area  = self.area.profile
        Ba    = self.Ba

        # calc
        #Fn, Fpi, Fpe = calc_F3(density,pressure_i,pressure_e,Gamma,Qi,Qe)
        A = area / Ba**2
        Fn = A * Gamma * pi**(1.5) / n**(0.5)
        # unused for now
        Fpi = A * Qi * pi**(2.5) / n**(1.5)
        Fpe = A * Qe * pi**(2.5) / n**(1.5)

        Fn  = profile(Fn,half=True,grad=True)
        Fpi = profile(Fpi,half=True,grad=True)
        Fpe = profile(Fpe,half=True,grad=True)
        # set inner boundary condition
        Fn .minus.profile[0] = 0
        Fpi.minus.profile[0] = 0
        Fpe.minus.profile[0] = 0
        # this actually 0 anyways, 
        #    because F ~ Gamma, which depends on grad n, 
        #    and grad n is small near the core

        # save
        self.Fn  = Fn
        self.Fpi = Fpi
        self.Fpe = Fpe


    # Compute A and B profiles for density and pressure
    #    this involves finite difference gradients
    def calc_flux_coefficients(self):
        
        # load
        n         = self.density
        pi        = self.pressure_i
        pe        = self.pressure_e
        Fn        = self.Fn
        dlogGamma = self.dlogGamma
        # new for pressure
        Fpi       = self.Fpi
        Fpe       = self.Fpe
        dlogQi    = self.dlogQi
        dlogQe    = self.dlogQe

        # normalization
        norm = (self.R_major / self.a_minor) / self.drho 

        # calculate and save
        self.Cn_n  = flux_coefficients(n,  Fn, dlogGamma, norm)
        self.Cn_pi = flux_coefficients(pi, Fn, dlogGamma, norm) # this needs to use dlog Gamma / d kappa_pi
        self.Cn_pe = flux_coefficients(pe, Fn, dlogGamma, norm)

        # new
        self.Cpi_n  = flux_coefficients(n,  Fpi, dlogQi, norm)
        self.Cpi_pi = flux_coefficients(pi, Fpi, dlogQi, norm) 
        self.Cpi_pe = flux_coefficients(pe, Fpi, dlogQi, norm)
        self.Cpe_n  = flux_coefficients(n,  Fpe, dlogQe, norm)
        self.Cpe_pi = flux_coefficients(pi, Fpe, dlogQe, norm)
        self.Cpe_pe = flux_coefficients(pe, Fpe, dlogQe, norm)
        # maybe these class definitions can be condensed

        # mu coefficients
        # needs kappas, should implement into profile
        # also 0 when G=H=K=0
        self.mu1 = 0
        self.mu2 = 0
        self.mu3 = 0

    def calc_psi_n(self):
    
        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1 
        geometry_factor = - grho / self.area.profile * self.drho
    
        # load
        Fnp = self.Fn.plus#.profile
        Fnm = self.Fn.minus#.profile
        n_p = self.density.plus.profile
        n_m = self.density.minus.profile
        pi_plus  = self.pressure_i.plus.profile
        pi_minus = self.pressure_i.minus.profile
        pe_plus  = self.pressure_e.plus.profile
        pe_minus = self.pressure_e.minus.profile
        #
        An_pos = self.Cn_n.plus.profile
        An_neg = self.Cn_n.minus.profile
        Bn     = self.Cn_n.zero.profile
        Ai_pos = self.Cn_pi.plus.profile
        Ai_neg = self.Cn_pi.minus.profile
        Bi     = self.Cn_pi.zero.profile
        Ae_pos = self.Cn_pe.plus.profile
        Ae_neg = self.Cn_pe.minus.profile 
        Be     = self.Cn_pe.zero.profile 
    
        # tri diagonal matrix elements
        g = geometry_factor
        psi_nn_plus  = g * (An_pos - Fnp / n_p / 4) 
        psi_nn_minus = g * (An_neg + Fnm / n_m / 4) 
        psi_nn_zero  = g * (Bn - ( Fnm/n_m - Fnp/n_p ) / 4) 
        # this (-) is a surprise. It disagrees with Barnes thesis
                                
        psi_npi_plus  = g * (Ai_pos + 3*Fnp / pi_plus / 4) 
        psi_npi_minus = g * (Ai_neg - 3*Fnm / pi_minus / 4) 
        psi_npi_zero  = g * (Bi - 3./4*( Fnm/pi_minus - Fnp/pi_plus) ) 
    
        psi_npe_plus  = g * Ae_pos
        psi_npe_minus = g * Ae_neg
        psi_npe_zero  = g * Be

        # save (automatically computes matricies in class function)
        self.psi_nn  = psi_profiles(psi_nn_zero,
                                   psi_nn_plus,
                                   psi_nn_minus, dirchlet=True)

        self.psi_npi = psi_profiles(psi_npi_zero,
                                   psi_npi_plus,
                                   psi_npi_minus)
        
        self.psi_npe = psi_profiles(psi_npe_zero,
                                   psi_npe_plus,
                                   psi_npe_minus)
        # I'm not sure if these need to be here, since they don't multiply n
        #    (!!!) LOOK HERE, if hunting for bugs
        #    M_npi[0,1] -= (psi_npi_minus.profile[0])  
        #    M_npe[0,1] -= (psi_npe_minus.profile[0]) 
   
    def calc_psi_pi(self):
    
        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1 
        geometry_factor = - grho / self.area.profile * self.drho
    
        # load
        F_p = self.Fpi.plus#.profile
        F_m = self.Fpe.minus#.profile
        n     = self.density.profile
        n_p = self.density.plus.profile
        n_m = self.density.minus.profile
        pi_plus  = self.pressure_i.plus.profile
        pi_minus = self.pressure_i.minus.profile
        pe_plus  = self.pressure_e.plus.profile
        pe_minus = self.pressure_e.minus.profile
        #
        An_pos = self.Cpi_n.plus.profile
        An_neg = self.Cpi_n.minus.profile
        Bn     = self.Cpi_n.zero.profile
        Ai_pos = self.Cpi_pi.plus.profile
        Ai_neg = self.Cpi_pi.minus.profile
        Bi     = self.Cpi_pi.zero.profile
        Ae_pos = self.Cpi_pe.plus.profile
        Ae_neg = self.Cpi_pe.minus.profile 
        Be     = self.Cpi_pe.zero.profile 
        #
        mu1 = self.mu1 # should be profiles when implemented
        mu2 = self.mu2 #   now these are all 0
        mu3 = self.mu3
    
        # tri diagonal matrix elements
        g = geometry_factor
        psi_pin_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1 / n 
        psi_pin_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1 / n
        psi_pin_zero  = g * (Bn -  3/4 * ( F_m/n_m - F_p/n_p ) ) 
        # this (-) is a surprise. It disagrees with Barnes thesis
                                
        psi_pipi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) 
        psi_pipi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) 
        psi_pipi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) ) 
    
        psi_pipe_plus  = g * Ae_pos
        psi_pipe_minus = g * Ae_neg
        psi_pipe_zero  = g * Be

        # save (automatically computes matricies in class function)
        self.psi_pin  = psi_profiles(psi_pin_zero,
                                     psi_pin_plus,
                                     psi_pin_minus)

        self.psi_pipi = psi_profiles(psi_pipi_zero,
                                     psi_pipi_plus,
                                     psi_pipi_minus, dirchlet=True)
        
        self.psi_pipe = psi_profiles(psi_pipe_zero,
                                     psi_pipe_plus,
                                     psi_pipe_minus)

    def calc_y_next(self):
        
        # load matrix
        M_nn  = self.psi_nn.matrix
        M_npi = self.psi_npi.matrix
        M_npe = self.psi_npe.matrix
        # load profiles
        n   = self.density
        pi  = self.pressure_i
        pe  = self.pressure_e
        Fn  = self.Fn
        Fpi = self.Fpi
        Fpe = self.Fpe

        # Invert Ax = b
        Amat = self.time_step_LHS()
        bvec = self.time_step_RHS()
        #Amat = time_step_LHS3(M_nn, M_npi, M_npe)
        #bvec = time_step_RHS3(n,pi,pe,
        #                      Fn,Fpi,Fpe,
        #                      M_nn, M_npi, M_npe)
        # can also use scipy, or special tridiag method
        Ainv = np.linalg.inv(Amat) 

        #y_next = Ainv @ bvec
        self.y_next = Ainv @ bvec
    

    def time_step_LHS(self):
 
        # load
        M_nn  = self.psi_nn .matrix[:-1, :-1]         
        M_npi = self.psi_npi.matrix[:-1, :-1]         
        M_npe = self.psi_npe.matrix[:-1, :-1]         
        # dropping the last point for Dirchlet boundary
        M_pin  = self.psi_pin .matrix[:-1, :-1]         
        M_pipi = self.psi_pipi.matrix[:-1, :-1]         
        M_pipe = self.psi_pipe.matrix[:-1, :-1]         
 
        N_block = self.N_radial - 1
        I = np.identity(N_block)
        Z = I*0 # block of 0s
        #Z = np.zeros_like(I)
        
        ## build block-diagonal matrices
        M = np.block([[ M_nn , M_npi , M_npe  ],
                      #[ M_pin, M_pipi, M_pipe ],
                      [Z    , I    , Z     ],
                      [Z    , Z    , I     ]])
        I3 = np.block([[I, Z, Z ],
                       [Z, I, Z ],
                       [Z, Z, I ]])
        Amat = I3 - self.dtau * self.alpha * M
        return Amat
    
    ### Define RHS
    def time_step_RHS(self):
 
        # load
        n_prev  = self.density.profile   [:-1]
        pi_prev = self.pressure_i.profile[:-1]
        pe_prev = self.pressure_e.profile[:-1]
        Fnp     = self.Fn.plus.profile[:-1]
        Fnm     = self.Fn.minus.profile[:-1]
        Fip     = self.Fpi.plus.profile[:-1]
        Fim     = self.Fpi.minus.profile[:-1]
        area    = self.area.profile[:-1]
        rax     = self.rho_axis[:-1]
        drho    = self.drho
        alpha   = self.alpha
        dtau    = self.dtau
        # load matrix
        psi_nn  = self.psi_nn.matrix
        psi_npi = self.psi_npi.matrix
        psi_npe = self.psi_npe.matrix
        psi_pipi = self.psi_pipi.matrix
   
        g = - 1/drho/area
        force_n  = g * (Fnp - Fnm)/2
        force_pi = g * (Fip - Fim)/2
        force_pe = 0
    
        Gaussian  = np.vectorize(mf.Gaussian)
        source_n  = Gaussian(rax, A=Sn_height, sigma=Sn_width)
        source_pi = Gaussian(rax, A=Spi_height,sigma=Spi_width)
        source_pe = Gaussian(rax, A=Spe_height,sigma=Spe_width)
    
        # init boundary condition
        #N = len(density.profile)
        N_radial_mat = self.N_radial - 1
        boundary_n  = np.zeros(N_radial_mat)
        boundary_pi = np.zeros(N_radial_mat)
        boundary_pe = np.zeros(N_radial_mat)

        # add information from Dirchlet point
        n_edge  = self.density.profile[-1]
        pi_edge = self.pressure_i.profile[-1]
        pe_edge = self.pressure_e.profile[-1]

        # get last column of second to last row
        #       there should be  a (-) from flipping the psi
        boundary_n[-1]   = psi_nn [-2,-1] * self.n_edge   \
                         + psi_npi[-2,-1] * self.pi_edge \
                         + psi_npe[-2,-1] * self.pe_edge 
        ### There is a bug here! (1/12)
        #boundary_n[-1]   = psi_nn[-2,-1] * self.n_edge # get last column of second to last row
        #boundary_pi[-1] = psi_pipi[-2,-1] * pi_edge
#        boundary_pe[-1] = psi_npe[-2,-1] * self.pi_edge

    
        # should each psi have its own bvec? rename bvec to bvec_n if so
        bvec_n  =  n_prev  + dtau*(1 - alpha)*force_n  + dtau*source_n  + dtau*alpha*boundary_n
        bvec_pi =  pi_prev + dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
        bvec_pe =  pe_prev + dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe
        
        # there was a major bug here with the pressure parts of RHS state vector

        #bvec3 = np.concatenate( [bvec_n, 0*bvec_pi, 0*bvec_pe] )
        bvec3 = np.concatenate( [bvec_n, bvec_pi, 0*bvec_pe] )
        return bvec3

    def update(self):

        # load
        y_next  = self.y_next
        n_edge  = self.n_edge
        pi_edge = self.pi_edge
        pe_edge = self.pe_edge
        n_edge  = self.density.profile[-1]
        pi_edge = self.pressure_i.profile[-1]
        pe_edge = self.pressure_e.profile[-1]
        N_mat = self.N_radial - 1

        n_next, pi_next, pe_next = np.reshape( y_next,(3,N_mat) )
        # check if legit
        n = np.concatenate([ [n_next[1]], n_next[1:], [n_edge] ]) 
        pi = np.concatenate([ [pi_next[1]], pi_next[1:], [pi_edge] ]) 
        pe = np.concatenate([ [pe_next[1]], pe_next[1:], [pe_edge] ])

        density = profile(n, grad=True, half=True, full=True)
        #pressure_i = profile(pi, grad=True, half=True, full=True)
        #pressure_e = profile(pe, grad=True, half=True, full=True)
        pressure_i = init_profile( density.profile * self.T0 )
        pressure_e = init_profile( density.profile * self.T0 )

        self.density    = density
        self.pressure_i = pressure_i
        self.pressure_e = pressure_e


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

# This class organizes the psi-profiles in tri-diagonal matrix
class psi_profiles():

    def __init__(self,psi_zero,
                      psi_plus,
                      psi_minus,
                      dirchlet=False):

        # save profiles
        self.plus  = profile( psi_plus )
        self.minus = profile( psi_minus )
        self.zero  = profile( psi_zero )

        # formulate matrix
        M = tri_diagonal(psi_zero,
                         psi_plus,
                         psi_minus)

        if (dirchlet):
            # make modification for boundary condition
            M[0,1] -= psi_minus[0]  

        # save matrix
        self.matrix = M

# a general class for handling profiles (n, p, F, gamma, Q, etc)
# with options to evaluate half steps and gradients at init
class profile():
    # should consider capitalizing Profile(), for good python form
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
        deriv[0]  = deriv[1]      # should a one-sided stencil be used here too?
                                  # should I set it to 0? in a transport solver, is the 0th point on axis?
                                  # I don't think GX will be run for the 0th point. So should that point be excluded from TRINITY altogether?
                                  #      or should it be included as a ghost point?

        # this is a second order accurate one-sided stencil
        #deriv[-1]  = ( 3./2* xj[-1] - 2*xj[-2] + 1./2* xj[-3])  /dx
        deriv[-1]  = ( 3*xj[-1] -4*xj[-2] + xj[-3])  / (2*dx)

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


    # operator overloads that automatically dereference the profiles
    def __add__(A,B):
        if isinstance(B, A.__class__):
            return A.profile + B.profile
        else:
            return A.profile + B

    def __sub__(A,B):
        if isinstance(B, A.__class__):
            return A.profile - B.profile
        else:
            return A.profile - B

    def __mul__(A,B):
        if isinstance(B, A.__class__):
            return A.profile * B.profile
        else:
            return A.profile * B

    def __truediv__(A,B):
        if isinstance(B, A.__class__):
            return A.profile / B.profile
        else:
            return A.profile / B

    def __rmul__(A,B):
        return A.__mul__(B)


# Initialize Trinity profiles
#     with default gradients, half steps, and full steps
def init_profile(x,debug=False):

    X = profile(x, grad=True, half=True, full=True)
    return X



# stub for new A,B coefficients that dont use F explicitly
#An_pos = profile( - (R_major/a_minor / drho) \
#                     * T**(3/2) / Ba**2 \   # need to make T.profile
#                     * Gamma.plus.grad.profile )



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
#arg_middle = np.s_[:-1] # the purpose of this expression is to remove "magic numbers" where we drop the last point due to Dirchlet boundary condition



