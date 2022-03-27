import numpy as np
import matplotlib.pyplot as plt
import pdb

import models as mf 

import profiles as pf
profile           = pf.Profile
flux_coefficients = pf.Flux_coefficients
psi_profiles      = pf.Psi_profiles

# ignore divide by 0 warnings
#np.seterr(divide='ignore', invalid='ignore')

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
                       rho_edge = 0.8,
                       Sn_height  = 0,  
                       Spe_height = 0,
                       Spi_height = 0, 
                       Sn_width   = 0.1,   
                       Spi_width  = 0.1, 
                       Spe_width  = 0.1,  
                       Sn_center  = 0.0,   
                       Spi_center = 0.0, 
                       Spe_center = 0.0,  
                       model      = 'GX'
                       ):

        self.N_radial = N           # if this is total points, including core and edge, then GX simulates (N-2) points
        self.n_core   = n_core
        self.n_edge   = n_edge
        self.pi_core   = pi_core
        self.pi_edge   = pi_edge
        self.pe_core   = pe_core
        self.pe_edge   = pe_edge

        self.model    = model

        self.rho_edge = rho_edge
        rho_axis = np.linspace(0,rho_edge,N) # radial axis
        self.rho_axis = rho_axis
        pf.rho_axis   = rho_axis

        self.dtau     = dtau
        self.alpha    = alpha
        self.N_steps  = N_steps
        self.N_prints = N_prints

        self.Sn_height  = Sn_height  
        self.Spi_height = Spi_height 
        self.Spe_height = Spe_height 
        self.Sn_width   = Sn_width      
        self.Spi_width  = Spi_width   
        self.Spe_width  = Spe_width    
        self.Sn_center  = Sn_center   
        self.Spi_center = Spi_center 
        self.Spe_center = Spe_center  

        self.time = 0

        ### will be from VMEC
        self.Ba      = Ba # average field on LCFS
        self.R_major = R_major # meter
        self.a_minor = a_minor # meter

        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1
        #grho = -1
        # BUG: grho should be > 0; while geo_factor = - grho / drho / area should be negative
        #      see Barnes (7.62) and (7.115)
        #      but doing so causes fluxes to evolve in opposite direction
        # adding this "artificial" (-) to grho fixes it
        drho       = rho_edge / (N-1)
        area       = profile(np.linspace(0.01,a_minor,N), half=True) # parabolic area, simple torus
        self.grho  = grho
        self.drho  = drho
        self.area  = area
        self.geometry_factor = - grho / (drho * area.profile)

        ### init profiles
        #     temporary profiles, later init from VMEC
        n  = (n_core - n_edge)*(1 - (rho_axis/rho_edge)**2) + n_edge
        pi = (pi_core-pi_edge)*(1 - (rho_axis/rho_edge)**2) + pi_edge
        pe = (pe_core-pe_edge)*(1 - (rho_axis/rho_edge)**2) + pe_edge

        # save
        self.density     = init_profile(n)
        self.pressure_i  = init_profile(pi)
        self.pressure_e  = init_profile(pe)

        ### init transport variables
#        zeros =  profile( np.zeros(N) )
#        self.Gamma     = zeros 
#        self.Qi        = zeros
#        self.Qe        = zeros
#        self.dlogGamma = zeros
#        self.dlogQi    = zeros
#        self.dlogQe    = zeros
#
#        ### init flux coefficients
#        self.Cn_n  = 0
#        self.Cn_pi = 0 
#        self.Cn_pe = 0
#
#        ### init psi profiles
#        self.psi_nn  = 0
#        self.psi_npi = 0
#        self.psi_npe = 0


        ### sources
        # temp, Gaussian model. Later this should be adjustable
        Gaussian  = np.vectorize(mf.Gaussian)
        rax = rho_axis
        self.source_n  = Gaussian(rax, A=Sn_height , sigma=Sn_width , x0=Sn_center)
        self.source_pi = Gaussian(rax, A=Spi_height, sigma=Spi_width, x0=Spi_center)
        self.source_pe = Gaussian(rax, A=Spe_height, sigma=Spe_width, x0=Spe_center)


        ### init flux models
        if (model == 'GX'):
            fout = 'gx-files/temp.gx'
            gx = mf.GX_Flux_Model(fout)
            gx.init_geometry()
    
            self.f_cmd = fout
            self.model_gx = gx

        elif (model == 'diffusive'):
            bm = mf.Barnes_Model2()
            self.barnes_model = bm

        else:
            self.model_G  = mf.Flux_model()
            self.model_Qi = mf.Flux_model()
            self.model_Qe = mf.Flux_model()

    # this is a toy model of Flux based on ReLU + neoclassical
    #     to be replaced by GX or STELLA import module
    def compute_flux(self):

        ### calc gradients
        grad_n  = self.density.grad.   midpoints 
        grad_pi = self.pressure_i.grad.midpoints
        grad_pe = self.pressure_e.grad.midpoints

        ### new 3/14
        # use the positions from flux tubes in between radial grid steps
        kn  = - self.density.grad_log   .midpoints 
        kpi = - self.pressure_i.grad_log.midpoints
        kpe = - self.pressure_e.grad_log.midpoints
        ###


        # run model (opportunity for parallelization)
        #Lx = np.array( [Ln_inv, Lpi_inv, Lpe_inv] )

        G_neo  = - self.model_G.neo  * grad_n
        Qi_neo = - self.model_Qi.neo * grad_pi
        Qe_neo = - self.model_Qe.neo * grad_pe
       

        ### Change these function calls to evaluations at the half grid
        s   = self
        vec = np.vectorize
        #G  = vec(s.model_G .flux)(*Lx) + G_neo 
        #Qi = vec(s.model_Qi.flux)(*Lx) + Qi_neo
        #Qe = vec(s.model_Qe.flux)(*Lx) + Qe_neo
        G  = vec(s.model_G .flux)(kn,0*kpi, 0*kpe) + G_neo 
        Qi = vec(s.model_Qi.flux)(0*kn, kpi, 0*kpe) + Qi_neo
        Qe = vec(s.model_Qe.flux)(0*kn, 0*kpi, kpe) + Qe_neo


        # derivatives
        #G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients)(*Lx)
        #Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(*Lx)
        #Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(*Lx)
        G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients) (kn,0*kpi, 0*kpe) 
        Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(0*kn, kpi, 0*kpe)
        Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(0*kn, 0*kpi, kpe)


        # save
        ### Instead of evaluating half here, set the half per force.
        self.Gamma  = pf.Flux_profile(G)
        self.Qi     = pf.Flux_profile(Qi) 
        self.Qe     = pf.Flux_profile(Qe) 
        
        self.G_n    = pf.Flux_profile(G_n   )
        self.G_pi   = pf.Flux_profile(G_pi  )
        self.G_pe   = pf.Flux_profile(G_pe  )
        self.Qi_n   = pf.Flux_profile(Qi_n )
        self.Qi_pi  = pf.Flux_profile(Qi_pi)
        self.Qi_pe  = pf.Flux_profile(Qi_pe)
        self.Qe_n   = pf.Flux_profile(Qe_n )
        self.Qe_pi  = pf.Flux_profile(Qe_pi)
        self.Qe_pe  = pf.Flux_profile(Qe_pe)

    def normalize_fluxes(self):

        # load
        n     = self.density   .midpoints 
        pi    = self.pressure_i.midpoints 
        pe    = self.pressure_e.midpoints 

        Gamma = self.Gamma.profile
        Qi    = self.Qi.profile
        Qe    = self.Qe.profile

        # properly, this should be defined for the flux tubes
        area  = self.area.midpoints
        Ba    = self.Ba

        # calc
        A = area / Ba**2
        Fn = A * Gamma * pi**(1.5) / n**(0.5)
        Fpi = A * Qi * pi**(2.5) / n**(1.5)
        Fpe = A * Qe * pi**(2.5) / n**(1.5)

        Fn  = pf.Flux_profile(Fn )
        Fpi = pf.Flux_profile(Fpi)
        Fpe = pf.Flux_profile(Fpe)

        # save
        self.Fn  = Fn
        self.Fpi = Fpi
        self.Fpe = Fpe

#    def compute_flux_old(self):
#
#        ### calc gradients
#        grad_n  = self.density.grad.profile 
#        grad_pi = self.pressure_i.grad.profile
#        grad_pe = self.pressure_e.grad.profile
#        kn  = - self.density.grad_log.profile     # L_n^inv
#        kpi = - self.pressure_i.grad_log.profile  # L_pi^inv
#        kpe = - self.pressure_e.grad_log.profile  # L_pe^inv
#
#        ### new 3/14
#        # use the positions from flux tubes in between radial grid steps
#        #kn  = - self.density.grad_log   .plus.profile #[:-1]  # computes an extra unphysical point
#        #kpi = - self.pressure_i.grad_log.plus.profile #[:-1] 
#        #kpe = - self.pressure_e.grad_log.plus.profile #[:-1] 
#        ###
#
#        #import pdb
#        #pdb.set_trace()
#
#        # run model (opportunity for parallelization)
#        #Lx = np.array( [Ln_inv, Lpi_inv, Lpe_inv] )
#
#        G_neo  = - self.model_G.neo  * grad_n
#        Qi_neo = - self.model_Qi.neo * grad_pi
#        Qe_neo = - self.model_Qe.neo * grad_pe
#       
#
#        ### Change these function calls to evaluations at the half grid
#        s   = self
#        vec = np.vectorize
#        #G  = vec(s.model_G .flux)(*Lx) + G_neo 
#        #Qi = vec(s.model_Qi.flux)(*Lx) + Qi_neo
#        #Qe = vec(s.model_Qe.flux)(*Lx) + Qe_neo
#        G  = vec(s.model_G .flux)(kn,0*kpi, 0*kpe) + G_neo 
#        Qi = vec(s.model_Qi.flux)(0*kn, kpi, 0*kpe) + Qi_neo
#        Qe = vec(s.model_Qe.flux)(0*kn, 0*kpi, kpe) + Qe_neo
#
#
#        # derivatives
#        #G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients)(*Lx)
#        #Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(*Lx)
#        #Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(*Lx)
#        G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients) (kn,0*kpi, 0*kpe) 
#        Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(0*kn, kpi, 0*kpe)
#        Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(0*kn, 0*kpi, kpe)
#
#
#        # save
#        ### Instead of evaluating half here, set the half per force.
#        #self.Gamma.plus   = profile(G[1: ] )
#        #self.Gamma.minus  = profile(G[:-1] )
#        #self.Qi        = profile(Qi) 
#        #self.Qe        = profile(Qe) 
#        #
#        #self.G_n   = profile(G_n   )
#        #self.G_pi  = profile(G_pi  )
#        #self.G_pe  = profile(G_pe  )
#        #self.Qi_n   = profile(Qi_n )
#        #self.Qi_pi  = profile(Qi_pi)
#        #self.Qi_pe  = profile(Qi_pe)
#        #self.Qe_n   = profile(Qe_n )
#        #self.Qe_pi  = profile(Qe_pi)
#        #self.Qe_pe  = profile(Qe_pe)
#        self.Gamma     = profile(G, half=True)
#        self.Qi        = profile(Qi, half=True) 
#        self.Qe        = profile(Qe, half=True) 
#        
#        self.G_n   = profile(G_n , half=True)
#        self.G_pi  = profile(G_pi, half=True)
#        self.G_pe  = profile(G_pe, half=True)
#        self.Qi_n   = profile(Qi_n , half=True)
#        self.Qi_pi  = profile(Qi_pi, half=True)
#        self.Qi_pe  = profile(Qi_pe, half=True)
#        self.Qe_n   = profile(Qe_n , half=True)
#        self.Qe_pi  = profile(Qe_pi, half=True)
#        self.Qe_pe  = profile(Qe_pe, half=True)
#
#
#    def normalize_fluxes_old(self):
#
#        # load
#        n     = self.density.profile
#        pi    = self.pressure_i.profile
#        pe    = self.pressure_e.profile
#        Gamma = self.Gamma.profile
#        Qi    = self.Qi.profile
#        Qe    = self.Qe.profile
#        area  = self.area.profile
#        Ba    = self.Ba
#
#        # calc
#        A = area / Ba**2
#        Fn = A * Gamma * pi**(1.5) / n**(0.5)
#        Fpi = A * Qi * pi**(2.5) / n**(1.5)
#        Fpe = A * Qe * pi**(2.5) / n**(1.5)
#
#        Fn  = profile(Fn,half=True,grad=True)
#        Fpi = profile(Fpi,half=True,grad=True)
#        Fpe = profile(Fpe,half=True,grad=True)
#        # set inner boundary condition
#        Fn .minus.profile[0] = 0
#        Fpi.minus.profile[0] = 0
#        Fpe.minus.profile[0] = 0
#        # this actually 0 anyways, 
#        #    because F ~ Gamma, which depends on grad n, 
#        #    and grad n is small near the core
#
#        # save
#        self.Fn  = Fn
#        self.Fpi = Fpi
#        self.Fpe = Fpe


    # Compute A and B profiles for density and pressure
    #    this involves finite difference gradients
    def calc_flux_coefficients(self):
        
        # load
        n         = self.density
        pi        = self.pressure_i
        pe        = self.pressure_e
        Fn        = self.Fn
        Fpi       = self.Fpi
        Fpe       = self.Fpe

        # normalization
        norm = 1 / self.a_minor / self.drho  # temp set R=1
        # because it should cancel with a R/L that I am also ignoring
        #norm = (self.R_major / self.a_minor) / self.drho 

        # calculate and save
        s = self
        self.Cn_n  = flux_coefficients(n,  Fn, s.Gamma, s.G_n, norm)
        self.Cn_pi = flux_coefficients(pi, Fn, s.Gamma, s.G_pi, norm) 
        self.Cn_pe = flux_coefficients(pe, Fn, s.Gamma, s.G_pe, norm)

        self.Cpi_n  = flux_coefficients(n,  Fpi, s.Qi, s.Qi_n, norm)
        self.Cpi_pi = flux_coefficients(pi, Fpi, s.Qi, s.Qi_pi, norm) 
        self.Cpi_pe = flux_coefficients(pe, Fpi, s.Qi, s.Qi_pe, norm)
        self.Cpe_n  = flux_coefficients(n,  Fpe, s.Qe, s.Qe_n, norm)
        self.Cpe_pi = flux_coefficients(pi, Fpe, s.Qe, s.Qe_pi, norm) 
        self.Cpe_pe = flux_coefficients(pe, Fpe, s.Qe, s.Qe_pe, norm)
        # maybe these class definitions can be condensed

        # mu coefficients
        # needs kappas, should implement into profile
        # also 0 when G=H=K=0
        self.mu1 = 0
        self.mu2 = 0
        self.mu3 = 0

    def calc_psi_n(self):
    
        # load
        Fnp = self.Fn.plus.profile
        Fnm = self.Fn.minus.profile
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
        g = self.geometry_factor
        psi_nn_plus  = g * (An_pos - Fnp / n_p / 4) 
        psi_nn_minus = g * (An_neg + Fnm / n_m / 4) 
        psi_nn_zero  = g * (Bn + ( Fnm/n_m - Fnp/n_p ) / 4) 
        #psi_nn_zero  = g * (Bn - ( Fnm/n_m - Fnp/n_p ) / 4) 
        # this (-) is a surprise. It disagrees with Barnes thesis (7.64)
                                
        psi_npi_plus  = g * (Ai_pos + 3*Fnp / pi_plus / 4) 
        psi_npi_minus = g * (Ai_neg - 3*Fnm / pi_minus / 4) 
        psi_npi_zero  = g * (Bi - 3./4*( Fnm/pi_minus - Fnp/pi_plus) ) 
    
        psi_npe_plus  = g * Ae_pos
        psi_npe_minus = g * Ae_neg
        psi_npe_zero  = g * Be

        ### add neoclassical term psi_s = -s * D / 2 / drho

        # get D
#        D = self.model_G.neo
#        psi_neo = - D / (2 * drho)
#        psi_nn_plus += psi_neo
#        psi_nn_minus += psi_neo
        # new code

        # save (automatically computes matricies in class function)
        self.psi_nn  = psi_profiles(psi_nn_zero,
                                    psi_nn_plus,
                                    psi_nn_minus, neumann=False)
                        # I don't know if this 'neumann' flag should be here. It doesn't make a big difference.

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
    
        # load
        F_p = self.Fpi.plus#.profile
        F_m = self.Fpi.minus#.profile
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
        g = self.geometry_factor * 2/3 # 2/3 is for pressure
        psi_pin_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1 / n 
        psi_pin_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1 / n
        psi_pin_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) 
        #psi_pin_zero  = g * (Bn -  3/4 * ( F_m/n_m - F_p/n_p ) ) 
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
                                     psi_pipi_minus, neumann=False)
        
        self.psi_pipe = psi_profiles(psi_pipe_zero,
                                     psi_pipe_plus,
                                     psi_pipe_minus)

    def calc_psi_pe(self):
    
        # load
        F_p = self.Fpe.plus.profile
        F_m = self.Fpe.minus.profile
        n     = self.density.profile
        n_p = self.density.plus.profile
        n_m = self.density.minus.profile
        pi_plus  = self.pressure_i.plus.profile
        pi_minus = self.pressure_i.minus.profile
        pe_plus  = self.pressure_e.plus.profile
        pe_minus = self.pressure_e.minus.profile
        #
        An_pos = self.Cpe_n.plus.profile
        An_neg = self.Cpe_n.minus.profile
        Bn     = self.Cpe_n.zero.profile
        Ai_pos = self.Cpe_pi.plus.profile
        Ai_neg = self.Cpe_pi.minus.profile
        Bi     = self.Cpe_pi.zero.profile
        Ae_pos = self.Cpe_pe.plus.profile
        Ae_neg = self.Cpe_pe.minus.profile 
        Be     = self.Cpe_pe.zero.profile 
        #
        mu1 = self.mu1 # should be profiles when implemented
        mu2 = self.mu2 #   now these are all 0
        mu3 = self.mu3
    
        # tri diagonal matrix elements
        g = self.geometry_factor * 2/3 # 2/3 is for pressure
        psi_pen_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1 / n 
        psi_pen_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1 / n
        psi_pen_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) 
        #psi_pin_zero  = g * (Bn -  3/4 * ( F_m/n_m - F_p/n_p ) ) 
        # this (-) is a surprise. It disagrees with Barnes thesis
                                
        psi_pepi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) 
        psi_pepi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) 
        psi_pepi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) ) 
    
        psi_pepe_plus  = g * Ae_pos
        psi_pepe_minus = g * Ae_neg
        psi_pepe_zero  = g * Be

        # save (automatically computes matricies in class function)
        self.psi_pen  = psi_profiles(psi_pen_zero,
                                     psi_pen_plus,
                                     psi_pen_minus)

        self.psi_pepi = psi_profiles(psi_pepi_zero,
                                     psi_pepi_plus,
                                     psi_pepi_minus, neumann=False)
        
        self.psi_pepe = psi_profiles(psi_pepe_zero,
                                     psi_pepe_plus,
                                     psi_pepe_minus)

    def time_step_LHS(self):
 
        # load, dropping last point for Dirchlet fixed boundary condition
        M_nn   = self.psi_nn .matrix[:-1, :-1]         
        M_npi  = self.psi_npi.matrix[:-1, :-1]   
        M_npe  = self.psi_npe.matrix[:-1, :-1]  

        # BUG: according to Barnes (7.115) there should be a factor of 2/3 here
        #         but adding it creates strange behavior (the profiles kink in the 3rd to last point)
        #         while removing it is more regular
        M_pin  = self.psi_pin .matrix[:-1, :-1] # * (2./3) 
        M_pipi = self.psi_pipi.matrix[:-1, :-1] # * (2./3) 
        M_pipe = self.psi_pipe.matrix[:-1, :-1] # * (2./3)      
 
        M_pen  = self.psi_pen .matrix[:-1, :-1] # * (2./3)       
        M_pepi = self.psi_pepi.matrix[:-1, :-1] # * (2./3)       
        M_pepe = self.psi_pepe.matrix[:-1, :-1] # * (2./3)       

        N_block = self.N_radial - 1
        I = np.identity(N_block)
        Z = I*0 # block of 0s
        
        ## build block-diagonal matrices
        M = np.block([
                      [ M_nn , M_npi , M_npe  ], 
                      [ M_pin, M_pipi, M_pipe ],
                      [ M_pen, M_pepi, M_pepe ]
                     ])

        I3 = np.block([[I, Z, Z ],
                       [Z, I, Z ],
                       [Z, Z, I ]])

        Amat = I3 - self.dtau * self.alpha * M
        return Amat
    
    ### Define RHS
    def time_step_RHS(self):
 
        # load
        n_prev  = self.density.profile    [:-1]
        pi_prev = self.pressure_i.profile [:-1]
        pe_prev = self.pressure_e.profile [:-1]
        Fnp     = self.Fn.plus.profile    [:-1]
        Fnm     = self.Fn.minus.profile   [:-1]
        Fip     = self.Fpi.plus.profile   [:-1]
        Fim     = self.Fpi.minus.profile  [:-1]
        Fep     = self.Fpe.plus.profile   [:-1]
        Fem     = self.Fpe.minus.profile  [:-1]
        area    = self.area.profile       [:-1]
        rax     = self.rho_axis           [:-1]
        drho    = self.drho
        alpha   = self.alpha
        dtau    = self.dtau

        # load matrix
        psi_nn  = self.psi_nn.matrix
        psi_npi = self.psi_npi.matrix
        psi_npe = self.psi_npe.matrix
        psi_pin  = self.psi_pin.matrix
        psi_pipi = self.psi_pipi.matrix
        psi_pipe = self.psi_pipe.matrix
        psi_pen  = self.psi_pen.matrix
        psi_pepi = self.psi_pepi.matrix
        psi_pepe = self.psi_pepe.matrix

        # compute forces (for alpha = 0, explicit mode)   
        grho = self.grho
        g = - grho/area
        force_n  = g * (Fnp - Fnm) / drho
        force_pi = g * (Fip - Fim) / drho
        force_pe = g * (Fep - Fem) / drho

        # load source terms
        source_n  = self.source_n[:-1]
        source_pi = self.source_pi[:-1]
        source_pe = self.source_pe[:-1]
    
        ### init boundary condition
        N_radial_mat = self.N_radial - 1
        boundary_n  = np.zeros(N_radial_mat)
        boundary_pi = np.zeros(N_radial_mat)
        boundary_pe = np.zeros(N_radial_mat)
        # get last column of second to last row
        #       there should be  a (-) from flipping the psi
        boundary_n[-1]   =  psi_nn [-2,-1] * self.n_edge   \
                          + psi_npi[-2,-1] * self.pi_edge  \
                          + psi_npe[-2,-1] * self.pe_edge 
        boundary_pi[-1]  =  psi_pin [-2,-1] * self.n_edge  \
                          + psi_pipi[-2,-1] * self.pi_edge \
                          + psi_pipe[-2,-1] * self.pe_edge 
        boundary_pe[-1]  =  psi_pen [-2,-1] * self.n_edge  \
                          + psi_pepi[-2,-1] * self.pi_edge \
                          + psi_pepe[-2,-1] * self.pe_edge 
    
        # should each psi have its own bvec? rename bvec to bvec_n if so
        bvec_n  =  n_prev  + dtau*(1 - alpha)*force_n  + dtau*source_n  + dtau*alpha*boundary_n   ## BUG! this is the source of peaking n-1 point
        #bvec_pi =  pi_prev + dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
        #bvec_pe =  pe_prev + dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe
        bvec_pi =  pi_prev + (2/3) * dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
        bvec_pe =  pe_prev + (2/3) * dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe

       
        # there was a major bug here with the pressure parts of RHS state vector

        bvec3 = np.concatenate( [bvec_n, bvec_pi, bvec_pe] )
        return bvec3

    ### inverts the matrix
    def calc_y_next(self):
        
        # Invert Ax = b
        Amat = self.time_step_LHS()
        bvec = self.time_step_RHS()

        Ainv = np.linalg.inv(Amat) 
        self.y_next = Ainv @ bvec
        
        # for debugging the A matrix
        # plt.figure(); plt.imshow( np.log(np.abs(Amat))); plt.show()
    

    def update(self):

        # load
        y_next  = self.y_next
        n_edge  = self.n_edge
        pi_edge = self.pi_edge
        pe_edge = self.pe_edge

        N_mat = self.N_radial - 1
        n_next, pi_next, pe_next = np.reshape( y_next,(3,N_mat) )

        # check if legit, the forcefully sets the core derivative to 0
        n  = np.concatenate([ [n_next[1]] , n_next[1:] , [n_edge]  ]) 
        pi = np.concatenate([ [pi_next[1]], pi_next[1:], [pi_edge] ]) 
        pe = np.concatenate([ [pe_next[1]], pe_next[1:], [pe_edge] ])

        self.density    = profile(n,  grad=True, half=True, full=True)
        self.pressure_i = profile(pi, grad=True, half=True, full=True)
        self.pressure_e = profile(pe, grad=True, half=True, full=True)

        # step time
        self.time += self.dtau

    def plot_sources(self):

        rax = self.rho_axis
        source_n  = self.source_n 
        source_pi = self.source_pi
        source_pe = self.source_pe

        plt.figure(figsize=(4,4))
        plt.plot(rax, source_n, '.-', label=r'$S_n$')
        plt.plot(rax, source_pi, '.-', label=r'$S_{p_i}$')
        plt.plot(rax, source_pe, '.-', label=r'$S_{p_e}$')
        plt.title('Sources')

        plt.legend()
        plt.grid()

    # can be retired
    # first attempt at exporting gradients for GX
    def write_GX_command(self,j,Time):
        
        # load gradient scale length
        kn  = - self.density.grad_log.profile     # L_n^inv
        kpi = - self.pressure_i.grad_log.profile  # L_pi^inv
        kpe = - self.pressure_e.grad_log.profile  # L_pe^inv

        rax = self.rho_axis
        sax = rax**2
        kti = kpi - kn
        R   = self.R_major

        fout = self.f_cmd
        with open(fout, 'a') as f:

            idx = np.arange(1, self.N_radial-1) # drop the first and last point
            for k in idx: 
                print('{:d}, {:d}, {:.2e}, {:.4e}, {:.4e}, {:.6e}, {:.6e}' \
                .format(j, k, Time, rax[k], sax[k], R*kti[k], R*kn[k]), file=f)
        




# Initialize Trinity profiles
#     with default gradients, half steps, and full steps
def init_profile(x,debug=False):

    x[0] = x[1]
    X = profile(x, grad=True, half=True, full=True)
    return X



# stub for new A,B coefficients that dont use F explicitly
#An_pos = profile( - (R_major/a_minor / drho) \
#                     * T**(3/2) / Ba**2 \   # need to make T.profile
#                     * Gamma.plus.grad.profile )



##### Evolve Trinity Equations

### Define LHS


# 1) should I treat the main equation as the middle of an array
# 2) or append the boundaries as ancillary to the main array?
# the first is more intuitive, but the second may be more efficient
#arg_middle = np.s_[:-1] # the purpose of this expression is to remove "magic numbers" where we drop the last point due to Dirchlet boundary condition



