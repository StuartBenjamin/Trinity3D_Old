import numpy as np
import matplotlib.pyplot as plt

import models as mf 
from netCDF4 import Dataset

import profiles as pf
profile           = pf.Profile
flux_coefficients = pf.Flux_coefficients
psi_profiles      = pf.Psi_profiles

import fusion_lib as fus
import Collisions 

# ignore divide by 0 warnings
#np.seterr(divide='ignore', invalid='ignore')

# This class contains TRINITY calculations and stores partial results as member objects
# There is a sub class for fluxes of each (n, pi, pe) evolution

_use_vmec = True # temp, put this in an input file later

class Trinity_Engine():

    def __init__(self, N = 10, # number of radial points
                       n_core = 4,
                       n_edge = 0.5,
                       Ti_core = 8,
                       Ti_edge = 2,
                       Te_core = 3,
                       Te_edge = .3,
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
                       model      = 'GX',
                       gx_path    = 'gx-files/run-dir/',
                       vmec_path  = './',
                       vmec_wout  = ''
                       ):

        self.N_radial = N           # if this is total points, including core and edge, then GX simulates (N-2) points
        self.n_core   = n_core
        self.n_edge   = n_edge
        self.Ti_core   = Ti_core
        self.Ti_edge   = Ti_edge
        self.Te_core   = Te_core
        self.Te_edge   = Te_edge

        self.pi_edge =  n_edge * Ti_edge
        self.pe_edge =  n_edge * Te_edge
        self.pi_core =  n_core * Ti_core
        self.pe_core =  n_core * Te_core

        self.model    = model

        self.rho_edge = rho_edge
        rho_axis = np.linspace(0,rho_edge,N)         # radial axis, N points
        mid_axis = (rho_axis[1:] + rho_axis[:-1])/2  # centers, (N-1) points
        self.rho_axis = rho_axis
        self.mid_axis = mid_axis
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

        # init normalizations
        self.norms = self.Normalizations(a_minor=a_minor)

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
        n  = (n_core  - n_edge) *(1 - (rho_axis/rho_edge)**2) + n_edge
        Ti = (Ti_core - Ti_edge)*(1 - (rho_axis/rho_edge)**2) + Ti_edge
        Te = (Te_core - Te_edge)*(1 - (rho_axis/rho_edge)**2) + Te_edge
        pi = n * Ti
        pe = n * Te

        # save
        self.density     = init_profile(n)
        self.pressure_i  = init_profile(pi)
        self.pressure_e  = init_profile(pe)

        # init collision model
        svec = Collisions.Collision_Model()
        svec.add_species( n, pi, mass=2, charge=1, ion=True, name='Deuterium')
        svec.add_species( n, pe, mass=1/1800, charge=-1, ion=False, name='electrons')
        self.collision_model = svec


        ### sources
        # temp, Gaussian model. Later this should be adjustable
        Gaussian  = np.vectorize(mf.Gaussian)
        self.aux_source_n  = Gaussian(rho_axis, A=Sn_height , sigma=Sn_width , x0=Sn_center)
        self.aux_source_pi = Gaussian(rho_axis, A=Spi_height, sigma=Spi_width, x0=Spi_center)
        self.aux_source_pe = Gaussian(rho_axis, A=Spe_height, sigma=Spe_width, x0=Spe_center)


        ### init flux models
        if (model == 'GX'):
            fout = 'gx-files/temp.gx'
            self.path = gx_path
            gx = mf.GX_Flux_Model(fout, 
                                  path = gx_path, 
                                  vmec_path = vmec_path,
                                  vmec_wout = vmec_wout,
                                  midpoints = mid_axis
                                  )
            self.f_cmd = fout
            self.vmec_wout = vmec_wout

            # read VMEC
            self.read_VMEC( vmec_wout, path=vmec_path, use_vmec=_use_vmec )

            gx.init_geometry()
            self.model_gx = gx
    

        elif (model == 'diffusive'):
            bm = mf.Barnes_Model2()
            self.barnes_model = bm

        else:
            zero_flux = False
            self.model_G  = mf.Flux_model(zero_flux=zero_flux)
            self.model_Qi = mf.Flux_model(zero_flux=zero_flux)
            self.model_Qe = mf.Flux_model(zero_flux=zero_flux)

    def read_VMEC(self, wout, path='gx-geometry/', use_vmec=False):

        self.vmec_wout = wout

        if wout == '':
            print('  Trinity Lib: no vmec file given, using default flux tubes for GX')
            return

        vmec = Dataset( path+wout, mode='r')

        if use_vmec:
            self.R_major = vmec.variables['Rmajor_p'][:]
            self.a_minor = vmec.variables['Aminor_p'][:]
            self.Ba      = vmec.variables['volavgB'][:]

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

    def calc_collisions(self):
        # this function computes the E terms (Barnes 7.73)
        # there is one for each species.

        # update profiles in collision lib
        cmod = self.collision_model
        cmod.update_profiles(self)
        cmod.compute_nu_ei()

        # convert units
        nu_ei      = cmod.nu_ei
        t_ref      = self.norms.t_ref
        gyro_scale = self.norms.gyro_scale
        nu_norm    = nu_ei * t_ref * gyro_scale**2

        # compute E 
        pi = self.pressure_i.profile
        pe = self.pressure_e.profile
        ni = self.density.profile
        ne = self.density.profile

        Ei = nu_norm * pi * ( (pe/ne)/(pi/ni) - 1 )
        Ee = nu_norm * pe * ( (pi/ni)/(pe/ne) - 1 )

        # save
        self.nu_ei = nu_ei
        self.nu_ei_norm = nu_norm
        self.Ei = Ei
        self.Ee = Ee
        

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
        # new code (this is actually not the right equation, or the right place, I will delete later)

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
        E  = self.Ei

        n        = self.density.profile
        n_p      = self.density.plus.profile
        n_m      = self.density.minus.profile
        pi       = self.pressure_i.profile
        pi_plus  = self.pressure_i.plus.profile
        pi_minus = self.pressure_i.minus.profile
        pe       = self.pressure_e.profile
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

        Zi, mi = self.collision_model.export_species(0) # hard coded index
        Ze, me = self.collision_model.export_species(1) 

        E_pi = - E * ( Zi / (Ze*pe - Zi*pi) + (3./2) * me*Zi / (mi*Ze*pe + me*Zi*pi) )
        E_pe =   E * ( Ze / (Ze*pe - Zi*pi) - (3./2) * mi*Ze / (mi*Ze*pe + me*Zi*pi) )

        # tri diagonal matrix elements
        g = self.geometry_factor * 2/3 # 2/3 is for pressure
        psi_pin_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1 / n 
        psi_pin_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1 / n 
        psi_pin_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) \
                          + (5./2) * (E/n)
                                
        psi_pipi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) 
        psi_pipi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) 
        psi_pipi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) ) + E_pi
    
        psi_pipe_plus  = g * Ae_pos
        psi_pipe_minus = g * Ae_neg
        psi_pipe_zero  = g * Be + E_pe

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
        E   = self.Ee

        n        = self.density.profile
        n_p      = self.density.plus.profile
        n_m      = self.density.minus.profile
        pi       = self.pressure_i.profile
        pi_plus  = self.pressure_i.plus.profile
        pi_minus = self.pressure_i.minus.profile
        pe       = self.pressure_e.profile
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

        Zi, mi = self.collision_model.export_species(0) # hard coded index
        Ze, me = self.collision_model.export_species(1) 

        E_pi =   E * ( Zi / (Zi*pi - Ze*pe) - (3./2) * me*Zi / (me*Zi*pi + mi*Ze*pe) )
        E_pe = - E * ( Zi / (Zi*pi - Ze*pe) + (3./2) * mi*Ze / (me*Zi*pi + mi*Ze*pe) )
    
        # tri diagonal matrix elements
        g = self.geometry_factor * 2/3 # 2/3 is for pressure
        psi_pen_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1 / n 
        psi_pen_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1 / n
        psi_pen_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) \
                          + (5./2) * (E/n)
                                
        psi_pepi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) 
        psi_pepi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) 
        psi_pepi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) )  + E_pi
    
        psi_pepe_plus  = g * Ae_pos
        psi_pepe_minus = g * Ae_neg
        psi_pepe_zero  = g * Be + E_pe

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


    # use auxiliary sources, add fusion power, subtract Bremstrahlung
    def calc_sources(self, alpha_heating=True,
                           brems_radiation=True,
                    ):
    
        # load axuiliary source terms
        aux_source_n  = self.aux_source_n #[:-1]
        aux_source_pi = self.aux_source_pi#[:-1]
        aux_source_pe = self.aux_source_pe#[:-1]
        
        # load profiles
        n_profile_m3 = self.density.profile * 1e20
        Ti_profile_keV = self.pressure_i.profile / self.density.profile 
        Te_profile_keV = self.pressure_e.profile / self.density.profile 


        # compute fusion power
        if (alpha_heating):
            Ti_profile_eV = Ti_profile_keV * 1e3
            P_fusion_Wm3, fusion_rate  \
                    = fus.alpha_heating_DT( n_profile_m3, Ti_profile_eV )
        else:
            P_fusion_Wm3 = 0 * n_profile_m3
            fusion_rate = 0 * n_profile_m3

        # compute bremstrahlung radiation
        if (brems_radiation):
            P_brems_Wm3 = fus.radiation_bremstrahlung(n_profile_m3/1e20, Te_profile_keV) 
        else:
            P_brems_Wm3 = 0 * n_profile_m3

        # non-dimensionalize 
        pressure_source_scale = self.norms.pressure_source_scale # converts from SI (W/m3)
        P_fusion = P_fusion_Wm3 * pressure_source_scale
        P_brems  = P_brems_Wm3  * pressure_source_scale

        # store
        self.P_fusion = P_fusion
        self.P_brems  = P_brems
        self.P_fusion_Wm3 = P_fusion_Wm3
        self.P_brems_Wm3  = P_brems_Wm3
        self.fusion_rate  = fusion_rate

        self.source_n  = aux_source_n
        self.source_pi = aux_source_pi + P_fusion
        self.source_pe = aux_source_pe - P_brems



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
        Ei      = self.Ei                 [:-1]
        Ee      = self.Ee                 [:-1]
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
        source_n  = self.source_n[:-1]   # later change this to be aux + fusion - brems
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
        bvec_n  =  n_prev  \
                     + dtau*(1 - alpha)*force_n  \
                     + dtau*source_n  + dtau*alpha*boundary_n   
        bvec_pi =  pi_prev \
                     + (2/3) * dtau*(1 - alpha) * (force_pi + Ei) \
                     + dtau*source_pi + dtau*alpha*boundary_pi
        bvec_pe =  pe_prev \
                     + (2/3) * dtau*(1 - alpha) * (force_pe + Ee) \
                     + dtau*source_pe + dtau*alpha*boundary_pe

       
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


    # a subclass for handling normalizations in Trinity
    class Normalizations():
        def __init__(self, n_ref = 1e20,     # m3
                           T_ref = 1e3,      # eV
                           B_ref = 1,        # T
                           m_ref = 1.67e-27, # kg, proton mass
                           a_minor = 1,      # m
                          ):

            self.e = 1.602e-19  # Colulomb
            self.c = 2.99e8     # m/s


            # this is a reference length scale
            #   it is the (v_T,ref / Omega_ref) the distance thermal particle travels in cyclotron time
            #   v_T = sqrt(2T/m_ref)
            #   Omega_ref = e B_ref / m_ref c
            #   m_ref is the proton mass.
            self.rho_ref = 4.57e-3 # m

            vT_ref = np.sqrt(2 * (T_ref*self.e) / m_ref)
       

            # this block current lives in calc_sources()
            #   it could simplify code to do it here
            #   but how to get a_minor out of the parent class?
            t_ref = a_minor / vT_ref
            p_ref = n_ref * T_ref * self.e
            gyro_scale = a_minor / self.rho_ref
            
            # converts from SI (W/m3)
            pressure_source_scale = t_ref / p_ref * gyro_scale**2 

            ### save
            self.n_ref = n_ref
            self.T_ref = T_ref
            self.B_ref = B_ref
            self.a_ref = a_minor # unlike the above, this is device specific rather than a code convention

            self.vT_ref     = vT_ref
            self.t_ref      = t_ref
            self.p_ref      = p_ref
            self.gyro_scale = gyro_scale
            self.pressure_source_scale = pressure_source_scale




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



