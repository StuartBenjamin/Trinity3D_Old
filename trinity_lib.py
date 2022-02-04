import numpy as np
import matplotlib.pyplot as plt
import pdb

import models as mf 

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
                       Sn_width   = 0.1,   
                       Sn_height  = 0,  
                       Spi_width  = 0.1, 
                       Spi_height = 0, 
                       Spe_width  = 0.1,  
                       Spe_height = 0 
                       ):

        self.N_radial = N           # if this is total points, including core and edge, then GX simulates (N-2) points
        self.n_core   = n_core
        self.n_edge   = n_edge
        self.pi_core   = pi_core
        self.pi_edge   = pi_edge
        self.pe_core   = pe_core
        self.pe_edge   = pe_edge
        #self.drho     = 1/N # for now assume equal spacing, 
                            #    could be computed in general
        self.rho_edge = rho_edge
        self.drho     = rho_edge / (N-1)
        rho_axis = np.linspace(0,rho_edge,N) # radial axis
        self.rho_axis = rho_axis

        self.dtau     = dtau
        self.alpha    = alpha

        ### will be from VMEC
        self.Ba      = Ba # average field on LCFS
        self.R_major = R_major # meter
        self.a_minor = a_minor # meter
        self.area     = profile(np.linspace(0.01,a_minor,N)) # parabolic area, simple torus


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
        self.source_n  = Gaussian(rax, A=Sn_height , sigma=Sn_width)
        self.source_pi = Gaussian(rax, A=Spi_height, sigma=Spi_width)
        self.source_pe = Gaussian(rax, A=Spe_height, sigma=Spe_width)

        ### init flux models
        self.model_G  = mf.Flux_model()
        self.model_Qi = mf.Flux_model()
        self.model_Qe = mf.Flux_model()

        ### init GX commands
        fout = 'gx-files/temp.gx'
        gx = mf.GX_Flux_Model(fout)
        gx.init_geometry()

        pdb.set_trace()

        self.f_cmd = fout
        self.model_gx = gx

    # this is a toy model of Flux based on ReLU + neoclassical
    #     to be replaced by GX or STELLA import module
    def compute_flux(self):

        ### calc gradients
        grad_n  = self.density.grad.profile 
        grad_pi = self.pressure_i.grad.profile
        grad_pe = self.pressure_e.grad.profile
        kn  = - self.density.grad_log.profile     # L_n^inv
        kpi = - self.pressure_i.grad_log.profile  # L_pi^inv
        kpe = - self.pressure_e.grad_log.profile  # L_pe^inv

        # run model (opportunity for parallelization)
        #Lx = np.array( [Ln_inv, Lpi_inv, Lpe_inv] )

        G_neo  = - self.model_G.neo  * grad_n
        Qi_neo = - self.model_Qi.neo * grad_pi
        Qe_neo = - self.model_Qe.neo * grad_pe
        
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
        A = area / Ba**2
        Fn = A * Gamma * pi**(1.5) / n**(0.5)
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
    
        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1 
        drho  = self.drho
        area  = self.area.profile
        geometry_factor = - grho / (area * drho)
    
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
        g = geometry_factor
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
    
        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1 
        drho  = self.drho
        area  = self.area.profile
        geometry_factor = - grho / (area * drho)
    
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
        g = geometry_factor
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
    
        # need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1 
        drho  = self.drho
        area  = self.area.profile
        geometry_factor = - grho / (area * drho)
    
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
        g = geometry_factor
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
        M_nn  = self.psi_nn .matrix[:-1, :-1]         
        M_npi = self.psi_npi.matrix[:-1, :-1]         
        M_npe = self.psi_npe.matrix[:-1, :-1]         

        M_pin  = self.psi_pin .matrix[:-1, :-1]         
        M_pipi = self.psi_pipi.matrix[:-1, :-1]         
        M_pipe = self.psi_pipe.matrix[:-1, :-1]         
 
        M_pen  = self.psi_pen .matrix[:-1, :-1]         
        M_pepi = self.psi_pepi.matrix[:-1, :-1]         
        M_pepe = self.psi_pepe.matrix[:-1, :-1]         

        N_block = self.N_radial - 1
        I = np.identity(N_block)
        Z = I*0 # block of 0s
        
        ## build block-diagonal matrices
        #M = np.block([
        #              [ M_nn , Z      , Z ],
        #              [ Z    , M_pipi , Z ],
        #              [ Z    , Z      , M_pepe ],
        #            ])
        M = np.block([
                      [ M_nn , M_npi , M_npe  ], # this should have factor 2/3
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
   
        grho = 1 # temp
        g = - grho/area
        force_n  = g * (Fnp - Fnm) / drho
        force_pi = g * (Fip - Fim) / drho
        force_pe = g * (Fep - Fem) / drho

        #Gaussian  = np.vectorize(mf.Gaussian)
        #source_n  = Gaussian(rax, A=Sn_height , sigma=Sn_width)
        #source_pi = Gaussian(rax, A=Spi_height, sigma=Spi_width)
        #source_pe = Gaussian(rax, A=Spe_height, sigma=Spe_width)
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
        bvec_pi =  pi_prev + dtau*(1 - alpha)*force_pi + dtau*source_pi + dtau*alpha*boundary_pi
        bvec_pe =  pe_prev + dtau*(1 - alpha)*force_pe + dtau*source_pe + dtau*alpha*boundary_pe
       
        # there was a major bug here with the pressure parts of RHS state vector

        #bvec3 = np.concatenate( [bvec_n, 0*bvec_pi, 0*bvec_pe] )
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
        n  = np.concatenate([ [n_next[1]] , n_next[1:] , [n_edge] ]) 
        pi = np.concatenate([ [pi_next[1]], pi_next[1:], [pi_edge] ]) 
        pe = np.concatenate([ [pe_next[1]], pe_next[1:], [pe_edge] ])

        self.density    = profile(n,  grad=True, half=True, full=True)
        self.pressure_i = profile(pi, grad=True, half=True, full=True)
        self.pressure_e = profile(pe, grad=True, half=True, full=True)

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
        

# the class computes and stores normalized flux F, AB coefficients, and psi for the tridiagonal matrix
# it will need a flux Q, and profiles nT
# it should know whether ions or electrons are being computed, or density...
class flux_coefficients():

    # x is state vector (n, pi, pe)
    # Y is normalized flux (F,I)
    # Z is dlog flux (d log Gamma / d L_x ), evaluated at +- half step
    def __init__(self,x,Y,Z,dZ,norm):

        self.state   = x
        self.flux    = Y # this is normalized flux F,I
        self.RawFlux = Z # this is Gamma,Q
        self.dRawFlux = dZ # this is Gamma,Q
        self.norm    = norm # normalizlation constant (R/a)/drho

        # plus,minus,zero : these are the A,B coefficients
        self.plus  = self.C_plus()
        self.minus = self.C_minus()
        self.zero  = self.C_zero()


    def C_plus(self):

        norm = self.norm

        x  = self.state.profile
        xp = self.state.plus.profile
        Yp = self.flux.plus.profile
        Zp = self.RawFlux.plus.profile
        dZp = self.dRawFlux.plus.profile

        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZp = np.nan_to_num( dZp / Zp )

        Cp = - norm * (x / xp**2) * Yp * dLogZp
        return profile(Cp)

    def C_minus(self):

        norm = self.norm
        
        x  = self.state.profile
        xm = self.state.minus.profile
        Ym = self.flux.minus.profile
        Zm = self.RawFlux.minus.profile
        dZm = self.dRawFlux.minus.profile
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZm = np.nan_to_num( dZm / Zm )

        Cm = - norm * (x / xm**2) * Ym * dLogZm
        return profile(Cm)

    def C_zero(self):

        norm = self.norm

        x  = self.state.profile
        xp = self.state.plus.profile
        xm = self.state.minus.profile
        xp1 = self.state.plus1.profile
        xm1 = self.state.minus1.profile
        
        Yp = self.flux.plus.profile
        Zp = self.RawFlux.plus.profile
        dZp = self.dRawFlux.plus.profile

        Ym = self.flux.minus.profile
        Zm = self.RawFlux.minus.profile
        dZm = self.dRawFlux.minus.profile
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZp = np.nan_to_num( dZp / Zp )
            dLogZm = np.nan_to_num( dZm / Zm )

        cp = xp1 / xp**2 * Yp * dLogZp
        cm = xm1 / xm**2 * Ym * dLogZm
        Cz = norm * ( cp + cm ) 
        return profile(Cz)


# This class organizes the psi-profiles in tri-diagonal matrix
class psi_profiles():

    def __init__(self,psi_zero,
                      psi_plus,
                      psi_minus,
                      neumann=False):

        # save profiles
        self.plus  = profile( psi_plus )
        self.minus = profile( psi_minus )
        self.zero  = profile( psi_zero )

        # formulate matrix
        M = tri_diagonal(psi_zero,
                         psi_plus,
                         psi_minus)

        if (neumann):
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
        deriv[0]  = 0
        #deriv[0]  = deriv[1]      # should a one-sided stencil be used here too?
                                  # should I set it to 0? in a transport solver, is the 0th point on axis?
                                  # I don't think GX will be run for the 0th point. So should that point be excluded from TRINITY altogether?
                                  #      or should it be included as a ghost point?

        # this is a second order accurate one-sided stencil
        deriv[-1]  = ( 3*xj[-1] -4*xj[-2] + xj[-3])  / (2*dx)

        # Bill, from fortran trinity
        deriv[-1]  = ( 23*xj[-1] -21*xj[-2] - 3*xj[-3] + xj[-4])  / (24*dx)

        return deriv
        # can recursively make gradient also a profile class
        # need to test this

    def log_gradient(self):
        # this is actually the gradient of the log...

        with np.errstate(divide='ignore', invalid='ignore'):
            gradlog = np.nan_to_num(self.gradient() / self.profile )

        return gradlog

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



