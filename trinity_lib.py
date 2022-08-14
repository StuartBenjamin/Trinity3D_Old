import numpy as np
import matplotlib.pyplot as plt

import models as mf 
from netCDF4 import Dataset

import profiles as pf # needed for setting pf.rho_axis
from profiles import Profile, Flux_profile, Flux_coefficients, Psi_profiles, init_profile


import fusion_lib as fus
import Collisions 
from Trinity_io import Trinity_Input


'''
This class contains the bulk of TRINITY calculations.
It stores partial calculations 
+ normalized fluxes
+ C-coefficients
+ the Tridiagnoal matrix 
as attributes. It also contains a subclass Normalizations that handles all normalizations.
'''

class Trinity_Engine():

    ### read inputs
    def load(self,x,string):
        # this is my toml find or
        tr3d = self.inputs
        try:
            return eval(string)
        except:
            return x

    def __init__(self, trinity_input,
                       N_radial = 10, # number of radial points
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
                       ext_source_file = '',
                       model      = 'GX',
                       D_neo      = 0.5,
                       no_collisions = False,
                       alpha_heating = False,
                       bremstrahlung = False,
                       gx_path    = 'gx-files/run-dir/',
                       vmec_path  = './',
                       vmec_wout  = '',
                       ):

        ### Loading Trinity Inputs
        '''
        need to sort out the order
        loads defaults first,
        then overwrite with input file as needed.
        '''

        tr3d = Trinity_Input(trinity_input)
        self.trinity_input_file = trinity_input
        self.inputs = tr3d
        

        N_radial = self.load( N_radial, "int(tr3d.inputs['grid']['N_radial'])" )
        rho_edge = float ( tr3d.inputs['grid']['rho_edge'] )
        dtau     = float ( tr3d.inputs['grid']['dtau'    ] )
        alpha    = float ( tr3d.inputs['grid']['alpha'   ] )
        N_steps  = int   ( tr3d.inputs['grid']['N_steps' ] )
        
        model    = tr3d.inputs['model']['model']
        D_neo    = float ( tr3d.inputs['model']['D_neo'] )
        
        n_core  = float ( tr3d.inputs['profiles']['n_core' ] )
        n_edge  = float ( tr3d.inputs['profiles']['n_edge' ] )
        Ti_core = float ( tr3d.inputs['profiles']['Ti_core'] )
        Ti_edge = float ( tr3d.inputs['profiles']['Ti_edge'] )
        Te_core = float ( tr3d.inputs['profiles']['Te_core'] )
        Te_edge = float ( tr3d.inputs['profiles']['Te_edge'] )
        
        Sn_height  = float ( tr3d.inputs['sources']['Sn_height' ] ) 
        Spi_height = float ( tr3d.inputs['sources']['Spi_height'] ) 
        Spe_height = float ( tr3d.inputs['sources']['Spe_height'] ) 
        Sn_width   = float ( tr3d.inputs['sources']['Sn_width'  ] ) 
        Spi_width  = float ( tr3d.inputs['sources']['Spi_width' ] ) 
        Spe_width  = float ( tr3d.inputs['sources']['Spe_width' ] ) 
        Sn_center  = float ( tr3d.inputs['sources']['Sn_center' ] ) 
        Spi_center = float ( tr3d.inputs['sources']['Spi_center'] ) 
        Spe_center = float ( tr3d.inputs['sources']['Spe_center'] ) 
        
        ext_source_file = tr3d.inputs['sources']['ext_source_file'] 
        # boolean as string
        no_collisions = self.load( no_collisions, "tr3d.inputs['debug']['no_collisions']" )
        alpha_heating = self.load( alpha_heating, "tr3d.inputs['debug']['alpha_heating']" )
        bremstrahlung = self.load( bremstrahlung, "tr3d.inputs['debug']['bremstrahlung']" )
       
        gx_path   = tr3d.inputs['path']['gx_path']
        vmec_path = tr3d.inputs['path']['vmec_path']
        vmec_wout = self.load( vmec_wout, "tr3d.inputs['geometry']['vmec_wout']")

        R_major   = float ( tr3d.inputs['geometry']['R_major'] ) 
        a_minor   = float ( tr3d.inputs['geometry']['a_minor'] ) 
        Ba        = float ( tr3d.inputs['geometry']['Ba'     ] ) 

        N_prints = int ( tr3d.inputs['log']['N_prints'] )
        f_save   = tr3d.inputs['log']['f_save']

        ### Finished Loading Trinity Inputs


        self.N_radial = N_radial         # if this is total points, including core and edge, then GX simulates (N-2) points
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

        self.no_collisions = no_collisions
        self.alpha_heating = alpha_heating
        self.bremstrahlung = bremstrahlung

        rho_inner = rho_edge / (2*N_radial - 1)
        rho_axis = np.linspace(rho_inner, rho_edge, N_radial) # radial axis, N points
        mid_axis = (rho_axis[1:] + rho_axis[:-1])/2  # centers, (N-1) points
        self.rho_axis = rho_axis
        self.mid_axis = mid_axis

        # set axis for Profiles library
        pf.rho_axis   = rho_axis
        pf.mid_axis   = mid_axis

        self.dtau     = dtau
        self.alpha    = alpha
        self.N_steps  = N_steps
        self.N_prints = N_prints

        self.rho_edge = rho_edge
        self.rho_inner = rho_inner

        self.f_save = f_save

        ### will be from VMEC
        self.Ba      = Ba # average field on LCFS
        self.R_major = R_major # meter
        self.a_minor = a_minor # meter

        # local variables
        self.time = 0
        self.t_idx = 0
        self.gx_idx = 0
        self.needs_new_flux = True
        self.needs_new_vmec = False


        # init normalizations
        self.norms = self.Normalizations(a_minor=a_minor)

        # TODO: need to implement <|grad rho|>, by reading surface area from VMEC
        grho = 1
        drho       = (rho_edge - rho_inner) / (N_radial - 1) # always const?
        area       = Profile(np.linspace(0.01,a_minor,N_radial), half=True) # parabolic area, simple torus
        # (bug) this looks problematic. The area model should follow the rho_axis, or it should come from VMEC
        self.grho  = grho
        self.drho  = drho
        self.area  = area
        self.geometry_factor = - grho / (drho * area.profile)

        ### init profiles
        #     temporary profiles, later init from VMEC
        n  = (n_core  - n_edge )*(1 - (rho_axis/rho_edge)**2) + n_edge
        Ti = (Ti_core - Ti_edge)*(1 - (rho_axis/rho_edge)**2) + Ti_edge
        Te = (Te_core - Te_edge)*(1 - (rho_axis/rho_edge)**2) + Te_edge
        pi = n * Ti
        pe = n * Te


        self.vmec_pressure_old = (pi + pe) * 1e20 * (1e3 * 1.6e-19) # for comparison later

        # save
        self.density     = init_profile(n)
        self.pressure_i  = init_profile(pi)
        self.pressure_e  = init_profile(pe)

        # init collision model
        svec = Collisions.Collision_Model()
        svec.add_species( n, pi, mass_p=2, charge_p=1, ion=True, name='Deuterium')
        svec.add_species( n, pe, mass_p=1/1800, charge_p=-1, ion=False, name='electrons')
        self.collision_model = svec

        ### init flux models
        if (model == "GX"):
            print("  flux model: GX")
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
            self.read_VMEC( vmec_wout, path=vmec_path )

            gx.init_geometry()
            self.model_gx = gx
    

        elif (model == "diffusive"):
            bm = mf.Barnes_Model2()
            self.barnes_model = bm

        elif (model == "ReLU-particle-only"):
            print("  flux model: ReLU-particle-only")
            self.model_G  = mf.Flux_model(D_neo=D_neo, zero_flux=False)
            self.model_Qi = mf.Flux_model(D_neo=D_neo, zero_flux=True)
            self.model_Qe = mf.Flux_model(D_neo=D_neo, zero_flux=True)

        elif (model == "zero-flux"):
            print("  flux model: zero-flux")
            self.model_G  = mf.Flux_model(D_neo=D_neo, zero_flux=True)
            self.model_Qi = mf.Flux_model(D_neo=D_neo, zero_flux=True)
            self.model_Qe = mf.Flux_model(D_neo=D_neo, zero_flux=True)

        else: # "ReLU"
            print("  flux model: ReLU (default)")
            self.model_G  = mf.Flux_model(D_neo=D_neo)
            self.model_Qi = mf.Flux_model(D_neo=D_neo)
            self.model_Qe = mf.Flux_model(D_neo=D_neo)

        # load sources (to do: split this into separate function self.load_source())
        if (ext_source_file == 'none'):

            self.Sn_height  = Sn_height  
            self.Spi_height = Spi_height 
            self.Spe_height = Spe_height 
            self.Sn_width   = Sn_width      
            self.Spi_width  = Spi_width   
            self.Spe_width  = Spe_width    
            self.Sn_center  = Sn_center   
            self.Spi_center = Spi_center 
            self.Spe_center = Spe_center  

            ### sources
            # temp, Gaussian model. Later this should be adjustable
            Gaussian  = np.vectorize(mf.Gaussian)
            self.aux_source_n  = Gaussian(rho_axis, A=Sn_height , sigma=Sn_width , x0=Sn_center)
            self.aux_source_pi = Gaussian(rho_axis, A=Spi_height, sigma=Spi_width, x0=Spi_center)
            self.aux_source_pe = Gaussian(rho_axis, A=Spe_height, sigma=Spe_width, x0=Spe_center)
            
            self.source_model = 'Gaussian'

        else:

            # this option reads an external source file
            with open(ext_source_file) as f_source:
                datain = f_source.readlines()
            print("\n  Reading external source file:", ext_source_file, "\n")

            data = np.array( [line.strip().split(',') for line in datain[1:]], float)
            rax_source, S_Qi, S_Qe, = data.T

            raxis = self.rho_axis
            Spi = np.interp( raxis, rax_source, S_Qi)
            Spe = np.interp( raxis, rax_source, S_Qe)
            Sn  = 0 * raxis # temp, particle transport turned off

            # (TODO) need to normalize
            particle_norm = self.norms.particle_source_scale
            pressure_norm = self.norms.pressure_source_scale

            self.aux_source_n  = Sn  * particle_norm
            self.aux_source_pi = Spi * pressure_norm
            self.aux_source_pe = Spe * pressure_norm

            self.source_model = 'external'
            self.ext_source_file = ext_source_file
        # end source function


        # Print Global Geometry information
        print("  Global Geometry Information")
        print(f"    R_major: {self.R_major:.2f} m")
        print(f"    a_minor: {self.a_minor:.2f} m")
        print(f"    Ba     : {self.Ba:.2f} T averge on LCFS \n")

    ##### End of __init__ function


    def read_VMEC(self, wout, path='gx-geometry/'):

        self.vmec_wout = wout

        if wout == '':
            print('  Trinity Lib: no vmec file given, using default flux tubes for GX')
            return

        # load global geometry from VMEC
        vmec = Dataset( path+wout, mode='r')
        self.R_major = vmec.variables['Rmajor_p'][:]
        self.a_minor = vmec.variables['Aminor_p'][:]
        self.Ba      = vmec.variables['volavgB'][:]

        self.vmec_data = vmec # data heavy?


    def get_flux(self):

        model = self.model

        if   (model == "GX"):

            if self.needs_new_flux:
                # calculates fluxes from GX
                self.model_gx.prep_commands(self, self.gx_idx) 
                self.gx_idx += 1
#            else:
#                print(" ENGAGE TRINITY SUBCYCLE ", f"t = {self.t_idx}")

        elif (model == "diffusive"):
            # test from MAB thesis (documented in models.py)
            self.barnes_model.compute_Q(self)

        else:
            # default, run a ReLU model
            self.compute_relu_flux() 

    def compute_relu_flux(self):
        #     move this to models.py?
        '''
        This is a toy model of Flux based on ReLU + neoclassical.
        It is interchangeable with gyrokinetic GX or STELLA models.
        '''

        ### calc gradients
        grad_n  = self.density   .grad.profile
        grad_pi = self.pressure_i.grad.profile
        grad_pe = self.pressure_e.grad.profile

        # use the positions from flux tubes in between radial grid steps
        kn  = - self.density.grad_log   .profile 
        kpi = - self.pressure_i.grad_log.profile
        kpe = - self.pressure_e.grad_log.profile

        # run model (opportunity for parallelization)
        G_neo  = - self.model_G.neo  * grad_n
        Qi_neo = - self.model_Qi.neo * grad_pi
        Qe_neo = - self.model_Qe.neo * grad_pe

        ### Change these function calls to evaluations at the half grid
        s   = self
        vec = np.vectorize
        G  = vec(s.model_G .flux)(kn, 0*kpi, 0*kpe) + G_neo 
        Qi = vec(s.model_Qi.flux)(0*kn, kpi-kn, 0*kpe) + Qi_neo
        Qe = vec(s.model_Qe.flux)(0*kn, 0*kpi, kpe-kn) + Qe_neo


        ### off diagonal is turned off
        G_n, G_pi, G_pe    = vec(s.model_G.flux_gradients )(kn,0*kpi, 0*kpe) 
        Qi_n, Qi_pi, Qi_pe = vec(s.model_Qi.flux_gradients)(0*kn, kpi-kn, 0*kpe)
        Qe_n, Qe_pi, Qe_pe = vec(s.model_Qi.flux_gradients)(0*kn, 0*kpi, kpe-kn)


        # save
        ### Instead of evaluating half here, set the half per force.
        self.Gamma  = Flux_profile(G)
        self.Qi     = Flux_profile(Qi) 
        self.Qe     = Flux_profile(Qe) 
        
        self.G_n    = Flux_profile(G_n   )
        self.G_pi   = Flux_profile(G_pi  )
        self.G_pe   = Flux_profile(G_pe  )
        self.Qi_n   = Flux_profile(Qi_n )
        self.Qi_pi  = Flux_profile(Qi_pi)
        self.Qi_pe  = Flux_profile(Qi_pe)
        self.Qe_n   = Flux_profile(Qe_n )
        self.Qe_pi  = Flux_profile(Qe_pi)
        self.Qe_pe  = Flux_profile(Qe_pe)

    def normalize_fluxes(self):
        '''
        Using (Gamma, Q) compute (F,G,H) from Eq 7.45, 7.74-76 in Michael's thesis.
        '''

        # load
        n     = self.density   .midpoints 
        pi    = self.pressure_i.midpoints 
        pe    = self.pressure_e.midpoints 

        Gamma = self.Gamma.profile
        Qi    = self.Qi.profile
        Qe    = self.Qe.profile


        area  = self.area.midpoints # this should be defined properly for fluxtubes
        Ba    = self.Ba
        grho  = self.grho
        a     = self.a_minor

        aLn  = - a * self.density.grad_log   .profile  # L_n^inv
        aLpi = - a * self.pressure_i.grad_log.profile  # L_pi^inv
        aLpe = - a * self.pressure_e.grad_log.profile  # L_pe^inv

        # calc
        A = area / Ba**2
        Fn = A * Gamma * pi**(1.5) / n**(0.5)
        Fpi = A * Qi * pi**(2.5) / n**(1.5)
        Fpe = A * Qe * pi**(2.5) / n**(1.5)

        # new 8/11
        B_factor = grho / Ba**2 
        Impurity_ratio = 1 # Zs/Zi
        # kappa1 is (7.78) kappa2 is (7.79)
        kappa1_i = 1.5 * aLpi - 2.5 * aLn
        kappa1_e = 1.5 * aLpe - 2.5 * aLn
        kappa2_i = aLn - aLpi
        kappa2_e = aLn - aLpe

        Gi = B_factor * Impurity_ratio * pi**1.5 * pi / n**1.5 * kappa1_i * Gamma
        Ge = B_factor * Impurity_ratio * pi**1.5 * pe / n**1.5 * kappa1_e * Gamma
        Hi = B_factor * pi**2.5 / n**1.5 * kappa2_i * Qi
        He = B_factor * pi**2.5 / n**1.5 * kappa2_e * Qe

        # save
        self.Fn   = Flux_profile( Fn )
        self.Fpi  = Flux_profile( Fpi )
        self.Fpe  = Flux_profile( Fpe )

        self.kappa1_i = Flux_profile( kappa1_i )
        self.kappa1_e = Flux_profile( kappa1_e )
        self.kappa2_i = Flux_profile( kappa2_i )
        self.kappa2_e = Flux_profile( kappa2_e )

        self.Gi  = Flux_profile( Gi )
        self.Ge  = Flux_profile( Ge )
        self.Hi  = Flux_profile( Hi )
        self.He  = Flux_profile( He )

# for debugging
#        self.Gamma.plot(title='Gamma',show=False)
#        self.Qi.plot(title='Qi',show=False)
#        self.Qe.plot(title='Qe',show=False)
##        self.Gi.plot(title='Gi',show=False)
##        self.Ge.plot(title='Ge',show=False)
##        self.Hi.plot(title='Hi',show=False)
##        self.He.plot(title='He',show=False)
##        self.kappa1_i.plot(title='kappa1_i',show=False)
##        self.kappa2_i.plot(title='kappa2_i',show=False)
##        self.kappa1_e.plot(title='kappa1_e',show=False)
##        self.kappa2_e.plot(title='kappa2_e',show=False)
#        plt.show()

    def calc_flux_coefficients(self):
        '''
        Computes A and B profiles for density and pressure.
        This involves finite difference gradients.
        '''
        
        # load
        n   = self.density
        pi  = self.pressure_i
        pe  = self.pressure_e
        Fn  = self.Fn
        Fpi = self.Fpi
        Fpe = self.Fpe

        Gamma = self.Gamma
        Qi    = self.Qi
        Qe    = self.Qe

        Gi  = self.Gi.profile
        Ge  = self.Ge.profile
        Hi  = self.Hi.profile
        He  = self.He.profile

        # normalization
        norm = 1 / self.a_minor / self.drho  # temp set R=1
        # because it should cancel with a R/L that I am also ignoring
        #norm = (self.R_major / self.a_minor) / self.drho 

        # calculate and save
        s = self
        self.Cn_n  = Flux_coefficients(n,  Fn, Gamma, s.G_n, norm)
        self.Cn_pi = Flux_coefficients(pi, Fn, Gamma, s.G_pi, norm) 
        self.Cn_pe = Flux_coefficients(pe, Fn, Gamma, s.G_pe, norm)

        self.Cpi_n  = Flux_coefficients(n,  Fpi, Qi, s.Qi_n, norm)
        self.Cpi_pi = Flux_coefficients(pi, Fpi, Qi, s.Qi_pi, norm) 
        self.Cpi_pe = Flux_coefficients(pe, Fpi, Qi, s.Qi_pe, norm)
        self.Cpe_n  = Flux_coefficients(n,  Fpe, Qe, s.Qe_n, norm)
        self.Cpe_pi = Flux_coefficients(pi, Fpe, Qe, s.Qe_pi, norm) 
        self.Cpe_pe = Flux_coefficients(pe, Fpe, Qe, s.Qe_pe, norm)
        # maybe these class definitions can be condensed

        # mu coefficients (to be deleted)
        self.mu1 = 0
        self.mu2 = 0
        self.mu3 = 0

        G_n = s.G_n.profile
        G_i = s.G_pi.profile
        G_e = s.G_pe.profile
        Qi_n = s.Qi_n.profile
        Qe_n = s.Qe_n.profile
        Qi_i = s.Qi_pi.profile
        Qe_i = s.Qe_pi.profile
        Qi_e = s.Qi_pe.profile
        Qe_e = s.Qe_pe.profile

        k1_i = s.kappa1_i.profile
        k1_e = s.kappa1_e.profile
        k2_i = s.kappa2_i.profile
        k2_e = s.kappa2_e.profile

        ### mu coefficients (Eq 7.109-7.111)
        # these mu's are missing 3rd K-term ~ H (EM potential)
        mu_1i = Gi * (G_n - 2.5/k1_i) + Hi * (Qi_n + 1/k2_i) 
        mu_1e = Ge * (G_n - 2.5/k1_e) + He * (Qe_n + 1/k2_e)  
        # what about when Gamma_e != Gamma_i (only for multiple species)
        mu_2i = Gi * (G_i + 1.5/k1_i) + Hi * (Qi_i - 1/k2_i) 
        mu_2e = Ge *  G_i             + He *  Qe_i   
        mu_3i = Gi *  G_e             + Hi *  Qi_e
        mu_3e = Ge * (G_e + 1.5/k1_e) + He * (Qe_e - 1/k2_e) 

        # save
        factor = 1. / (2 * self.drho)
        self.mu1_i = Flux_profile( factor * mu_1i )
        self.mu1_e = Flux_profile( factor * mu_1e )
        self.mu2_i = Flux_profile( factor * mu_2i )
        self.mu2_e = Flux_profile( factor * mu_2e )
        self.mu3_i = Flux_profile( factor * mu_3i )
        self.mu3_e = Flux_profile( factor * mu_3e )


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
        
        if self.no_collisions == "True":  
            # could write this to skil the function and return instead
            self.Ei = Ei*0
            self.Ee = Ei*0

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

        # save, computes matricies in class function
        self.psi_nn  = Psi_profiles(psi_nn_zero,
                                    psi_nn_plus,
                                    psi_nn_minus)

        self.psi_npi = Psi_profiles(psi_npi_zero,
                                    psi_npi_plus,
                                    psi_npi_minus)
        
        self.psi_npe = Psi_profiles(psi_npe_zero,
                                    psi_npe_plus,
                                    psi_npe_minus)
   
    def calc_psi_pi(self):
    
        # load
        F_p = self.Fpi.plus#.profile
        F_m = self.Fpi.minus#.profile
        E  = self.Ei

        n        = self.density       .profile
        n_p      = self.density.plus  .profile
        n_m      = self.density.minus .profile
        n_pp     = self.density.plus1 .profile
        n_mm     = self.density.minus1.profile

        pi       = self.pressure_i       .profile
        pi_plus  = self.pressure_i.plus  .profile
        pi_minus = self.pressure_i.minus .profile
        pi_pp    = self.pressure_i.plus1 .profile
        pi_mm    = self.pressure_i.minus1.profile

        pe       = self.pressure_e       .profile
        pe_plus  = self.pressure_e.plus  .profile
        pe_minus = self.pressure_e.minus .profile
        pe_pp    = self.pressure_e.plus1 .profile
        pe_mm    = self.pressure_e.minus1.profile
        
        An_pos = self.Cpi_n.plus  .profile
        An_neg = self.Cpi_n.minus .profile
        Bn     = self.Cpi_n.zero  .profile
        Ai_pos = self.Cpi_pi.plus .profile
        Ai_neg = self.Cpi_pi.minus.profile
        Bi     = self.Cpi_pi.zero .profile
        Ae_pos = self.Cpi_pe.plus .profile
        Ae_neg = self.Cpi_pe.minus.profile 
        Be     = self.Cpi_pe.zero .profile 
        
        Zi, mi = self.collision_model.export_species(0) # hard coded index
        Ze, me = self.collision_model.export_species(1) 

        E_pi = - E * ( Zi / (Ze*pe - Zi*pi) + 3./2 * me*Zi / (mi*Ze*pe + me*Zi*pi) )
        E_pe =   E * ( Ze / (Ze*pe - Zi*pi) - 3./2 * mi*Ze / (mi*Ze*pe + me*Zi*pi) )

        # new
        mu1 = self.mu1_i.full.profile
        mu2 = self.mu2_i.full.profile
        mu3 = self.mu3_i.full.profile

        G = self.Gi.full.profile
        H = self.Hi.full.profile

        # tri diagonal matrix elements
        g = self.geometry_factor 
        psi_pin_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1/n 
        psi_pin_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1/n 
        psi_pin_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) \
                          + 5/2 * E/n - 3/2 * (G + H)/n \
                          + mu1/n * (n_pp - n_mm)/n
                                
        psi_pipi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) - mu2/pi
        psi_pipi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) + mu2/pi
        psi_pipi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) ) \
                           + E_pi + 5/2 * (G + H)/pi \
                           + mu2/pi * (pi_pp - pi_mm)/pi
    
        psi_pipe_plus  = g * Ae_pos - mu3/pe
        psi_pipe_minus = g * Ae_neg + mu3/pe
        psi_pipe_zero  = g * Be + E_pe + mu3/pe * (pe_pp - pe_mm) / pe
                           

        # save (automatically computes matricies in class function)
        self.psi_pin  = Psi_profiles(psi_pin_zero,
                                     psi_pin_plus,
                                     psi_pin_minus)

        self.psi_pipi = Psi_profiles(psi_pipi_zero,
                                     psi_pipi_plus,
                                     psi_pipi_minus, neumann=False)
        
        self.psi_pipe = Psi_profiles(psi_pipe_zero,
                                     psi_pipe_plus,
                                     psi_pipe_minus)

    def calc_psi_pe(self):
    
        # load
        F_p = self.Fpe.plus.profile
        F_m = self.Fpe.minus.profile
        E   = self.Ee

        n        = self.density       .profile
        n_p      = self.density.plus  .profile
        n_m      = self.density.minus .profile
        n_pp     = self.density.plus1 .profile
        n_mm     = self.density.minus1.profile

        pi       = self.pressure_i       .profile
        pi_plus  = self.pressure_i.plus  .profile
        pi_minus = self.pressure_i.minus .profile
        pi_pp    = self.pressure_i.plus1 .profile
        pi_mm    = self.pressure_i.minus1.profile

        pe       = self.pressure_e       .profile
        pe_plus  = self.pressure_e.plus  .profile
        pe_minus = self.pressure_e.minus .profile
        pe_pp    = self.pressure_e.plus1 .profile
        pe_mm    = self.pressure_e.minus1.profile

        An_pos = self.Cpe_n.plus  .profile
        An_neg = self.Cpe_n.minus .profile
        Bn     = self.Cpe_n.zero  .profile
        Ai_pos = self.Cpe_pi.plus .profile
        Ai_neg = self.Cpe_pi.minus.profile
        Bi     = self.Cpe_pi.zero .profile
        Ae_pos = self.Cpe_pe.plus .profile
        Ae_neg = self.Cpe_pe.minus.profile 
        Be     = self.Cpe_pe.zero .profile 

        # new
        mu1 = self.mu1_e.full.profile
        mu2 = self.mu2_e.full.profile
        mu3 = self.mu3_e.full.profile

        G = self.Ge.full.profile
        H = self.He.full.profile

        Zi, mi = self.collision_model.export_species(0) # hard coded index
        Ze, me = self.collision_model.export_species(1) 

        E_pi =   E * ( Zi / (Zi*pi - Ze*pe) - 3/2 * me*Zi / (me*Zi*pi + mi*Ze*pe) )
        E_pe = - E * ( Zi / (Zi*pi - Ze*pe) + 3/2 * mi*Ze / (me*Zi*pi + mi*Ze*pe) )
    
        # tri diagonal matrix elements
        g = self.geometry_factor 
        psi_pen_plus  = g * (An_pos - 3/4 * F_p / n_p) - mu1/n 
        psi_pen_minus = g * (An_neg + 3/4 * F_m / n_m) + mu1/n
        psi_pen_zero  = g * (Bn +  3/4 * ( F_m/n_m - F_p/n_p ) ) \
                          + 5/2 * E/n - 3/2 * (G + H)/n \
                          + mu1/n * (n_pp - n_mm)/n
                                
        psi_pepi_plus  = g * (Ai_pos + 5/4 * F_p / pi_plus ) - mu2/pi
        psi_pepi_minus = g * (Ai_neg - 5/4 * F_m / pi_minus) + mu2/pi
        psi_pepi_zero  = g * (Bi - 5/4 * ( F_m/pi_minus - F_p/pi_plus) )  \
                           + E_pi - 3/2 * G/pi + 5/2 * H/pi \
                           + mu2/pi * (pi_pp - pi_mm)/pi
    
        psi_pepe_plus  = g * Ae_pos - mu3/pe
        psi_pepe_minus = g * Ae_neg + mu3/pe
        psi_pepe_zero  = g * Be + E_pe + G/pe + mu3/pe * (pe_pp - pe_mm)/pe


        # save (automatically computes matricies in class function)
        self.psi_pen  = Psi_profiles(psi_pen_zero,
                                     psi_pen_plus,
                                     psi_pen_minus)

        self.psi_pepi = Psi_profiles(psi_pepi_zero,
                                     psi_pepi_plus,
                                     psi_pepi_minus, neumann=False)
        
        self.psi_pepe = Psi_profiles(psi_pepe_zero,
                                     psi_pepe_plus,
                                     psi_pepe_minus)

    def time_step_LHS(self):
 
        # load, dropping last point for Dirchlet fixed boundary condition
        M_nn   = self.psi_nn .matrix[:-1, :-1]         
        M_npi  = self.psi_npi.matrix[:-1, :-1]   
        M_npe  = self.psi_npe.matrix[:-1, :-1]  

        # factor 2/3 for pressure (Barnes 7.115)
        M_pin  = self.psi_pin .matrix[:-1, :-1] * (2./3) 
        M_pipi = self.psi_pipi.matrix[:-1, :-1] * (2./3) 
        M_pipe = self.psi_pipe.matrix[:-1, :-1] * (2./3)      
 
        M_pen  = self.psi_pen .matrix[:-1, :-1] * (2./3)       
        M_pepi = self.psi_pepi.matrix[:-1, :-1] * (2./3)       
        M_pepe = self.psi_pepe.matrix[:-1, :-1] * (2./3)       

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
    def calc_sources(self):
    
        # load axuiliary source terms
        aux_source_n  = self.aux_source_n #[:-1]
        aux_source_pi = self.aux_source_pi#[:-1]
        aux_source_pe = self.aux_source_pe#[:-1]
        
        # load profiles
        n_profile_m3 = self.density.profile * 1e20
        Ti_profile_keV = self.pressure_i.profile / self.density.profile 
        Te_profile_keV = self.pressure_e.profile / self.density.profile 


        # compute fusion power
        if (self.alpha_heating == "True"):
            Ti_profile_eV = Ti_profile_keV * 1e3
            P_fusion_Wm3, fusion_rate  \
                    = fus.alpha_heating_DT( n_profile_m3, Ti_profile_eV )
        else:
            P_fusion_Wm3 = 0 * n_profile_m3
            fusion_rate = 0 * n_profile_m3

        # compute bremstrahlung radiation
        if (self.bremstrahlung == "True"):
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
        self.source_pi = aux_source_pi
        self.source_pe = aux_source_pe + P_fusion - P_brems


    ### Calculate the A Matrix
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

        Gi      = self.Gi.full.profile    [:-1]
        Ge      = self.Ge.full.profile    [:-1]
        Hi      = self.Hi.full.profile    [:-1]
        He      = self.He.full.profile    [:-1]

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

        # save for power balance 
        self.force_n  = force_n 
        self.force_pi = force_pi
        self.force_pe = force_pe

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
   
        ### RHS of Ax = b
        bvec_n  =  n_prev  \
                     + dtau*(1 - alpha)*force_n  \
                     + dtau*alpha*boundary_n  \
                     + dtau*source_n  
        bvec_pi =  pi_prev \
                     + (2/3) * dtau*(1 - alpha) * (force_pi + Ei + Gi + Hi) \
                     + (2/3) * dtau*alpha*boundary_pi \
                     + dtau*source_pi 
        bvec_pe =  pe_prev \
                     + (2/3) * dtau*(1 - alpha) * (force_pe + Ee + Ge + He) \
                     + (2/3) * dtau*alpha*boundary_pe \
                     + dtau*source_pe 
        # 7/18, these boundary terms were added based on Kittel, but they are not in MAB's thesis. They are important for getting the dynamics at the second to edge point correct.

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
        '''
        Load the results from y = Ab, update profiles
        '''

        # new data for next time step
        y_next  = self.y_next   

        # old data from previous time step
        n_prev = self.density.profile
        pi_prev = self.pressure_i.profile
        pe_prev = self.pressure_e.profile

        # fixed boundary conditions
        n_edge  = self.n_edge   
        pi_edge = self.pi_edge
        pe_edge = self.pe_edge

        N_mat = self.N_radial - 1
        n_next, pi_next, pe_next = np.reshape( y_next,(3,N_mat) )

        n  = np.concatenate([ n_next , [n_edge]  ]) 
        pi = np.concatenate([ pi_next, [pi_edge] ]) 
        pe = np.concatenate([ pe_next, [pe_edge] ])

        self.density    = Profile(n,  grad=True, half=True, full=True)
        self.pressure_i = Profile(pi, grad=True, half=True, full=True)
        self.pressure_e = Profile(pe, grad=True, half=True, full=True)

        # step time
        self.time += self.dtau
        self.t_idx += 1 # this is an integer index of all time steps

        ## record change
        delta_pi = profile_diff(pi, pi_prev) #np.std(pi - pi_prev)
        delta_pe = profile_diff(pe, pe_prev) #np.std(pe - pe_prev)
        delta_n  = profile_diff(n , n_prev) #np.std(n  - n_prev)
        print("*****")
        print(f"(dpi, dpe, dn) = {delta_pi}, {delta_pe}, {delta_n}")
        print("*****")

### How to choose the threshold?
##   Maybe track convergence instead of magnitude?
#        if (delta_pi > 0.018):
#            self.needs_new_flux = False
#        else:
#            self.needs_new_flux = True

        p_SI = (pi + pe) * 1e20 * (1e3 * 1.6e-19)
        self.vmec_pressure = p_SI

        if profile_diff(p_SI, self.vmec_pressure_old) > 0.1:
            print("***** needs new VMEC ****** threshold exceeds 10%")
            self.needs_new_vmec = True
            self.vmec_pressure_old = p_SI # maybe move this elsewhere

    def reset_fluxtubes(self):

        if self.needs_new_vmec == False:
            return

        ### needs new vmec

        # run VMEC (wait)

        # update the wout

        # gx.init_geometry() # make new flux tubes from wout
        # maybe rename this to make_fluxtubes()


    # a subclass for handling normalizations in Trinity
    class Normalizations():
        def __init__(self, n_ref = 1e20,     # m3
                           T_ref = 1e3,      # eV
                           B_ref = 1,        # T
                           m_ref = 1.67e-27, # kg, proton mass
                           a_minor = 1,      # m
                          ):

            # could get these constants from scipy
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
            particle_source_scale = t_ref / n_ref * gyro_scale**2 

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
            self.particle_source_scale = particle_source_scale


def profile_diff(arr, old):
    # this is an L_infinity norm
    return np.max( np.abs( (arr - old)/old ) )
