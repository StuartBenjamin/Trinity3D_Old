import numpy as np
import subprocess
from datetime import datetime
import time as _time

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner
#from Collisions import Collision_Model

# read GX output
import trinity_lib as trl
import profiles as pf
import GX_io as gx_io
import os
from glob import glob

'''
This library contains model functons for fluxes.

+ there is an analytic ReLU model called Flux_model
+ there is an anlaytic model from Barnes' thesis
+ there is the GX flux model
'''

def ReLU(x,a=1,m=1):
    '''
       piecewise-linear function
       can model Gamma( critical temperature gradient scale length ), for example
       x is a number, a and m are constants

       inputs : a number
       outputs: a number
    '''
    if (x < a):
        return 0
    else:
        return m*(x-a)

ReLU = np.vectorize(ReLU,otypes=[np.float64])

def Step(x,a=0.5,m=1.):
    # derivative of ReLU (is just step function)
    if (x < a):
        return 0
    else:
        return m


# for a particle and heat sources
def Gaussian(x, A=2, sigma=.3, x0=0):
    exp = - ( (x - x0) / sigma)**2  / 2
    return A * np.e ** exp


'''
analytic flux model based on ReLU + neoclassical
'''
# rename as ReluFluxModel
class Flux_model():

    def __init__(self,
               # neoclassical diffusion coefficient
               D_neo  = 0, # 0.1
               # critical gradient
               n_critical_gradient  = 1, 
               pi_critical_gradient = 2,
               pe_critical_gradient = 2, # 9/5 this used to be 1,1,1
               # slope of flux(Ln) after onset
               n_flux_slope  = 0.5, # 9/5 this used to be 1.1,1.1,1.1
               pi_flux_slope = 0.5,
               pe_flux_slope = 0.5,
               zero_flux = False,
               ):

        if zero_flux:

            # setting the critical gradient a=1e6 should temporarily turn off flux 
            #    default is 0.5
             n_critical_gradient  = 1e6 
             pi_critical_gradient = 1e6
             pe_critical_gradient = 1e6

             # also set neoclassical flux to zero
             D_neo = 0

        # store
        self.neo = D_neo
        self.n_critical_gradient  = n_critical_gradient   
        self.pi_critical_gradient = pi_critical_gradient 
        self.pe_critical_gradient = pe_critical_gradient 
        self.n_flux_slope  = n_flux_slope  
        self.pi_flux_slope = pi_flux_slope 
        self.pe_flux_slope = pe_flux_slope 

    # input: 3 inverse gradient length scales
    #        kn = 1/L_n = grad n / n = grad ln n
    def flux(self, kn, kpi, kpe):

        ### modelling turbulence from three types of gradients
        D_n  = ReLU(kn , a=self.n_critical_gradient , m=self.n_flux_slope ) 
        D_pi = ReLU(kpi, a=self.pi_critical_gradient, m=self.pi_flux_slope) 
        D_pe = ReLU(kpe, a=self.pe_critical_gradient, m=self.pe_flux_slope) 

        # mix the contributions from all profiles
        D_turb = D_n + D_pi + D_pe # does not include neoclassical part
        return D_turb

    # compute the derivative with respect to gradient scale length
    #     dx is an arbitrary step size
    def flux_gradients(self, kn, kpi, kpe, step = 0.1):
        
        # turbulent flux calls
        d0  = self.flux(kn, kpi, kpe)
        dn  = self.flux(kn + step, kpi, kpe)
        dpi = self.flux(kn, kpi + step, kpe)
        dpe = self.flux(kn, kpi, kpe + step)

        # finite differencing
        grad_Dn  = (dn-d0) / step
        grad_Dpi = (dpi-d0) / step
        grad_Dpe = (dpe-d0) / step

        return grad_Dn, grad_Dpi, grad_Dpe

    # move this function from model to Trinity_lib, because it does not depend on the particular model


class Barnes_Model2():
    """
    This test model follows Eq (7.163) in Section 7.8.1 of Michael Barnes' thesis
    """
    # should this test automatically turn off sources?

    def __init__(self, D = 1):

        self.D = D

    def compute_Q(self,engine, step=0.1):

        pi = engine.pressure_i.midpoints
        pe = engine.pressure_e.midpoints

        Lpi = - engine.pressure_i.grad_log.profile  # a / L_pi
        Lpe = - engine.pressure_e.grad_log.profile  # a / L_pe

        D = self.D
        Qi = 1.5 * D * Lpi / pi**(-1.5) # assume p_ref = pi
        Qe = 1.5 * D * Lpe * pe / pi**(-2.5)

        zero  = 0*pi
        Gamma = zero  # do not evolve particles

        # Perturb
        Qi_pi = 1.5 * D * (Lpi+step) / pi**(-1.5) # assume p_ref = pi
        Qe_pe = 1.5 * D * (Lpe+step) * pe / pi**(-2.5)

        dQi_pi = (Qi_pi - Qi) / step
        dQe_pe = (Qe_pe - Qe) / step

        # save
        engine.Gamma  = pf.Flux_profile(zero  )
        engine.Qi     = pf.Flux_profile(Qi    ) 
        engine.Qe     = pf.Flux_profile(Qe    ) 
        engine.G_n    = pf.Flux_profile(zero  )
        engine.G_pi   = pf.Flux_profile(zero  )
        engine.G_pe   = pf.Flux_profile(zero  )
        engine.Qi_n   = pf.Flux_profile(zero  )
        engine.Qi_pi  = pf.Flux_profile(dQi_pi)
        engine.Qi_pe  = pf.Flux_profile(zero  )
        engine.Qe_n   = pf.Flux_profile(zero  )
        engine.Qe_pi  = pf.Flux_profile(zero  )
        engine.Qe_pe  = pf.Flux_profile(dQe_pe)


WAIT_TIME = 1  # this should come from the Trinity Engine
class GX_Flux_Model():
    '''
    How Trinity handles all things GX!
    '''

    def __init__(self, engine, 
                       gx_root='gx-files/', 
                       path='run-dir/', 
                       vmec_path='./',
                       vmec_wout="",
                       midpoints=[]
                ):

        self.engine = engine

#        gx_root = "gx-files/"  # this is part of repo, don't change # old, 9/7
        #f_input = 'gx-adiabatic-electrons.in' # Moose, this should be renamed as adiabatic electrons.
        #f_geo   = 'gx-geometry-adiabatic-electrons.ing' # sample input file, to be part of repo
        # Moose add option to specify GX input file in trinity.in

        print('self.engine.two_species_ionscale is {}'.format(self.engine.two_species_ionscale))
        #if self.engine.two_species == True: # We are not finding this...
        #    f_input = 'gx-two-kinetic-species.in' # Moose, this should be renamed as adiabatic electrons.
        #    f_geo   = 'gx-geometry-two-kinetic-species.ing' # sample input file, to be part of repo
        #    print('Found two species input file.')
        #elif self.engine.kinetic_ions == False: # Adiabatic ion simulation.
        #    f_input = 'gx-adiabatic-ions.in' # Moose, this should be renamed as adiabatic electrons.
        #    f_geo   = 'gx-geometry-adiabatic-ions.ing' # sample input file, to be part of repo
        #    print('Found adiabatic ions input file.')
        #elif self.engine.kinetic_electrons == False: # Adiabatic electron simulation.
        #    f_input = 'gx-adiabatic-electrons.in' # Moose, this should be renamed as adiabatic electrons.
        #    f_geo   = 'gx-geometry-adiabatic-electrons.ing' # sample input file, to be part of repo
        #    print('Found adiabatic electrons input file.')

        # Moose
        if self.engine.two_species_ionscale == True: # We are not finding this...
            f_input_ionscale = 'gx-two-kinetic-species.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found two species input file at ion scales.')
        elif self.engine.kinetic_ions_ionscale == False: # Adiabatic ion simulation.
            f_input_ionscale = 'gx-adiabatic-ions.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found adiabatic ions input file at ion scales.')
        if self.engine.kinetic_electrons_ionscale == False: # Adiabatic electron simulation.
            f_input_ionscale = 'gx-adiabatic-electrons.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found adiabatic electrons input file at ion scales.')

        if self.engine.two_species_electronscale == True: # We are not finding this...
            f_input_electronscale = 'gx-two-kinetic-species.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found two species input file at electron scales.')
        elif self.engine.kinetic_ions_electronscale == False: # Adiabatic ion simulation.
            f_input_electronscale = 'gx-adiabatic-ions.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found adiabatic ions input file at electron scales.')
        elif self.engine.kinetic_electrons_electronscale == False: # Adiabatic electron simulation.
            f_input_electronscale = 'gx-adiabatic-electrons.in' # Moose, this should be renamed as adiabatic electrons.
            print('Found adiabatic electrons input file at electron scales.')
 
        #### Geo is independent of ion or electron flux tube scales / kinetic species.
        f_geo = 'gx-geometry-two-kinetic-species.ing'

        ### Check file path
        print("  Looking for GX files")
        print("    GX input path:", gx_root)
        print("      expecting GX template at ion scales:", gx_root + f_input_ionscale)
        print("      expecting GX template at electron scales:", gx_root + f_input_electronscale)
        print("      expecting GX executable:", gx_root + "gx")
        print("      expecting GX-VMEC template:", gx_root + f_geo)
        print("      expecting GX-VMEC executable:", gx_root + "convert_VMEC_to_GX")
        print("    VMEC path:", vmec_path)
        print("      expecting VMEC wout:", vmec_path + vmec_wout)
        print("    GX-Trinity output path:", path)

        found_path = os.path.exists(path)
        if (found_path == False):
            print(f"      creating new output dir {path}")
            os.mkdir(path)

        found_gx = os.path.exists(path+"gx")
        if (found_gx == False):
            print(f"      copying gx executable from root into {path}")
            cmd = f"cp {gx_root}gx {path}gx"
            os.system(cmd)

        print("")

        # check using something like
        # os.listdir(vmec_path).find(vmec_wout)

        ###  load an input template
        #    later, this should come from Trinity input file
        if self.engine.ionscale_fluxtube == True:
            self.input_template_ionscale = GX_Runner(gx_root + f_input_ionscale)
        if self.engine.electronscale_fluxtube == True:
            self.input_template_electronscale = GX_Runner(gx_root + f_input_electronscale)
        # Moose to make each fluxtube different.
        self.path = path # this is the GX output path (todo: rename)
        # check that path exists, if it does not, mkdir and copy gx executable
        
        self.midpoints = midpoints # Moose to rename cell_centres or axial_midpoints
        self.vmec_path = vmec_path
        self.vmec_wout = vmec_wout
        self.gx_root = gx_root
        self.f_geo   = f_geo # template convert geometry input

### retired 8/14
#        ### This keeps a record of GX comands, it might be retired
#        # init file for writing GX commands
#
#        with  open(fname,'w') as f:
#            print('t_idx, r_idx, time, r, s, tprim, fprim', file=f)
#
#        # store file name (includes path)
#        self.fname = fname
#        self.f_handle = open(fname, 'a')
#        ###

        self.processes = []

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [ p.wait() for p in self.processes ]
        print(exitcodes)
        self.processes = [] # reset

        # could add some sort of timer here


    def make_fluxtubes(self):
    #def init_geometry(self): # rename 8/15

        ### load fluxtube geometry
        # these should come from Trinity input file
        
        vmec = self.vmec_wout
        if vmec: # is not blank

            # else launch fluxtubes from VMEC
            f_geo     = self.f_geo
            #f_geo     = 'gx-geometry-sample.ing' # sample input file, to be part of repo ## 7/21 can delete
            geo_path  = self.gx_root  # this says where the convert executable lives, and where to find the sample .ing file
            out_path  = self.path
            vmec_path = self.vmec_path

            geo_template = gx_io.VMEC_GX_geometry_module( self.engine,
                                                 f_sample = f_geo,
                                                 input_path = geo_path,
                                                 output_path = out_path,
                                                 tag = vmec[5:-3]
                                              )
            geo_template.set_vmec( vmec, 
                          vmec_path   = vmec_path, 
                          output_path = out_path )

            geo_files = []
            N_fluxtubes = len(self.midpoints)
            print('N_fluxtubes is {}'.format(N_fluxtubes))
            # Moose, don't include ion and electron scale as separate flux tube?
            for j in np.arange(N_fluxtubes):
                rho = self.midpoints[j]
                f_geometry = geo_template.init_radius(rho,j) 
                geo_files.append(out_path + f_geometry)

            # kludgy fix, if the inner most fluxtube is too small for VMEC resolution
            #     just copy the second inner most fluxtube
            #     the gradients will be different (and correct) even though the geometries are faked
            #if len(geo_files) < len(self.midpoints):
            #    geo_files = np.concatenate( [[geo_files[0]], geo_files] )
            # one solution could be to constrain trinity to rho > 0.3 (psi ~ 0.1)

        else:
            # load default files (assumed to be existing)
            print('  no VMEC wout given, loading default files')
            geo_files = [ 'gx-files/gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.295_gds21_nt_38_geo.nc',  
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.500_gds21_nt_40_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.704_gds21_nt_42_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.897_gds21_nt_42_geo.nc']

        #print(' Found these fluxtubes', geo_files) # removed 8/14
        print("")

        ### store fluxtubes in a list
        self.flux_tubes = []
        ### Moose load fluxtubes at each ion and electron scale.
        print('self.engine.ionscale_fluxtube is {}'.format(self.engine.ionscale_fluxtube))
        if self.engine.ionscale_fluxtube == True:
            for fin in geo_files:
                self.load_fluxtube(fin, fluxtube_scale = "ion")
                print('Loading ion scale flux tubes.')
        if self.engine.electronscale_fluxtube == True:
            for fin in geo_files:
                self.load_fluxtube(fin, fluxtube_scale = "electron")
                print('Loading electron scale flux tubes.')

        #print('self.flux_tubes is {}'.format(self.flux_tubes))

        # make a directory for restart files
        restart_dir = self.path + 'restarts' # this name is hard-coded to agree with that in gx_command(), a better name may be restart_dir/ or a variable naming such
        if os.path.exists(restart_dir) == False:
            os.mkdir(restart_dir)

    def create_geometry_from_vmec(self,wout):

        # load sample geometry file

        # make changes (i.e. specify radius, ntheta etc.)

        # write new file and run GX-VMEC geometry module

        # wait, load new geometry files
        pass

    def load_fluxtube(self, f_geo, fluxtube_scale = "ion"):

        ft = FluxTube(f_geo)       # init an instance of fluxtube class
        #if self.engine.ionscale_fluxtube == True:
        #    ft.load_gx_input(self.input_template_ionscale) # Moose.
        if fluxtube_scale == "ion":
            print('Loading ionscale fluxtube')
            ft.load_gx_input(self.input_template_ionscale)
        if fluxtube_scale == "electron":
            print('Loading electronscale fluxtube')
            ft.load_gx_input(self.input_template_electronscale)
        # save
        self.flux_tubes.append(ft)


    def prep_commands(self, engine, # pointer to pull profiles from trinity engine
                            t_id,   # integer time index in trinity
                            step = 0.1, # absolute step size for perturbing gradients
                     ):

        self.t_id = t_id
        self.time = engine.time

        # preparing dimensionless (tprim = L_ref/LT) for GX
        Ln  = - engine.density.grad_log   .profile  # a / L_n
        Lpi = - engine.pressure_i.grad_log.profile  # a / L_pi
        Lpe = - engine.pressure_e.grad_log.profile  # a / L_pe

        # get normalizations for GX, dens and temp
        n  = engine.density   .midpoints
        pi = engine.pressure_i.midpoints
        pe = engine.pressure_e.midpoints

        Ti = pi/n
        Te = pe/n

        Tref = Ti # hard-coded convention Moose this could cause issues.
        gx_Ti = Ti/Tref
        gx_Te = Te/Tref

        # turbulent flux calls, for each radial fluxtube
        mid_axis = engine.mid_axis
        idx = np.arange( len(mid_axis) ) 

        # Launch ion scale fluxtubes.
        if engine.ionscale_fluxtube == True: 
            f0_ionscale   = [''] * len(idx) 
            fn_ionscale   = [''] * len(idx) 
            fti_ionscale  = [''] * len(idx) 
            fte_ionscale  = [''] * len(idx) 

            Q0i_baseline_ionscale   = np.zeros( len(idx) )
            Qi_ti_scan_ionscale   = np.zeros( len(idx) )
            Qi_n_scan_ionscale  = np.zeros( len(idx) )
            Qi_te_scan_ionscale  = np.zeros( len(idx) )
            Q0e_baseline_ionscale  = np.zeros( len(idx) )
            Qe_te_scan_ionscale  = np.zeros( len(idx) )
            Qe_n_scan_ionscale  = np.zeros( len(idx) )
            Qe_ti_scan_ionscale  = np.zeros( len(idx) )

        for j in idx: # Loop over radial positions.
            rho = mid_axis[j]
            kn  = Ln [j]
            kpi = Lpi[j]
            kpe = Lpe[j]

            kti = kpi - kn
            kte = kpe - kn

            temp_i = gx_Ti[j]
            temp_e = gx_Te[j]

            # writes the GX input file and calls the slurm 
            scale = 1 + step
            print('Launching ion scale flux tubes. j is {}'.format(j))
            f0_ionscale [j] = self.gx_command(j, rho, kn      , kti       , kte        , '0iscale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'ionscale') # Moose
            fn_ionscale [j] = self.gx_command(j, rho, kn*scale , kti        , kte        , '1iscale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'ionscale' ) # Always submit density scan.
            if engine.kinetic_ions_ionscale == True: # If kinetic ions (includes both adiabatic electron and two kinetic species simulations), perturb LTi.
                # Moose why is this perturbing ion temperature gradient, whereas fpe perturbs electron pressure gradient? Weird.
                fti_ionscale[j] = self.gx_command(j, rho, kn      , kti*scale , kte        , '2iscale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'ionscale')
                print('fti_ionscale[j] is {}'.format(fti_ionscale[j]))

            if engine.kinetic_electrons_ionscale == True: # If kinetic electrons (includes both adiabatic ion and two kinetic species simulations), perturb LTe.
                fte_ionscale[j] = self.gx_command(j, rho, kn      , kti        , kte*scale , '3iscale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'ionscale' )
                # Moose kpe --> kte

        # Launch electron scale fluxtubes. _electronscale suffix indicates 'electron gyroradius scale'.
        if engine.electronscale_fluxtube == True:

            f0_electronscale   = [''] * len(idx)
            fn_electronscale   = [''] * len(idx)
            fti_electronscale  = [''] * len(idx)
            fte_electronscale  = [''] * len(idx)

            Q0i_baseline_electronscale   = np.zeros( len(idx) )
            Qi_ti_scan_electronscale   = np.zeros( len(idx) )
            Qi_n_scan_electronscale  = np.zeros( len(idx) )
            Qi_te_scan_electronscale  = np.zeros( len(idx) )
            Q0e_baseline_electronscale  = np.zeros( len(idx) )
            Qe_te_scan_electronscale  = np.zeros( len(idx) )
            Qe_n_scan_electronscale  = np.zeros( len(idx) )
            Qe_ti_scan_electronscale  = np.zeros( len(idx) )

        for j in idx: # Loop over radial positions. Moose, do I need to add additional j indexing for electron scale stuff?
            rho = mid_axis[j]
            kn  = Ln [j]
            kpi = Lpi[j]
            kpe = Lpe[j]

            kti = kpi - kn
            kte = kpe - kn

            temp_i = gx_Ti[j]
            temp_e = gx_Te[j]

            # writes the GX input file and calls the slurm 
            scale = 1 + step
            print('Launching electron scale flux tubes. j is {}'.format(j))
            f0_electronscale [j] = self.gx_command(j, rho, kn      , kti       , kte        , '0escale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'electronscale', kinetic_ions = engine.kinetic_ions_electronscale) # Moose
            fn_electronscale [j] = self.gx_command(j, rho, kn*scale , kti        , kte        , '1escale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'electronscale', kinetic_ions = engine.kinetic_ions_electronscale )
            if engine.kinetic_ions_electronscale == True: # If kinetic ions (includes both adiabatic electron and two kinetic species simulations), perturb LTi.
                # Moose why is this perturbing ion temperature gradient, whereas fpe perturbs electron pressure gradient? Weird.
                fti_electronscale[j] = self.gx_command(j, rho, kn      , kti*scale , kte        , '2escale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'electronscale', kinetic_ions = engine.kinetic_ions_electronscale)
            if engine.kinetic_electrons_electronscale == True: # If kinetic electrons (includes both adiabatic ion and two kinetic species simulations), perturb LTe.
                fte_electronscale[j] = self.gx_command(j, rho, kn      , kti        , kte*scale , '3escale', temp_i = temp_i, temp_e = temp_e , flux_tube_type = 'electronscale' , kinetic_ions = engine.kinetic_ions_electronscale)

            ## there is choice, relative step * or absolute step +?

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        # Moose, need to add all the fluxes, Sep 13 2022. Go through tomorrow and add more.

        # Sep 14 2022: fluxes now complete.

        # Loop over radius, read ion scale fluxes.

        # NOTATION:
        # Q0i_pi_scan: Q is heat flux, i is ion species, ti is ion temperature gradient scan, _scan means perturbed wrt Lti

        if engine.ionscale_fluxtube == True:
            print('starting to read ion scale fluxes.')
            print('engine.kinetic_ions_ionscale is ' + str(engine.kinetic_ions_ionscale))
            print('idx is {}'.format(idx))
            for j in idx: 

                kn  = Ln [j]
                kpi = Lpi[j]
                kpe = Lpe[j]
                kti = kpi - kn
                kte = kpe - kn

                # loop over fluxtubes
                print('second engine.kinetic_ions_ionscale is ' + str(engine.kinetic_ions_ionscale))
                print('j is {}'.format(j))
                print('logic engine.kinetic_ions_ionscale == True is ... {}'.format(engine.kinetic_ions_ionscale == True))
                print('type(engine.kinetic_ions_ionscale ==) {}'.format(type(engine.kinetic_ions_ionscale)))
                print('type(engine.kinetic_electrons_ionscale ==) {}'.format(type(engine.kinetic_electrons_ionscale)))
                print('type(engine.two_species_ionscale ==) {}'.format(type(engine.two_species_ionscale)))

                if engine.kinetic_ions_ionscale == True: # response of ion heat flux to ion temp perturbation. Moose, needs cleaning up
                    Q0i_baseline_ionscale [j] = return_gx_heat_flux(f0_ionscale [j], 0) # First argument is flux tube simulation, second is species index: 0 for ions, 1 for electrons.
                    print('Q0i_baseline_ionscale is {}'.format(Q0i_baseline_ionscale))
                    print('fti_ionscale[j] is {}'.format(fti_ionscale[j]))
                    Qi_ti_scan_ionscale[j] = return_gx_heat_flux(fti_ionscale[j], 0)
                    Qi_n_scan_ionscale [j] = return_gx_heat_flux(fn_ionscale [j], 0)

                if engine.two_species_ionscale == True: #  'cross terms': 
                    Qe_ti_scan_ionscale[j] = return_gx_heat_flux(fti_ionscale[j], 1)
                    Qi_te_scan_ionscale[j] = return_gx_heat_flux(fte_ionscale[j], 0)

                if engine.kinetic_electrons_ionscale == True:
                    if engine.two_species_ionscale == True:
                        Q0e_baseline_ionscale [j] = return_gx_heat_flux(f0_ionscale [j], 1)
                        Qe_te_scan_ionscale[j] = return_gx_heat_flux(fte_ionscale[j], 1)
                        Qe_n_scan_ionscale [j] = return_gx_heat_flux(fn_ionscale [j], 1) # Moose activate once we have non-adiabatic electrons.
                    else:
                        Q0e_baseline_ionscale [j] = return_gx_heat_flux(f0_ionscale [j], 0)
                        Qe_te_scan_ionscale[j] = return_gx_heat_flux(fte_ionscale[j], 0)
                        Qe_n_scan_ionscale [j] = return_gx_heat_flux(fn_ionscale [j], 0) # Moose activate once we have non-adiabatic electrons.

                if engine.kinetic_ions_ionscale == False:
                    Q0i_baseline_ionscale = 0*Q0e_baseline_ionscale
                    Qi_ti_scan_ionscale = 0*Qe_te_scan_ionscale
                    Qi_n_scan_ionscale = 0*Qe_n_scan_ionscale 

                if engine.kinetic_electrons_ionscale == False:
                    Qe_ti_scan_ionscale = 0*Qi_ti_scan_ionscale
                    Qi_te_scan_ionscale = 0*Qi_ti_scan_ionscale

                if engine.two_species_ionscale == False:
                    Q0e_baseline_ionscale = 0*Q0i_baseline_ionscale
                    Qe_te_scan_ionscale = 0*Qi_ti_scan_ionscale
                    Qe_n_scan_ionscale = 0*Qi_n_scan_ionscale

	    # record dQ / dLx
            if engine.kinetic_ions_ionscale == True: # If kinetic ions (includes both adiabatic electron and two kinetic species simulations), perturb LTi.
                Qi_ti_deriv_ionscale  =  (Qi_ti_scan_ionscale - Q0i_baseline_ionscale) / (kti * step) # Moose changing denom to (kti * step) from (Lpi * step)
                Qi_n_deriv_ionscale   =  (Qi_n_scan_ionscale  - Q0i_baseline_ionscale) / (Ln * step)
            if engine.two_species_ionscale == True: # If two kinetic species.
                Qe_ti_deriv_ionscale  =  (Qe_ti_scan_ionscale - Q0e_baseline_ionscale) / (kti * step)
                Qi_te_deriv_ionscale  =  (Qi_te_scan_ionscale - Q0i_baseline_ionscale) / (kte * step)
            if engine.kinetic_electrons_ionscale == True: # If kinetic electrons (includes both adiabatic ion and two kinetic species simulations), perturb LTe.
                Qe_te_deriv_ionscale  =  (Qe_te_scan_ionscale - Q0e_baseline_ionscale) / (kte * step)
                Qe_n_deriv_ionscale   =  (Qe_n_scan_ionscale  - Q0e_baseline_ionscale) / (Ln * step)
            if engine.kinetic_ions_ionscale == False:
                Qi_ti_deriv_ionscale  = 0*Qe_te_deriv_ionscale # Moose, is this correct? Set to zero? Alternatives: Qe_te_deriv_ionscale
                Qi_n_deriv_ionscale   = 0*Qe_n_deriv_ionscale # Moose, is this correct? Set to zero? Alternatives: Qe_n_deriv_ionscale
            if engine.kinetic_electrons_ionscale == False:
                Qe_te_deriv_ionscale  = 0*Qi_ti_deriv_ionscale
                Qe_n_deriv_ionscale   = 0*Qi_n_deriv_ionscale
            if engine.two_species_ionscale == False:
                Qe_ti_deriv_ionscale  =  0*Qi_ti_deriv_ionscale
                Qi_te_deriv_ionscale  =  0*Qi_ti_deriv_ionscale

	    # need to add neoclassical diffusion

            # save, this is what engine.compute_flux() writes
            zero = 0*Q0i_baseline_ionscale
            eps = 1e-8 + zero # want to avoid divide by zero

            print('Q0i_baseline_ionscale is {}'.format(Q0i_baseline_ionscale))

            engine.Gamma_ionscale  = pf.Flux_profile(eps  ) # Moose, set these up to get profile data.
            engine.Qi_ionscale     = pf.Flux_profile(Q0i_baseline_ionscale) 
            engine.Qe_ionscale     = pf.Flux_profile(Q0e_baseline_ionscale) 
            engine.G_n_ionscale    = pf.Flux_profile(zero ) #???Moose
            engine.G_pi_ionscale   = pf.Flux_profile(zero )
            engine.G_pe_ionscale   = pf.Flux_profile(zero )
            engine.Qi_n_ionscale   = pf.Flux_profile(Qi_n_deriv_ionscale ) # d Q / d Ln
            engine.Qi_pi_ionscale  = pf.Flux_profile(Qi_ti_deriv_ionscale) # d Q / d LTi
            engine.Qi_pe_ionscale  = pf.Flux_profile(Qi_te_deriv_ionscale)
            engine.Qe_n_ionscale   = pf.Flux_profile(Qe_n_deriv_ionscale)
            engine.Qe_pi_ionscale  = pf.Flux_profile(Qe_ti_deriv_ionscale)
            engine.Qe_pe_ionscale  = pf.Flux_profile(Qe_te_deriv_ionscale)

        # Loop over radius, read electron scale fluxes, find heat flux gradients.
        if engine.electronscale_fluxtube == True: # Moose string, not boolean. Change.
            print('starting to read electron scale fluxes.')
            for j in idx:

                kn  = Ln [j]
                kpi = Lpi[j]
                kpe = Lpe[j]
                kti = kpi - kn
                kte = kpe - kn

                # loop over fluxtubes
                if engine.kinetic_ions_electronscale == True: # response of ion heat flux to ion temp perturbation. Moose, needs cleaning up
                    Q0i_baseline_electronscale [j] = return_gx_heat_flux(f0_electronscale [j], 0) # First argument is flux tube simulation, second is species index: 0 for ions, 1 for electrons.
                    Qi_ti_scan_electronscale[j] = return_gx_heat_flux(fti_electronscale[j], 0)
                    Qi_n_scan_electronscale [j] = return_gx_heat_flux(fn_electronscale [j], 0)

                if engine.two_species_electronscale == True: #  'cross terms': 
                    Qe_ti_scan_electronscale[j] = return_gx_heat_flux(fti_electronscale[j], 1)
                    Qi_te_scan_electronscale[j] = return_gx_heat_flux(fte_electronscale[j], 0)

                if engine.kinetic_electrons_electronscale == True:
                    if engine.two_species_electronscale == True:
                        Q0e_baseline_electronscale [j] = return_gx_heat_flux(f0_electronscale [j], 1)
                        Qe_te_scan_electronscale[j] = return_gx_heat_flux(fte_electronscale[j], 1)
                        Qe_n_scan_electronscale [j] = return_gx_heat_flux(fn_electronscale [j], 1) # Moose activate once we have non-adiabatic electrons.
                    else:
                        Q0e_baseline_electronscale [j] = return_gx_heat_flux(f0_electronscale [j], 0)
                        Qe_te_scan_electronscale[j] = return_gx_heat_flux(fte_electronscale[j], 0)
                        Qe_n_scan_electronscale [j] = return_gx_heat_flux(fn_electronscale [j], 0) # Moose activate once we have non-adiabatic electrons.

                if engine.kinetic_ions_electronscale == False:
                    Q0i_baseline_electronscale = 0*Q0e_baseline_electronscale
                    Qi_ti_scan_electronscale = 0*Qe_te_scan_electronscale
                    Qi_n_scan_electronscale = 0*Qe_n_scan_electronscale

                if engine.kinetic_electrons_electronscale == False:
                    Qe_ti_scan_electronscale = 0*Qi_ti_scan_electronscale
                    Qi_te_scan_electronscale = 0*Qi_ti_scan_electronscale

                if engine.two_species_electronscale == False:
                    Q0e_baseline_electronscale = 0*Q0i_baseline_electronscale
                    Qe_te_scan_electronscale = 0*Qi_ti_scan_electronscale
                    Qe_n_scan_electronscale = 0*Qi_n_scan_electronscale

            # Moose: how to choose which gradient scans to perform.
            # local shorthands, contained within function. More broadly, more verbose variable names, better`

            # record the heat flux
            #Qflux_electronscale  =  Q0
            # record dQ / dLx
            if engine.kinetic_ions_electronscale == True: # If kinetic ions (includes both adiabatic electron and two kinetic species simulations), perturb LTi.
                Qi_ti_deriv_electronscale  =  (Qi_ti_scan_electronscale - Q0i_baseline_electronscale) / (kti * step) # Moose changing denom to (kti * step) from (Lpi * step)
                Qi_n_deriv_electronscale   =  (Qi_n_scan_electronscale  - Q0i_baseline_electronscale) / (Ln * step)
            if engine.two_species_electronscale == True: # If two kinetic species.
                Qe_ti_deriv_electronscale  =  (Qe_ti_scan_electronscale - Q0e_baseline_electronscale) / (kti * step)
                Qi_te_deriv_electronscale  =  (Qi_te_scan_electronscale - Q0i_baseline_electronscale) / (kte * step)
            if engine.kinetic_electrons_electronscale == True: # If kinetic electrons (includes both adiabatic ion and two kinetic species simulations), perturb LTe.
                Qe_te_deriv_electronscale  =  (Qe_te_scan_electronscale - Q0e_baseline_electronscale) / (kte * step)
                Qe_n_deriv_electronscale   =  (Qe_n_scan_electronscale  - Q0e_baseline_electronscale) / (Ln * step)
            if engine.kinetic_ions_electronscale == False:
                Qi_ti_deriv_electronscale  = 0*Qe_te_deriv_electronscale # Moose, is this correct? Set to zero? Alternatives: Qe_te_deriv_electronscale
                Qi_n_deriv_electronscale   = 0*Qe_n_deriv_electronscale # Moose, is this correct? Set to zero? Alternatives: Qe_n_deriv_electronscale
            if engine.kinetic_electrons_electronscale == False:
                Qe_te_deriv_electronscale  = 0*Qi_ti_deriv_electronscale
                Qe_n_deriv_electronscale   = 0*Qi_n_deriv_electronscale
            if engine.two_species_electronscale == False:
                Qe_ti_deriv_electronscale  =  0*Qi_ti_deriv_electronscale
                Qi_te_deriv_electronscale  =  0*Qi_ti_deriv_electronscale

            # need to add neoclassical diffusion

            # save, this is what engine.compute_flux() writes
            zero = 0*Q0e_baseline_electronscale
            eps = 1e-8 + zero # want to avoid divide by zero

            engine.Gamma_electronscale  = pf.Flux_profile(eps  )
            engine.Qi_electronscale     = pf.Flux_profile(Q0i_baseline_electronscale)
            engine.Qe_electronscale     = pf.Flux_profile(Q0e_baseline_electronscale)
            engine.G_n_electronscale    = pf.Flux_profile(zero ) #???Moose
            engine.G_pi_electronscale   = pf.Flux_profile(zero )
            engine.G_pe_electronscale   = pf.Flux_profile(zero )
            engine.Qi_n_electronscale   = pf.Flux_profile(Qi_n_deriv_electronscale )
            engine.Qi_pi_electronscale  = pf.Flux_profile(Qi_ti_deriv_electronscale)
            engine.Qi_pe_electronscale  = pf.Flux_profile(Qi_te_deriv_electronscale)
            engine.Qe_n_electronscale   = pf.Flux_profile(Qe_n_deriv_electronscale)
            engine.Qe_pi_electronscale  = pf.Flux_profile(Qe_ti_deriv_electronscale)
            engine.Qe_pe_electronscale  = pf.Flux_profile(Qe_te_deriv_electronscale)

        if engine.ionscale_fluxtube == False:

            # save, this is what engine.compute_flux() writes
            zero = 0*Q0e_baseline_electronscale
            eps = 1e-8 + zero # want to avoid divide by zero

            engine.Gamma_ionscale  = pf.Flux_profile(eps  )
            engine.Qi_ionscale     = pf.Flux_profile(zero)
            engine.Qe_ionscale     = pf.Flux_profile(zero)
            engine.G_n_ionscale    = pf.Flux_profile(zero ) #???Moose
            engine.G_pi_ionscale   = pf.Flux_profile(zero )
            engine.G_pe_ionscale   = pf.Flux_profile(zero )
            engine.Qi_n_ionscale   = pf.Flux_profile(zero )
            engine.Qi_pi_ionscale  = pf.Flux_profile(zero)
            engine.Qi_pe_ionscale  = pf.Flux_profile(zero)
            engine.Qe_n_ionscale   = pf.Flux_profile(zero)
            engine.Qe_pi_ionscale  = pf.Flux_profile(zero)
            engine.Qe_pe_ionscale  = pf.Flux_profile(zero)

        if engine.electronscale_fluxtube == False:

            # save, this is what engine.compute_flux() writes
            zero = 0*Q0i_baseline_ionscale
            eps = 1e-8 + zero # want to avoid divide by zero

            engine.Gamma_electronscale  = pf.Flux_profile(eps  )
            engine.Qi_electronscale     = pf.Flux_profile(zero)
            engine.Qe_electronscale     = pf.Flux_profile(zero)
            engine.G_n_electronscale    = pf.Flux_profile(zero ) #???Moose
            engine.G_pi_electronscale   = pf.Flux_profile(zero )
            engine.G_pe_electronscale   = pf.Flux_profile(zero )
            engine.Qi_n_electronscale   = pf.Flux_profile(zero )
            engine.Qi_pi_electronscale  = pf.Flux_profile(zero )
            engine.Qi_pe_electronscale  = pf.Flux_profile(zero )
            engine.Qe_n_electronscale   = pf.Flux_profile(zero )
            engine.Qe_pi_electronscale  = pf.Flux_profile(zero )
            engine.Qe_pe_electronscale  = pf.Flux_profile(zero )
 
        # Moose, adding fluxes across flux tube scales. Easier to do elsewhere.
        #if (engine.ionscale_fluxtube == 'False') and (engine.electronscale_fluxtube == 'False'):

    #  sets up GX input, executes GX, returns input file name
    def gx_command(self, r_id, rho, kn, kti, kte, job_id, 
                         temp_i=1,temp_e=1, flux_tube_type = 'ionscale', kinetic_ions = True): # Moose
        # this version perturbs for the gradient
        # (temp, should be merged as option, instead of duplicating code)
        # job_id describes whether flux tube is ion or electron scale.
        
        ## this code used to get (kn,kp) as an input, now we take kT directly
        #s = rho**2
        #kti = kpi - kn
        #kte = kpe - kn

        t_id = self.t_id # time integer
        
        #.format(t_id, r_id, time, rho, s, kti, kn), file=f)
        if (self.engine.ionscale_fluxtube == True and self.engine.electronscale_fluxtube == True): ### If both ion and electron scale flux tubes!
            if flux_tube_type == 'ionscale':
                ft = self.flux_tubes[r_id] # Retrieve a single fluxtube for r_id. Ion scale flux tubes go first, so index is just r_id.
            else:
                N_fluxtubes = len(self.midpoints)
                ft = self.flux_tubes[r_id + N_fluxtubes] # Retrieve a single fluxtube for r_id. Electron scale flux tubes go second, so index is r_id + N_fluxtubes.
            print('Whoop! Launching both ion and electron scale flux tubes.')
        else:
            ft = self.flux_tubes[r_id] # Retrieve a single fluxtube for r_id.
        if kinetic_ions == False: # When setting gradients, ordering of kinetic ion and electron species in input files depends on whether ions are kinetic or not.
            ft.set_gradients(kn, kti, kte, kinetic_ions = False)
            ft.set_dens_temp(temp_i, temp_e, kinetic_ions = False)
        else:
            ft.set_gradients(kn, kti, kte) # Moose, we need to set the input file correctly! Sep 23 2022. If adiabatic ions, need to set correctly...
            ft.set_dens_temp(temp_i, temp_e)

        # Moose for now assume n_e = n_i.
        # Moose place for changing y0 whether ion or electron scale. In future, can add more complicated 
        # function to calculate y0_electronscale, for now, y0_electronscale = (rho_i/rho_e)y0_ionscale

        # Get the rescaled y0
        '''

        ------ 1) Model 1

        We use a simple model to choose y0 and the outer scale turbulence: ky_min rho_{s} = A (L_{Ts} / a), and so
        
        y0_s/rho_ref = (rho_s/rho_ref) 1/ky_min rho_s = (rho_s/rho_ref) (a/L_{Ts})/A

        For CBC, a/LTs = 2.49, y0 = 10, proton mass, so A = 1/4. HEALTH WARNING: assumes CBC uses protium, not deuterium. To check.

        Thus,

        y0_s /rho_ref= 4*(a/L_{Ts}) (rho_s/rho_ref) = 4*(a/L_{Ts}) sqrt(m_s T_s/ m_ref T_ref) 
        
        In GX normalized units, 
        
        y0_GX = y0_s /rho_ref = 4*tprim_GX * sqrt(m_{sGX} T_{sGX})

        ------ 2) Model 2

        y0_GX = y0_ionscale * sqrt(m_{sGX} T_{sGX})


        '''

        if flux_tube_type == 'ionscale':
            #mass_i = Collision_Model.m_mp[0] # Get ion mass. This is in proton masses.
            # Better to read gx input file and get the ion mass? However, for now, always assume gx input files use proton mass as reference mass.
            mass_i = self.engine.collision_model.m_mp[0]
            ft.set_fluxtube_scale(temp_i, mass_i, kti, y0model = 'CBC')
            ft.set_fluxtube_hyperviscosity(temp_i, mass_i,hyperviscousmodel = 'basic')
            ft.set_fluxtube_timescale(temp_i, mass_i, tmodel = 'basic')

        if flux_tube_type == 'electronscale':
            #mass_e = Collision_Model.m_mp[1] # Get electron mass.
            mass_e = self.engine.collision_model.m_mp[1] # Get electron mass.
            ft.set_fluxtube_scale(temp_e, mass_e, kte, y0model = 'CBC')
            ft.set_fluxtube_hyperviscosity(temp_e, mass_e,hyperviscousmodel = 'basic')
            ft.set_fluxtube_timescale(temp_e, mass_e, tmodel = 'basic')
            # Set fluxtube hyperviscosity?

        # to be specified by Trinity input file, or by time stamp
        #root = 'gx-files/'
        path = self.path
        tag  = 't{:02}-r{:}-{:}'.format(t_id, r_id, job_id)

        fout  = tag + '.in'
        fsave = tag + '-restart.nc'

        ### Decide whether to load restart
        if (t_id == 0): 
            # skip only for first time step
            ft.gx_input.inputs['Restart']['restart'] = 'false'
        else:
            ft.gx_input.inputs['Restart']['restart'] = 'true'
            fload = f"restarts/t{t_id-1}-r{r_id}-{job_id}save.nc"
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + fload)
            ft.gx_input.inputs['Controls']['init_amp'] = '0.0'
            # restart from the same file (prev time step), to ensure parallelizability

        #### save restart file (always)
        ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
        fsave = 'restarts/t{:}-r{:}-{:}save.nc'.format(t_id, r_id, job_id)
        ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + fsave)

        ### execute
        ft.gx_input.write(path + fout) # Moose variables live in gx_input
        gx_filename = self.run_gx(tag, path) # this returns a file name
        print('In GX_command, gx_filename is {}'.format(gx_filename))
        return gx_filename


    def run_gx(self,tag,path):

        f_nc = path + tag + '.nc'
        if ( os.path.exists(f_nc) == False ):

            # attempt to call
            system = os.environ['GK_SYSTEM']

            cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', path+'gx', path+tag+'.in'] # stellar
            if system == 'traverse':
                # traverse does not recognize path/to/gx as an executable
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', 'gx', path+tag+'.in'] # traverse
    
            print('Calling', tag)
            print_time()
            f_log = path + 'log.' +tag
            with open(f_log, 'w') as fp:

                print('   running:', tag)
                p = subprocess.Popen(cmd, stdout=fp) # Popen permits running parallel jobs.
                self.processes.append(p)
        else:
            print('  gx output {:} already exists'.format(tag) )

        return f_nc # this is a file name

        
    # first attempt at exporting gradients for GX
    def write_command(self, r_id, rho, kn, kpi, kpe):
        
        s = rho**2
        kti = kpi - kn

        t_id = self.t_id # time integer
        time = self.time # time [s]

        f = self.f_handle

        #print('t_idx, r_idx, time, r, s, tprim, fprim', file=f)
        print('{:d}, {:d}, {:.2e}, {:.4e}, {:.4e}, {:.6e}, {:.6e}' \
        .format(t_id, r_id, time, rho, s, kti, kn), file=f)

###
def print_time():

    dt = datetime.now()
    #ts = datetime.timestamp(dt)
    #print('  time', ts)
    print('  time:', dt)

# double the inside point (no fluxtube run there)
### unused
#def array_cat(arr):
#    return np.concatenate( [ [arr[0]] , arr ] )

# read a GX netCDF output file, returns HEAT flux only: Moose, does it return for two species?
# Moose: for two species, also add particle flux
def return_gx_heat_flux(f_nc, species_number = 0): # species_number = 0 for ions, 1 for electrons
    #try:
        print('f_nc is {}'.format(f_nc))
        qflux = gx_io.read_GX_qflux_output( f_nc , species_number)
        if ( np.isnan(qflux).any() ):
             print('  nans found in', f_nc, '(setting NaNs to 0)')
             qflux = np.nan_to_num(qflux)

        tag = f_nc.split('/')[-1]
        print('  {:} qflux: {:}'.format(tag, qflux))
        return qflux

    #except:
    #    print('  issue reading heat flux', f_nc)
    #    return 0 # for safety, this will be problematic


# read a GX netCDF output file, returns PARTICLE flux only: Moose, does it return for two species?
# Moose: for two species, also add particle flux
def return_gx_particle_flux(f_nc):
    try:
        qflux = gx_io.read_GX_pflux_output( f_nc )
        if ( np.isnan(qflux).any() ):
             print('  nans found in', f_nc, '(setting NaNs to 0)')
             qflux = np.nan_to_num(qflux)

        tag = f_nc.split('/')[-1]
        print('  {:} pflux: {:}'.format(tag, pflux))
        return qflux

    except:
        print('  issue reading particle flux', f_nc)
        return 0 # for safety, this will be problematic


# read a GX netCDF output file, returns MOMENTUM flux only: Moose, does it return for two species?
# Moose: for two species, also add particle flux
def return_gx_momentum_flux(f_nc):
    try:
        qflux = gx_io.read_GX_mflux_output( f_nc )
        if ( np.isnan(qflux).any() ):     
             print('  nans found in', f_nc, '(setting NaNs to 0)')
             qflux = np.nan_to_num(qflux)

        tag = f_nc.split('/')[-1]
        print('  {:} mflux: {:}'.format(tag, mflux))
        return qflux

    except:
        print('  issue reading momentum flux', f_nc)
        return 0 # for safety, this will be problematic

