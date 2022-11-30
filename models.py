import numpy as np
import subprocess
from datetime import datetime
import time as _time

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner, GX_Output

# read GX output
import trinity_lib as trl
from profiles import Profile, Flux_profile
from GX_io import GX_Output, VMEC_GX_geometry_module
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
        engine.Gamma  = Flux_profile(zero  )
        engine.Qi     = Flux_profile(Qi    ) 
        engine.Qe     = Flux_profile(Qe    ) 
        engine.G_n    = Flux_profile(zero  )
        engine.G_pi   = Flux_profile(zero  )
        engine.G_pe   = Flux_profile(zero  )
        engine.Qi_n   = Flux_profile(zero  )
        engine.Qi_pi  = Flux_profile(dQi_pi)
        engine.Qi_pe  = Flux_profile(zero  )
        engine.Qe_n   = Flux_profile(zero  )
        engine.Qe_pi  = Flux_profile(zero  )
        engine.Qe_pe  = Flux_profile(dQe_pe)


WAIT_TIME = 1  # this should come from the Trinity Engine
class GX_Flux_Model():

    def __init__(self, engine, 
                       gx_root='gx-files/', 
                       gx_template='gx-sample.in',
                       path='run-dir/', 
                       vmec_path='./',
                       vmec_wout="",
                       midpoints=[]
                ):

        self.engine = engine

        f_geo   = 'gx-geometry-sample.ing' # sample input file, to be part of repo

        ### Check file path
        print("  Looking for GX files")
        print("    GX input path:", gx_root)
        print("      expecting GX template:", gx_root + gx_template)
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
        self.input_template = GX_Runner(gx_root + gx_template)
        self.path = path # this is the GX output path (todo: rename)
        # check that path exists, if it does not, mkdir and copy gx executable
        
        self.midpoints = midpoints
        self.vmec_path = vmec_path
        self.vmec_wout = vmec_wout
        self.gx_root   = gx_root
        self.gx_template = gx_template
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

        ### load flux tube geometry
        # these should come from Trinity input file
        
        vmec = self.vmec_wout
        if vmec: # is not blank

            # else launch flux tubes from VMEC
            f_geo     = self.f_geo
            geo_path  = self.gx_root  # this says where the convert executable lives, and where to find the sample .ing file
            out_path  = self.path
            vmec_path = self.vmec_path

            geo_template = VMEC_GX_geometry_module( self.engine,
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
            for j in np.arange(N_fluxtubes):
                rho = self.midpoints[j]
                f_geometry = geo_template.init_radius(rho,j) 
                geo_files.append(out_path + f_geometry)

        else:
            # load default files (assumed to be existing)
            # 10/6 we should delete this soon, use nested circles (or even a fresh wout) as default instead
            print('  no VMEC wout given, loading default files')
            geo_files = [ 'gx-files/gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.295_gds21_nt_38_geo.nc',  
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.500_gds21_nt_40_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.704_gds21_nt_42_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.897_gds21_nt_42_geo.nc']

        print("")

        ### store flux tubes in a list
        self.flux_tubes = []
        for fin in geo_files:
            self.load_fluxtube(fin)


        # make a directory for restart files
        restart_dir = self.path + 'restarts' # this name is hard-coded to agree with that in gx_command(), a better name may be restart_dir/ or a variable naming such
        if os.path.exists(restart_dir) == False:
            os.mkdir(restart_dir)


    # is this being used? 9/28
    # I think the functionality got implemented somewhere else
    def create_geometry_from_vmec(self,wout):

        # load sample geometry file

        # make changes (i.e. specify radius, ntheta etc.)

        # write new file and run GX-VMEC geometry module

        # wait, load new geometry files
        pass

    def load_fluxtube(self, f_geo):

        ft = FluxTube(f_geo)       # init an instance of flux tube class
        ft.load_gx_input(self.input_template)

        # save
        self.flux_tubes.append(ft)


    def prep_commands(self, engine, # pointer to pull profiles from trinity engine
                            step = 0.3, # relativestep size for perturbing gradients
                            abs_step = 0.3, # absolute step size for perturbing gradients
                     ):

        self.time = engine.time
        self.t_id = engine.t_idx

        # preparing dimensionless (tprim = L_ref/LT) for GX
        Ln  = - engine.density.grad_log   .profile  # a / L_n
        Lpi = - engine.pressure_i.grad_log.profile  # a / L_pi
        Lpe = - engine.pressure_e.grad_log.profile  # a / L_pe

        LTi = Lpi - Ln
        LTe = Lpe - Ln

        # get normalizations for GX, dens and temp
        n = engine.density.midpoints
        pi = engine.pressure_i.midpoints
        pe = engine.pressure_e.midpoints

        Ti = pi/n
        Te = pe/n

## this is one way to do it, using reference temperatures
#  would it be more clear to just have keV values in absolute?
        Tref = Ti # hard-coded convention
        GX_Ti = Ti/Tref
        GX_Te = Te/Tref

        # get reference values for GX
        p_cgs = pi*1e17                                                   
        try:
            B_cgs = engine.flux_norms.B_ref*1e4
        except:
            B_cgs = engine.Ba*1e4
        beta_ref = 4.03e-11*p_cgs/(B_cgs*B_cgs) 
        vtref_gx = (Tref*engine.norms.e/engine.norms.m_ref/2)**0.5  # GX vt does not include sqrt(2). assumes deuterium reference ions. need to generalize.
        try:
            # this should really be flux_norms.a_ref, but we are assuming trinity and GX use same a_ref  
            a_ref = flux_norms.a_ref
        except:
            a_ref = engine.norms.a_ref
        tref_gx = a_ref / vtref_gx 

        # get collision freq for GX
        coll = engine.collision_model
        coll.update_profiles(engine)
        nu_ii = Profile(coll.collision_frequency(0),half=True).midpoints * tref_gx
        nu_ee = Profile(coll.collision_frequency(1),half=True).midpoints * tref_gx

        # get turbulent flux at each radial flux tube
        mid_axis = engine.mid_axis
        idx = np.arange( len(mid_axis) ) 

        f0   = [''] * len(idx) 
        fn   = [''] * len(idx) 
        fpi  = [''] * len(idx) 
        fpe  = [''] * len(idx) 

# delete
#        Q0   = np.zeros( len(idx) )
#        Qpi  = np.zeros( len(idx) )
#        Qpe  = np.zeros( len(idx) )
#        Qn   = np.zeros( len(idx) )

        Gamma = np.zeros_like(idx, dtype=float)
        Qi    = np.zeros_like(idx, dtype=float)
        Qe    = np.zeros_like(idx, dtype=float)
        # underscore indicates derivative: G_n = d Gamma / d(a kn)
        G_n   = np.zeros_like(idx, dtype=float)
        G_pi  = np.zeros_like(idx, dtype=float)
        G_pe  = np.zeros_like(idx, dtype=float)
        Qi_n  = np.zeros_like(idx, dtype=float)
        Qi_pi = np.zeros_like(idx, dtype=float)
        Qi_pe = np.zeros_like(idx, dtype=float)
        Qe_n  = np.zeros_like(idx, dtype=float)
        Qe_pi = np.zeros_like(idx, dtype=float)
        Qe_pe = np.zeros_like(idx, dtype=float)
        # use ones like here to avoid divide by 0 (will be overwritten if used)
        dkn   =  np.ones_like(idx, dtype=float)
        dkti  =  np.ones_like(idx, dtype=float)
        dkte  =  np.ones_like(idx, dtype=float)

        # attempt to load previous values, to determine whether to use restart
        try:
            Qi_prev = engine.Qi.profile
            Qi_n_prev = engine.Qi_n.profile
            Qi_pi_prev = engine.Qi_pi.profile
            Qi_pe_prev = engine.Qi_pe.profile
        except:
            Qi_prev    = np.zeros(len(idx))
            Qi_n_prev  = np.zeros(len(idx))
            Qi_pi_prev = np.zeros(len(idx))
            Qi_pe_prev = np.zeros(len(idx))

        eps = 1e-10 
        restart_0  = abs(Qi_prev) > eps
        restart_pi = abs(Qi_pi_prev) > eps
        restart_pe = abs(Qi_pe_prev) > eps
        restart_n  = abs(Qi_n_prev) > eps

        scale = 1 + step
        for j in idx: 

            rho = mid_axis[j]
            kn  = Ln [j]
            kti = LTi[j]
            kte = LTe[j]

            gx_ti = GX_Ti[j]
            gx_te = GX_Te[j]
            beta  = beta_ref[j]
            nu_i  = nu_ii[j]
            nu_e  = nu_ee[j]

            res_0 = restart_0[j]
            res_pi = restart_pi[j]
            res_pe = restart_pe[j]
            res_n = restart_n[j]

            # writes the GX input file and calls the slurm 
            f0[j] = self.gx_command(j, rho, kn      , kti      , kte      , '0', 
                          temp_i=gx_ti, temp_e = gx_te, nu_ii = nu_i, nu_ee = nu_e, 
                          restart=res_0, beta_ref= beta )

            if engine.evolve_temperature:

                if not engine.fix_ions and not engine.adiabatic_species == "ion":
                    kti_pert = max(kti*scale, kti + abs_step)
                    dkti[j] = kti_pert - kti
                    fpi[j] = self.gx_command(j, rho, kn      , kti_pert , kte      , '2', 
                             temp_i=gx_ti, temp_e = gx_te, nu_ii = nu_i, nu_ee = nu_e, 
                             restart=res_pi, beta_ref= beta )

                if not engine.fix_electrons and not engine.adiabatic_species == "electron":
                    kte_pert = max(kte*scale, kte + abs_step)
                    dkte[j] = kte_pert - kte
                    fpe[j] = self.gx_command(j, rho, kn      , kti      , kte_pert , '3', 
                             temp_i=gx_ti, temp_e = gx_te, nu_ii = nu_i, nu_ee = nu_e, 
                             restart=res_pe, beta_ref= beta)

            if engine.evolve_density:
                # perturb density gradient at fixed pressure gradient
                kn_pert = kn + abs_step
                kti_pert = kti - abs_step
                kte_pert = kte - abs_step
                dkn[j] = kn_pert - kn
                fn [j] = self.gx_command(j, rho, kn_pert , kti_pert , kte_pert , '1', 
                         temp_i=gx_ti, temp_e = gx_te, nu_ii = nu_i, nu_ee = nu_e, 
                         restart=res_n, beta_ref= beta )

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)
        print('GX runs complete: starting to read')

        # collect normalizations
        grho = []
        area = []
        B_ref = []
        a_ref = []

        '''
        In this variable notation

        T is temp gradient scale length
        delta is the step size

        Q0   is the base case                          : Q(T)
        Qpi  is array of fluxes at pi perturbation     : Q(T + delta)
        Q_pi is array of derivatives of flux by step   : dQ/delta
        '''
 
        Gamma, Qi   , Qe    = read_gx_array(f0 )
        G_pi , Qi_pi, Qe_pi = read_gx_array(fpi)
        G_pe , Qi_pe, Qe_pe = read_gx_array(fpe)
        G_n  , Qi_n , Qe_n  = read_gx_array(fn )

#        for j in idx: 
#            # loop over flux tubes
#
#
###            Gamma[j], Qi[j]   , Qe[j]    = read_gx( f0[j]  )
###            G_pi[j] , Qi_pi[j], Qe_pi[j] = read_gx( fpi[j] )
###            G_pe[j] , Qi_pe[j], Qe_pe[j] = read_gx( fpe[j] )
###            G_n[j]  , Qi_n[j] , Qe_n[j]  = read_gx( fn[j]  )
#
#
#            gx_data = read_gx(f0 [j])
##            Gamma[j] = gx_data.pflux
##            Qi[j] = gx_data.qflux_i
##            Qe[j] = gx_data.qflux_e
##
##            gx_data = read_gx(fpi [j])
##            G_pi[j] = gx_data.pflux
##            Qi_pi[j] = gx_data.qflux_i
##            Qe_pi[j] = gx_data.qflux_e
##
##            gx_data = read_gx(fpe [j])
##            G_pe[j] = gx_data.pflux
##            Qi_pe[j] = gx_data.qflux_i
##            Qe_pe[j] = gx_data.qflux_e
##
##            gx_data = read_gx(fn [j])
##            G_n[j] = gx_data.pflux
##            Qi_n[j] = gx_data.qflux_i
##            Qe_n[j] = gx_data.qflux_e
##
#            a_ref.append(gx_data.a_ref)
#            B_ref.append(gx_data.B_ref)
#            grho.append(gx_data.grhoavg)
#            area.append(gx_data.surfarea)
#
#        import pdb
#        pdb.set_trace()
#        grho = np.asarray(grho)
#        area = np.asarray(area)
#        B_ref = np.asarray(B_ref)
#        a_ref = np.asarray(a_ref)
#        # get flux-tube normalizing and geometric quantities on same grid as fluxes
#        engine.flux_norms.B_ref = Flux_profile(B_ref)
#        engine.flux_norms.a_ref = Flux_profile(a_ref)
#        engine.flux_norms.grho  = Flux_profile(grho)
#        engine.flux_norms.area  = Flux_profile(area)
#        engine.flux_norms.geometry_factor = Flux_profile(- grho / (engine.drho * area))

        # record dflux / dLx
        dG_n   =  (G_n -  Gamma) / dkn
        dG_pi  =  (G_pi - Gamma) / dkti
        dG_pe  =  (G_pe - Gamma) / dkte

        dQi_n  =  (Qi_n - Qi) / dkn
        dQi_pi  =  (Qi_pi - Qi) / dkti
        dQi_pe  =  (Qi_pe - Qi) / dkte

        dQe_n  =  (Qe_n - Qe) / dkn
        dQe_pi  =  (Qe_pi - Qe) / dkti
        dQe_pe  =  (Qe_pe - Qe) / dkte

        # record the heat flux (temp diagnostic)
        rec = engine.record_flux
        rec['Q0'].append( Qi       )
        rec['Q1'].append( Qi_pi      )
        rec['dQ'].append( dQi_pi    )
        rec['kT'].append( LTi      )
        rec['dk'].append( dkti )

        # need to add neoclassical diffusion

        # save, this is what engine.compute_flux() writes
        engine.Gamma  = Flux_profile(Gamma)
        engine.Qi     = Flux_profile(Qi) 
        engine.Qe     = Flux_profile(Qe) 
        engine.G_n    = Flux_profile(dG_n)
        engine.G_pi   = Flux_profile(dG_pi)
        engine.G_pe   = Flux_profile(dG_pe)
        engine.Qi_n   = Flux_profile(dQi_n )
        engine.Qi_pi  = Flux_profile(dQi_pi)
        engine.Qi_pe  = Flux_profile(dQi_pe)
        engine.Qe_n   = Flux_profile(dQe_n )
        engine.Qe_pi  = Flux_profile(dQe_pi)
        engine.Qe_pe  = Flux_profile(dQe_pe)

    #  sets up GX input, executes GX, returns input file name
    def gx_command(self, r_id, rho, kn, kti, kte, job_id, 
                         temp_i=1,temp_e=1, nu_ii=0, nu_ee=0, beta_ref=0, restart=True):
        
        t_id = self.engine.t_idx # time integer
        p_id = self.engine.p_idx # Newton iteration number
        prev_p_id = self.engine.prev_p_id # Newton iteration number

        ft = self.flux_tubes[r_id] 
        ft.set_profiles(temp_i, temp_e, nu_ii, nu_ee, beta_ref)
        ft.set_gradients(kn, kti, kte)
        
        # to be specified by Trinity input file, or by time stamp
        #root = 'gx-files/'
        path = self.path
        tag  = f"t{t_id:02}-p{p_id}-r{r_id}-{job_id}"
        '''
        job_id needs a better name
        It currently refers to {base, pi, pe, n} perturbations
        It can also refer to ion scale or electron scale flux tubes
        '''

        fout  = tag + '.in'
        f_save = tag + '-restart.nc'

        ### Decide whether to load restart
        if (not restart or t_id == 0): 
            ft.gx_input.inputs['Restart']['restart'] = 'false'
            ft.gx_input.inputs['Initialization']['init_amp'] = '1e-3'

        else:
            ft.gx_input.inputs['Restart']['restart'] = 'true'
            f_load = f"restarts/saved-t{t_id-1:02d}-p{prev_p_id}-r{r_id}-{job_id}.nc" 
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + f_load)
            ft.gx_input.inputs['Initialization']['init_amp'] = '0.0'

        
        #### save restart file (always)
        ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
        f_save = f"restarts/saved-t{t_id:02d}-p{p_id}-r{r_id}-{job_id}.nc"
        ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + f_save)


        ### execute
        ft.gx_input.write(path + fout)
        fname = self.run_gx(tag, path) # this returns a file name
        return fname


    def run_gx(self,tag,path):

        f_nc = path + tag + '.nc'
        if ( os.path.exists(f_nc) == False ):

            # attempt to call
            system = os.environ['GK_SYSTEM']

            cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', '--exclusive', path+'gx', path+tag+'.in'] # stellar
            if system == 'traverse':
                # traverse does not recognize path/to/gx as an executable
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', 'gx', path+tag+'.in'] # traverse
            if system == 'satori':
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gres=gpu:1', path+'gx', path+tag+'.in'] # satori
    
            print('Calling', tag)
            print_time()
            f_log = path + 'log.' +tag
            with open(f_log, 'w') as fp:

                print('   running:', tag)
                p = subprocess.Popen(cmd, stdout=fp)
                self.processes.append(p)

            # tally completed GX run
            self.engine.gx_idx += 1 

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
    print('  time:', dt)
    #ts = datetime.timestamp(dt)
    #print('  time', ts)


def read_gx_array(f_array, eps=0):
    '''
    Reads an array of gx netcdf outputs
        (interpretted as one per radial location)

    Extracts Gamma, Qi, Qe from each file
    Returns an radial point array for each flux
    '''

    pflux = []
    qflux_i = []
    qflux_e = []

    for f in f_array:

        if f == '':
            pflux.append(eps)
            qflux_i.append(eps)
            qflux_e.append(eps)
            continue

        gx_data = GX_Output(f)
        pflux  .append(gx_data.pflux  )
        qflux_i.append(gx_data.qflux_i)
        qflux_e.append(gx_data.qflux_e)

    pflux = np.array(pflux)
    qflux_i = np.array(qflux_i)
    qflux_e = np.array(qflux_e)
    return pflux, qflux_i, qflux_e

        
# old can be deleted
def read_gx(f_nc):
    if f_nc == '':
        return 0, 0, 0

    # read a GX netCDF output file, returns flux
    try:
        gx_data = GX_Output(f_nc)

#        return gx_data
        return np.array( [gx_data.qflux_i, gx_data.qflux_e, gx_data.pflux] )

    except:
        print('  issue reading', f_nc)
#        exit(1)
        return 0, 0, 0

    tag = f_nc.split('/')[-1]
    print('  {:} pflux: {:}, qflux_i: {:}, qflux_e: {:}'.format(tag, gx_data.pflux, gx_data.qflux_i, gx_data.qflux_e))

def read_gx_old(f_nc):
    # read a GX netCDF output file, returns flux
    try:

        gx = GX_Output(f_nc)
        qflux = gx.qflux
        pflux = gx.plfux # in construction: this gives profile, not estimate

        q_median = gx.median_estimator(qflux)
        p_median = gx.median_estimator(pflux)

        tag = f_nc.split('/')[-1]
        print(f"  {tag} qflux, pflux: {q_median}, {p_median}")
        return q_median

    except:
        print('  issue reading', f_nc)
        return 0 # for safety, this will be problematic

