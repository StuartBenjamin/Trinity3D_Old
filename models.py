import numpy as np
import subprocess
from datetime import datetime
import time as _time

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner

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
        engine.dG_n    = pf.Flux_profile(zero  )
        engine.dG_pi   = pf.Flux_profile(zero  )
        engine.dG_pe   = pf.Flux_profile(zero  )
        engine.dQi_n   = pf.Flux_profile(zero  )
        engine.dQi_pi  = pf.Flux_profile(dQi_pi)
        engine.dQi_pe  = pf.Flux_profile(zero  )
        engine.dQe_n   = pf.Flux_profile(zero  )
        engine.dQe_pi  = pf.Flux_profile(zero  )
        engine.dQe_pe  = pf.Flux_profile(dQe_pe)


WAIT_TIME = 1  # this should come from the Trinity Engine
class GX_Flux_Model():

    def __init__(self, engine, 
                       gx_root='', 
                       gx_template='gx-sample.in',
                       path='run-dir/', 
                       eq_path='',
                       eq_file='',
                       eq_model='VMEC',
                       midpoints=[]
                ):

        self.engine = engine

#        gx_root = "gx-files/"  # this is part of repo, don't change # old, 9/7
        #f_input = 'gx-sample.in'  # removed 10/16
        f_geo   = 'gx-geometry-sample.ing' # sample input file, to be part of repo

        GX_PATH = os.environ.get("GX_PATH") 

        ### Check file path
        print("  Looking for GX files")
        print("    GX input path:", gx_root)
        print("      expecting GX template:", gx_root + gx_template)
        print("      expecting GX executable:", GX_PATH + "gx")
        if(eq_model == "VMEC"):
            print("      expecting GX-VMEC template:", gx_root + f_geo)
            print("      expecting GX-VMEC executable:", gx_binpath + "convert_VMEC_to_GX")
            print("    VMEC path:", eq_path)
            print("      expecting VMEC wout:", eq_path + eq_file)
        print("    GX-Trinity output path:", path)

        found_path = os.path.exists(path)
        if (found_path == False):
            print(f"      creating new output dir {path}")
            os.mkdir(path)

        found_gx = os.path.exists(GX_PATH+"gx")
        if (found_gx == False):
            print("  Error: gx executable not found! Make sure the GX_PATH environment variable is set.")
            exit(1)

        print("")


        # check using something like
        # os.listdir(eq_path).find(eq_file)


        ###  load an input template
        #    later, this should come from Trinity input file
        self.input_template = GX_Runner(gx_root + gx_template)
        self.path = path # this is the GX output path (todo: rename)
        # check that path exists, if it does not, mkdir and copy gx executable
        
        self.midpoints = midpoints
        self.eq_path = eq_path
        self.eq_file = eq_file
        self.eq_model = eq_model
        self.gx_template = gx_template
        self.gx_root = gx_root
        self.f_geo   = f_geo # template convert geometry input

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
        N_fluxtubes = len(self.midpoints)
        self.flux_tubes = []
        
        if self.eq_model == "VMEC": # is not blank

            # else launch flux tubes from VMEC
            vmec = self.eq_file
            f_geo     = self.f_geo
            geo_path  = self.gx_root  # this says where the convert executable lives, and where to find the sample .ing file
            out_path  = self.path
            eq_path = self.eq_path

            geo_template = gx_io.VMEC_GX_geometry_module( self.engine,
                                                 f_sample = f_geo,
                                                 input_path = geo_path,
                                                 output_path = out_path,
                                                 tag = vmec[5:-3]
                                              )
            geo_template.set_vmec( vmec, 
                          eq_path   = eq_path, 
                          output_path = out_path )

            geo_files = []
            for j in np.arange(N_fluxtubes):
                rho = self.midpoints[j]
                f_geometry = geo_template.init_radius(rho,j) 
                geo_files.append(out_path + f_geometry)

            ### store flux tubes in a list
            for fin in geo_files:
                self.load_fluxtube(f_geo=fin)

        elif self.eq_model == "geqdsk":
            for j in np.arange(N_fluxtubes):
                rho = self.midpoints[j]
                self.load_fluxtube(rho=rho)

        else:
            print("ERROR: eq_model = {self.eq_model} not recognized.")
            exit(1)

        print("")

        # make a directory for restart files
        restart_dir = self.path + 'restarts' # this name is hard-coded to agree with that in gx_command(), a better name may be restart_dir/ or a variable naming such
        if os.path.exists(restart_dir) == False:
            os.mkdir(restart_dir)


    def load_fluxtube(self, rho=None, f_geo=None):

        ft = FluxTube(self.input_template, rho=rho, f_geo=f_geo)       # init an instance of flux tube class

        # save
        self.flux_tubes.append(ft)


    def prep_commands(self, engine, # pointer to pull profiles from trinity engine
                            step = 0.2, # relative step size for perturbing gradients
                            abs_step = 0.3, # absolute size for perturbing gradients
                            overwrite = False,
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
        gx_Ti = Ti/Tref
        gx_Te = Te/Tref

        vtref = (Tref*engine.norms.e/engine.norms.m_ref/2)**0.5  # GX vt does not include sqrt(2). this assumes deuterium reference ions. need to generalize.
        coll = engine.collision_model
        coll.update_profiles(engine)
        nu_ii = pf.Profile(coll.collision_frequency(0),half=True).midpoints*engine.norms.a_ref/vtref  # this should really be flux_norms.a_ref, but we are assuming trinity and GX use same a_ref  
        nu_ee = pf.Profile(coll.collision_frequency(1),half=True).midpoints*engine.norms.a_ref/vtref  # this should really be flux_norms.a_ref, but we are assuming trinity and GX use same a_ref

        # compute reference beta
        p_cgs = pi*1e17                                                   
        try:
            B_cgs = engine.flux_norms.B_ref*1e4
        except:
            B_cgs = engine.Ba*1e4
        beta_ref = 4.03e-11*p_cgs/(B_cgs*B_cgs) 

        # turbulent flux calls, for each radial flux tube
        mid_axis = engine.mid_axis
        idx = np.arange( len(mid_axis) ) 

        f0   = [''] * len(idx) 
        fn   = [''] * len(idx) 
        fpi  = [''] * len(idx) 
        fpe  = [''] * len(idx) 

        Gamma  = np.zeros(len(idx))
        Qi     = np.zeros(len(idx))
        Qe     = np.zeros(len(idx))
        G_n    = np.zeros(len(idx))
        G_pi   = np.zeros(len(idx))
        G_pe   = np.zeros(len(idx))
        Qi_n   = np.zeros(len(idx))
        Qi_pi  = np.zeros(len(idx))
        Qi_pe  = np.zeros(len(idx))
        Qe_n   = np.zeros(len(idx))
        Qe_pi  = np.zeros(len(idx))
        Qe_pe  = np.zeros(len(idx))
        dkn  = np.zeros( len(idx) )
        dkti  = np.zeros( len(idx) )
        dkte  = np.zeros( len(idx) )

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

        for j in idx: 

            rho = mid_axis[j]
            kn  = Ln [j]
            kti = LTi[j]
            kte = LTe[j]

            # writes the GX input file and calls the slurm 
            scale = 1 + step

            restart_0 = Qi_prev[j] > 1e-10 and Qi_prev[j] < 1e2
            f0 [j] = self.gx_command(j, rho, kn      , kti      , kte      , '0', temp_i=gx_Ti[j], temp_e = gx_Te[j], nu_ii = nu_ii[j], nu_ee = nu_ee[j], restart=restart_0, beta_ref= beta_ref[j], overwrite=overwrite )

            if engine.evolve_density:
                restart_n = Qi_n_prev[j] > 1e-10 and Qi_n_prev[j] < 1e2
                # perturb density gradient at fixed pressure gradient
                kn_pert = kn + abs_step
                kti_pert = kti - abs_step
                kte_pert = kte - abs_step
                dkn[j] = kn_pert - kn
                fn [j] = self.gx_command(j, rho, kn_pert , kti_pert , kte_pert , '1', temp_i=gx_Ti[j], temp_e = gx_Te[j], nu_ii = nu_ii[j], nu_ee = nu_ee[j], restart=restart_n, beta_ref= beta_ref[j], overwrite=overwrite )

            if engine.evolve_temperature:
                if not engine.fix_ions and not engine.adiabatic_species == "ion":
                    restart_pi = Qi_pi_prev[j] > 1e-10 and Qi_pi_prev[j] < 1e2
                    kti_pert = max(kti*(1+step), kti + abs_step)
                    dkti[j] = kti_pert - kti
                    fpi[j] = self.gx_command(j, rho, kn      , kti_pert , kte      , '2', temp_i=gx_Ti[j], temp_e = gx_Te[j], nu_ii = nu_ii[j], nu_ee = nu_ee[j], restart=restart_pi, beta_ref= beta_ref[j], overwrite=overwrite )
                if not engine.fix_electrons and not engine.adiabatic_species == "electron":
                    restart_pe = Qi_pe_prev[j] > 1e-10 and Qi_pe_prev[j] < 1e2
                    kte_pert = max(kte*(1+step), kte + abs_step)
                    dkte[j] = kte_pert - kte
                    fpe[j] = self.gx_command(j, rho, kn      , kti      , kte_pert , '3', temp_i=gx_Ti[j], temp_e = gx_Te[j], nu_ii = nu_ii[j], nu_ee = nu_ee[j], restart=restart_pe , beta_ref= beta_ref[j], overwrite=overwrite)

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)
        grho = []
        area = []
        B_ref = []
        a_ref = []

        print('starting to read')
        for j in idx: 
            # loop over flux tubes
            gx_data = read_gx(f0 [j])
            Gamma[j] = gx_data.pflux
            Qi[j] = gx_data.qflux_i
            Qe[j] = gx_data.qflux_e

            gx_data = read_gx(fn [j])
            G_n[j] = gx_data.pflux
            Qi_n[j] = gx_data.qflux_i
            Qe_n[j] = gx_data.qflux_e

            gx_data = read_gx(fpi [j])
            G_pi[j] = gx_data.pflux
            Qi_pi[j] = gx_data.qflux_i
            Qe_pi[j] = gx_data.qflux_e

            gx_data = read_gx(fpe [j])
            G_pe[j] = gx_data.pflux
            Qi_pe[j] = gx_data.qflux_i
            Qe_pe[j] = gx_data.qflux_e

            a_ref.append(gx_data.a_ref)
            B_ref.append(gx_data.B_ref)
            grho.append(gx_data.grhoavg)
            area.append(gx_data.surfarea)

        grho = np.asarray(grho)
        area = np.asarray(area)
        B_ref = np.asarray(B_ref)
        a_ref = np.asarray(a_ref)

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

        # need to add neoclassical diffusion

        engine.Gamma  = pf.Flux_profile(Gamma)
        engine.Qi     = pf.Flux_profile(Qi) 
        engine.Qe     = pf.Flux_profile(Qe) 
        engine.G_n    = pf.Flux_profile(G_n)
        engine.G_pi   = pf.Flux_profile(G_pi)
        engine.G_pe   = pf.Flux_profile(G_pe)
        engine.Qi_n   = pf.Flux_profile(Qi_n )
        engine.Qi_pi  = pf.Flux_profile(Qi_pi)
        engine.Qi_pe  = pf.Flux_profile(Qi_pe)
        engine.Qe_n   = pf.Flux_profile(Qe_n )
        engine.Qe_pi  = pf.Flux_profile(Qe_pi)
        engine.Qe_pe  = pf.Flux_profile(Qe_pe)

        engine.dG_n    = pf.Flux_profile(dG_n)
        engine.dG_pi   = pf.Flux_profile(dG_pi)
        engine.dG_pe   = pf.Flux_profile(dG_pe)
        engine.dQi_n   = pf.Flux_profile(dQi_n )
        engine.dQi_pi  = pf.Flux_profile(dQi_pi)
        engine.dQi_pe  = pf.Flux_profile(dQi_pe)
        engine.dQe_n   = pf.Flux_profile(dQe_n )
        engine.dQe_pi  = pf.Flux_profile(dQe_pi)
        engine.dQe_pe  = pf.Flux_profile(dQe_pe)

        # get flux-tube normalizing and geometric quantities on same grid as fluxes
        engine.flux_norms.B_ref = pf.Flux_profile(B_ref)
        engine.flux_norms.a_ref = pf.Flux_profile(a_ref)
        engine.flux_norms.grho = pf.Flux_profile(grho)
        engine.flux_norms.area = pf.Flux_profile(area)
        engine.flux_norms.geometry_factor = pf.Flux_profile(- grho / (engine.drho * area))

    #  sets up GX input, executes GX, returns input file name
    def gx_command(self, r_id, rho, kn, kti, kte, job_id, 
                         temp_i=1,temp_e=1, nu_ii=0, nu_ee=0, beta_ref=0, restart=True, overwrite=False):
        # this version perturbs for the gradient
        # (temp, should be merged as option, instead of duplicating code)
        
        ## this code used to get (kn,kp) as an input, now we take kT directly
        #s = rho**2
        #kti = kpi - kn
        #kte = kpe - kn

        #t_id = self.t_id # time integer
        t_id = self.engine.t_idx # time integer
        p_id = self.engine.p_idx # Newton iteration number
        prev_p_id = self.engine.prev_p_id # Newton iteration number

#        self.engine.gx_idx += 1 

        #.format(t_id, r_id, time, rho, s, kti, kn), file=f)
        ft = self.flux_tubes[r_id] 
        ft.set_profiles(temp_i, temp_e, nu_ii, nu_ee, beta_ref)
        #ft.set_dens_temp(temp_i, temp_e)
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
        if (t_id == 0 or restart == False): 
            # skip only for first time step
            ft.gx_input.inputs['Restart']['restart'] = 'false'
            ft.gx_input.inputs['Initialization']['init_amp'] = '1e-3'

        else:
            ft.gx_input.inputs['Restart']['restart'] = 'true'
            f_load = f"restarts/saved-t{t_id-1:02d}-p{prev_p_id}-r{r_id}-{job_id}.nc" 
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + f_load)
            ft.gx_input.inputs['Initialization']['init_amp'] = '0.0'
            # restart from the same file (prev time step), to ensure parallelizability

        
        #### save restart file (always)
        ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
        f_save = f"restarts/saved-t{t_id:02d}-p{p_id}-r{r_id}-{job_id}.nc"
        ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + f_save)


        ### execute
        ft.gx_input.write(path + fout, overwrite=overwrite)
        qflux = self.run_gx(tag, path, overwrite=overwrite) # this returns a file name
        return qflux



    def run_gx(self,tag,path, overwrite=False):

        f_nc = path + tag + '.nc'
        if ( os.path.exists(f_nc) == False or overwrite ):

            # attempt to call
            system = os.environ['GK_SYSTEM']
            GX_PATH = os.environ.get("GX_PATH") 

            cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', '--exclusive', GX_PATH+'gx', path+tag+'.in'] # stellar
            if system == 'traverse':
                # traverse does not recognize path/to/gx as an executable
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gres=gpu:1', '--exact', '--overcommit', GX_PATH+'gx', path+tag+'.in'] # traverse
            #if system == 'traverse' and self.engine.evolve_density:
            #    # traverse does not recognize path/to/gx as an executable
            #    cmd = ['srun', '-N', '1', '-t', '4:00:00', '--ntasks=2', '--gres=gpu:2', '--exact', '--overcommit', GX_PATH+'gx', path+tag+'.in'] # traverse
            if system == 'satori':
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gres=gpu:1', '--exclusive', GX_PATH+'gx', path+tag+'.in'] # satori
    
            print('Calling', tag, 'with', cmd)
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


###
def print_time():

    dt = datetime.now()
    print('  time:', dt)
    #ts = datetime.timestamp(dt)
    #print('  time', ts)


## is this being used? 9/28
#  if so, it should be replaced by GX_Output class in GX_io.py
def read_gx(f_nc):
    if f_nc == '':
        return 1e-8, None
    # read a GX netCDF output file, returns flux
    try:
        gx_data = gx_io.GX_Output(f_nc)

        tag = f_nc.split('/')[-1]
        print('  {:} pflux: {:}, qflux_i: {:}, qflux_e: {:}'.format(tag, gx_data.pflux, gx_data.qflux_i, gx_data.qflux_e))
        return gx_data

    except:
        print('  issue reading', f_nc)
        exit(1)
        return 0 # for safety, this will be problematic

