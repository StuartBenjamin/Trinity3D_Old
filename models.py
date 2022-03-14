import numpy as np
import pdb
import subprocess
from datetime import datetime
import time as _time

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner

# read GX output
from os.path import exists
import trinity_lib as trl
import GX_io as gout

### this library contains model functons for flux behavior

def ReLU(x,a=0.5,m=1):
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


# for a particle source
def Gaussian(x,sigma=.3,A=2):
    w = - (x/sigma)**2  / 2
    return A * np.e ** w



# this is a toy model of Flux based on ReLU + neoclassical
#     to be replaced by GX or STELLA import module
class Flux_model():

    def __init__(self,
               # neoclassical diffusion coefficient
               D_neo  = 0.1, 
               # critical gradient
               n_critical_gradient  = .5, 
               pi_critical_gradient = .5,
               pe_critical_gradient = .5,
               # slope of flux(Ln) after onset
               n_flux_slope  = 1.1, 
               pi_flux_slope = 1.1,
               pe_flux_slope = 1.1 ):

        # store
        self.neo = D_neo
        self.neo = 0 # turn off neo for debugging
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
        D_n  = ReLU(kn , a=self.n_critical_gradient , m=self.n_flux_slope ) #* 0  # turn off Gamma for debugging
        D_pi = ReLU(kpi, a=self.pi_critical_gradient, m=self.pi_flux_slope) #*0
        D_pe = ReLU(kpe, a=self.pe_critical_gradient, m=self.pe_flux_slope) #*0

        D_turb = D_n + D_pi + D_pe # does not include neoclassical part
#        D_turb = 0 # turn turbulence off for debugging
        return D_turb

    # compute the derivative with respect to gradient scale length
    #     dx is an arbitrary step size
    def flux_gradients(self, kn, kpi, kpe, step = 0.1):
        
        # turbulent flux calls
        d0 = self.flux(kn, kpi, kpe)
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

    def __init__(self, D = 1):

        self.D = D

    def compute_Q(self,engine, step=0.1):

        pi = engine.pressure_i.profile
        pe = engine.pressure_e.profile

        a = engine.a_minor
        Lpi = - a * engine.pressure_i.grad_log.profile  # L_pi^inv
        Lpe = - a * engine.pressure_e.grad_log.profile  # L_pe^inv

        # Qs = 3/2 D (a/Lps) ps / pi**(-5/2)
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
        engine.Gamma  = trl.profile(zero, half=True)
        engine.Qi     = trl.profile(Qi, half=True) 
        engine.Qe     = trl.profile(Qe, half=True) 
        engine.G_n    = trl.profile(zero , half=True)
        engine.G_pi   = trl.profile(zero, half=True)
        engine.G_pe   = trl.profile(zero, half=True)
        engine.Qi_n   = trl.profile(zero , half=True)
        engine.Qi_pi  = trl.profile(dQi_pi, half=True)
        engine.Qi_pe  = trl.profile(zero, half=True)
        engine.Qe_n   = trl.profile(zero , half=True)
        engine.Qe_pi  = trl.profile(zero, half=True)
        engine.Qe_pe  = trl.profile(dQe_pe, half=True)


WAIT_TIME = 1  # this should come from the Trinity Engine
class GX_Flux_Model():

    def __init__(self,fname):


        ###  load an input template
        #    later, this should come from Trinity input file
        f_input = 'gx-files/gx-sample.in' 
        self.input_template = GX_Runner(f_input)
        #self.path = 'temp/'
        self.path = 'run-dir/'

        ### This keeps a record of GX comands, it might be retired
        # init file for writing GX commands

        with  open(fname,'w') as f:
            print('t_idx, r_idx, time, r, s, tprim, fprim', file=f)

        # store file name (includes path)
        self.fname = fname
        self.f_handle = open(fname, 'a')
        ###

        self.processes = []

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [ p.wait() for p in self.processes ]
        print(exitcodes)
        self.processes = [] # reset

        # could add some sort of timer here


    def init_geometry(self):
        # assumes GX input files already exist

        ### load flux tube geometry
        # these should come from Trinity input file
        geo_inputs = ['gx-files/gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc',
        'gx-files/gx_wout_gonzalez-2021_psiN_0.295_gds21_nt_38_geo.nc',  
        'gx-files/gx_wout_gonzalez-2021_psiN_0.500_gds21_nt_40_geo.nc',
        'gx-files/gx_wout_gonzalez-2021_psiN_0.704_gds21_nt_42_geo.nc',
        'gx-files/gx_wout_gonzalez-2021_psiN_0.897_gds21_nt_42_geo.nc']

        ### list for storing flux tubes
        self.flux_tubes = []
        for fin in geo_inputs:
            self.load_fluxtube(fin)

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


    def prep_commands(self, engine,
                                  t_id,
                                  time,
                                  step = 0.1):

        self.t_id = t_id
        self.time = time

        # should pass (R/Lx) to GX
        rax = engine.rho_axis
        # load gradient scale length
        #R   = engine.R_major
        #Ln  = - R * engine.density.grad_log.profile     # L_n^inv
        #Lpi = - R * engine.pressure_i.grad_log.profile  # L_pi^inv
        #Lpe = - R * engine.pressure_e.grad_log.profile  # L_pe^inv
        # this R should be a profile (?)
        a = engine.a_minor
        Ln  = - a * engine.density.grad_log.profile     # L_n^inv
        Lpi = - a * engine.pressure_i.grad_log.profile  # L_pi^inv
        Lpe = - a * engine.pressure_e.grad_log.profile  # L_pe^inv

        # turbulent flux calls, for each radial flux tube
        idx = np.arange(1, engine.N_radial) # drop first point
        #idx = np.arange(1, engine.N_radial-1) # drop the first and last point

        f0   = [''] * len(idx) 
        fn   = [''] * len(idx) 
        fpi  = [''] * len(idx) 
        fpe  = [''] * len(idx) 

        Q0   = np.zeros( len(idx) )
        Qn   = np.zeros( len(idx) )
        Qpi  = np.zeros( len(idx) )
        Qpe  = np.zeros( len(idx) )
        for j in idx: 
            rho = rax[j]
            kn  = Ln [j]
            kpi = Lpi[j]
            kpe = Lpe[j]

            # stores a log of the GX calls
            self.write_command(j, rho, kn       , kpi       , kpe       )
            self.write_command(j, rho, kn + step, kpi       , kpe       )
            self.write_command(j, rho, kn       , kpi + step, kpe       )
            self.write_command(j, rho, kn       , kpi       , kpe + step)

            f0 [j-1] = self.gx_command(j, rho, kn      , kpi        , kpe        , '0' )
            fn [j-1] = self.gx_command(j, rho, kn+step , kpi        , kpe        , '1' )
            fpi[j-1] = self.gx_command(j, rho, kn      , kpi + step , kpe        , '2' )
            fpe[j-1] = self.gx_command(j, rho, kn      , kpi        , kpe + step , '3' )

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        print('starting to read')
        for j in (idx-1): 
            Q0 [j] = read_gx(f0 [j])
            Qn [j] = read_gx(fn [j])
            Qpi[j] = read_gx(fpi[j])
            Qpe[j] = read_gx(fpe[j])

        Qflux  = array_cat(Q0)
        Qi_n   = array_cat( (Qn -Q0) / step )
        Qi_pi  = array_cat( (Qpi-Q0) / step )
        Qi_pe  = array_cat( (Qpe-Q0) / step )

        # save, this is what engine.compute_flux() writes
        zero = 0*Qflux
        eps = 1e-8 + zero # want to avoid divide by zero
        engine.Gamma  = trl.profile(eps, half=True)
        engine.Qi     = trl.profile(Qflux, half=True) 
        engine.Qe     = trl.profile(Qflux, half=True) 
        engine.G_n    = trl.profile(zero , half=True)
        engine.G_pi   = trl.profile(zero, half=True)
        engine.G_pe   = trl.profile(zero, half=True)
        engine.Qi_n   = trl.profile(Qi_n , half=True)
        engine.Qi_pi  = trl.profile(Qi_pi, half=True)
        engine.Qi_pe  = trl.profile(Qi_pe, half=True)
        engine.Qe_n   = trl.profile(Qi_n , half=True)
        engine.Qe_pi  = trl.profile(Qi_pi, half=True)
        engine.Qe_pe  = trl.profile(Qi_pe, half=True)
        # set electron flux = to ions for now

    #  sets up GX input, executes GX, returns input file name
    def gx_command(self, r_id, rho, kn, kpi, kpe, job_id):
        # this version perturbs for the gradient
        # (temp, should be merged as option, instead of duplicating code)
        
        #s = rho**2
        kti = kpi - kn
        kte = kpe - kn

        t_id = self.t_id # time integer

        #.format(t_id, r_id, time, rho, s, kti, kn), file=f)
        ft = self.flux_tubes[r_id - 1]
        ft.set_gradients(kn, kti, kte)
        
        # to be specified by Trinity input file, or by time stamp
        root = 'gx-files/'
        path = self.path
        tag  = 't{:}-r{:}-{:}'.format(t_id, r_id, job_id)

        fout  = tag + '.in'
        fsave = tag + '-restart.nc'

        ### Decide whether to load restart
        if (t_id == 0): 
            # first time step
            ft.gx_input.inputs['Restart']['restart'] = 'false'

        else:
            ft.gx_input.inputs['Restart']['restart'] = 'true'
            fload = 't{:}-r{:}-restart.nc'.format(t_id-1, r_id)
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(root + path + fload)
            ft.gx_input.inputs['Controls']['init_amp'] = '0.0'
            # restart from the same file (prev time step), to ensure parallelizability

        ### Decide whether to save restart
        if (job_id == '0'):
            # save basepoint
            ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
            fsave = 't{:}-r{:}-restart.nc'.format(t_id, r_id)
            ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(root + path + fsave)

        else:
            # perturb gradients
            ft.gx_input.inputs['Restart']['save_for_restart'] = 'false' 
            # make sure I don't redundantly rewrite the restart file here

        ### execute
        ft.gx_input.write(root + path + fout)
        qflux = self.run_gx(tag, root+path)
        return qflux


    def run_gx(self,tag,path):

        f_nc = path + tag + '.nc'
        if ( exists(f_nc) == False ):

            # attempt to call
            cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', path+'./gx', path+tag+'.in'] # new gx binary
            #cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', path+'./gx', path+tag]
    
            print('Calling', tag)
            print_time()
            f_log = path + 'log.' +tag
            with open(f_log, 'w') as fp:

                print('   running:', tag)
                p = subprocess.Popen(cmd, stdout=fp)
                self.processes.append(p)
                #self.wait() # temp, debug
    
            print('slurm gx completed')
            print_time()

        else:
            print('  gx output {:} already exists'.format(tag) )

        return f_nc

        
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

# double the inside point (no flux tube run there)
def array_cat(arr):
    return np.concatenate( [ [arr[0]] , arr ] )

# read a GX netCDF output file
def read_gx(f_nc):
    try:
        qflux = gout.read_GX_output( f_nc )

        tag = f_nc.split('/')[-1]
        print('  {:} qflux: {:}'.format(tag, qflux))
        return qflux

    except:
        print('  issue reading', f_nc)
        # pdb.set_trace
        return 0 # for safety, this will be problematic

