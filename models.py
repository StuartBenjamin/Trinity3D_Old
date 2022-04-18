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


# for a particle and heat sources
def Gaussian(x, A=2, sigma=.3, x0=0):
    exp = - ( (x - x0) / sigma)**2  / 2
    return A * np.e ** exp



# this is a toy model of Flux based on ReLU + neoclassical
#     to be replaced by GX or STELLA import module
class Flux_model():

    def __init__(self,
               # neoclassical diffusion coefficient
               D_neo  = 0.5, 
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
        #self.neo = 0 # turn off neo for debugging
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
        #D_turb = 0 # turn turbulence off for debugging
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

    def __init__(self, fname, # is this used?
                       path='run-dir/', 
                       vmec_path='./',
                       vmec_wout="",
                       midpoints=[]
                ):


        ###  load an input template
        #    later, this should come from Trinity input file
        f_input = 'gx-files/gx-sample.in' 
        self.input_template = GX_Runner(f_input)
        self.path = path
        # check that path exists, if it does not, mkdir and copy gx executable
        
        self.midpoints = midpoints
        self.vmec_path = vmec_path
        self.vmec_wout = vmec_wout

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

        ### load flux tube geometry
        # these should come from Trinity input file
        
        vmec = self.vmec_wout
        if vmec != "":

            # else launch flux tubes from VMEC
            f_geo     = 'gx-geometry-sample.ing'
            geo_path  = 'gx-files/'    # this says where the convert executable lives, and where to find the sample .ing file
            out_path  = self.path
            vmec_path = self.vmec_path

            ing = gx_io.VMEC_GX_geometry_module( f_sample = f_geo,
                                                 input_path = geo_path,
                                                 output_path = out_path,
                                                 tag = vmec[5:-3]
                                              )
            ing.set_vmec( vmec, 
                          vmec_path   = vmec_path, 
                          output_path = out_path )

            #rax = [0.435, 0.615, 0.753, 0.869] # hard coded from Bill
            #rax = [1.888888925e-01, 3.777777851e-01, 5.666666627e-01, 7.555555701e-01] # from Noah
            #for rho in rax:
            for rho in self.midpoints:
                ing.init_radius(rho) 

            # gather output files
            geo_files = glob(out_path + 'gx*geo.nc')

            # kludgy fix, if the inner most flux tube is too small for VMEC resolution
            #     just copy the second inner most flux tube
            #     the gradients will be different (and correct) even though the geometries are faked
            #if len(geo_files) < len(self.midpoints):
            #    geo_files = np.concatenate( [[geo_files[0]], geo_files] )

        else:
            # load default files (assumed to be existing)
            print('  no VMEC wout given, loading default files')
            geo_files = [ 'gx-files/gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.295_gds21_nt_38_geo.nc',  
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.500_gds21_nt_40_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.704_gds21_nt_42_geo.nc',
                          'gx-files/gx_wout_gonzalez-2021_psiN_0.897_gds21_nt_42_geo.nc']

        print(' Found these flux tubes', geo_files)

        ### store flux tubes in a list
        self.flux_tubes = []
        for fin in geo_files:
            self.load_fluxtube(fin)


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
        #rax = engine.rho_axis
        # load gradient scale length
        #R   = engine.R_major
        # this R should be a profile (?)
        a = engine.a_minor
        Ln  = - a * engine.density.grad_log   .midpoints  # L_n^inv
        Lpi = - a * engine.pressure_i.grad_log.midpoints  # L_pi^inv
        Lpe = - a * engine.pressure_e.grad_log.midpoints  # L_pe^inv

        # turbulent flux calls, for each radial flux tube
        mid_axis = engine.mid_axis
        idx = np.arange( len(mid_axis) ) 

        f0   = [''] * len(idx) 
        fn   = [''] * len(idx) 
        fpi  = [''] * len(idx) 
        fpe  = [''] * len(idx) 

        Q0   = np.zeros( len(idx) )
        Qn   = np.zeros( len(idx) )
        Qpi  = np.zeros( len(idx) )
        Qpe  = np.zeros( len(idx) )

        for j in idx: 
            rho = mid_axis[j]
            #rho = rax[j]
            kn  = Ln [j]
            kpi = Lpi[j]
            kpe = Lpe[j]

            # stores a log of the GX calls (this step is not actually necessary)
            #  I should add time stamps somehow
#            self.write_command(j, rho, kn       , kpi       , kpe       )
#            self.write_command(j, rho, kn + step, kpi       , kpe       )
#            self.write_command(j, rho, kn       , kpi + step, kpe       )
#            self.write_command(j, rho, kn       , kpi       , kpe + step)


            # writes the GX input file and calls the slurm job
            f0 [j] = self.gx_command(j, rho, kn      , kpi        , kpe        , '0' )
            fpi[j] = self.gx_command(j, rho, kn      , kpi + step , kpe        , '2' )
            #fn [j] = self.gx_command(j, rho, kn+step , kpi        , kpe        , '1' )
            #fpe[j] = self.gx_command(j, rho, kn      , kpi        , kpe + step , '3' )

            # turn off density, since particle flux is set to 0
            # turn off pe, since Qe = Qi

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        print('starting to read')
        for j in idx: 
            Q0 [j] = read_gx(f0 [j])
            Qpi[j] = read_gx(fpi[j])
            #Qn [j] = read_gx(fn [j])
            #Qpe[j] = read_gx(fpe[j])

        Qflux  =  Q0
        Qi_pi  =  (Qpi - Q0) / step 
        #Qi_n   =  (Qn  - Q0) / step 
        #Qi_pe  =  (Qpe - Q0) / step 
        Qi_n = 0*Q0 # this is already the init state
        Qi_pe = Qi_pi


        # need to add neoclassical diffusion

        # save, this is what engine.compute_flux() writes
        zero = 0*Qflux
        eps = 1e-8 + zero # want to avoid divide by zero

        engine.Gamma  = pf.Flux_profile(eps  )
        engine.Qi     = pf.Flux_profile(Qflux) 
        engine.Qe     = pf.Flux_profile(Qflux) 
        engine.G_n    = pf.Flux_profile(zero )
        engine.G_pi   = pf.Flux_profile(zero )
        engine.G_pe   = pf.Flux_profile(zero )
        engine.Qi_n   = pf.Flux_profile(Qi_n )
        engine.Qi_pi  = pf.Flux_profile(Qi_pi)
        engine.Qi_pe  = pf.Flux_profile(Qi_pe)
        engine.Qe_n   = pf.Flux_profile(Qi_n )
        engine.Qe_pi  = pf.Flux_profile(Qi_pi)
        engine.Qe_pe  = pf.Flux_profile(Qi_pe)
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
        ft = self.flux_tubes[r_id] 
        ft.set_gradients(kn, kti, kte)
        
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
            fload = 'restarts/t{:}-r{:}-{:}save.nc'.format(t_id-1, r_id, job_id)
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + fload)
            ft.gx_input.inputs['Controls']['init_amp'] = '0.0'
            # restart from the same file (prev time step), to ensure parallelizability

        
        #### save restart file (always)
        ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
        fsave = 'restarts/t{:}-r{:}-{:}save.nc'.format(t_id, r_id, job_id)
        ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + fsave)


#  old conventions (these restart all types 0-3 from a single node
#        ### Decide whether to load restart
#        if (t_id == 0): 
#            # first time step
#            ft.gx_input.inputs['Restart']['restart'] = 'false'
#
#        else:
#            ft.gx_input.inputs['Restart']['restart'] = 'true'
#            fload = 't{:}-r{:}-restart.nc'.format(t_id-1, r_id)
#            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + fload)
#            ft.gx_input.inputs['Controls']['init_amp'] = '0.0'
#            # restart from the same file (prev time step), to ensure parallelizability
#
#        ### Decide whether to save restart
#        if (job_id == '0'):
#            # save basepoint
#            ft.gx_input.inputs['Restart']['save_for_restart'] = 'true'
#            fsave = 't{:}-r{:}-restart.nc'.format(t_id, r_id)
#            ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + fsave)
#
#        else:
#            # perturb gradients
#            ft.gx_input.inputs['Restart']['save_for_restart'] = 'false' 
#            # make sure I don't redundantly rewrite the restart file here
#
        ### execute
        ft.gx_input.write(path + fout)
        qflux = self.run_gx(tag, path) # this returns a file name
        return qflux



    def run_gx(self,tag,path):

        f_nc = path + tag + '.nc'
        if ( os.path.exists(f_nc) == False ):

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

# double the inside point (no flux tube run there)
### unused
#def array_cat(arr):
#    return np.concatenate( [ [arr[0]] , arr ] )

# read a GX netCDF output file, returns flux
def read_gx(f_nc):
    try:
        qflux = gx_io.read_GX_output( f_nc )
        if ( np.isnan(qflux).any() ):
             print('  nans found in', f_nc, '(setting NaNs to 0)')
             qflux = np.nan_to_num(qflux)

        tag = f_nc.split('/')[-1]
        print('  {:} qflux: {:}'.format(tag, qflux))
        return qflux

    except:
        print('  issue reading', f_nc)
        return 0 # for safety, this will be problematic

