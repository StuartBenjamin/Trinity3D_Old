import numpy as np
import pdb
import subprocess
from datetime import datetime

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner

# read GX output
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



class GX_Flux_Model():

    def __init__(self,fname):


        ###  load an input template
        #    later, this should come from Trinity input file
        f_input = 'gx-files/gx-sample.in' 
        self.input_template = GX_Runner(f_input)


        ### This keeps a record of GX comands, it might be retired
        # init file for writing GX commands

        with  open(fname,'w') as f:
            print('t_idx, r_idx, time, r, s, tprim, fprim', file=f)

        # store file name (includes path)
        self.fname = fname
        self.f_handle = open(fname, 'a')
        ###


    def init_geometry(self):

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
        R   = engine.R_major
        rax = engine.rho_axis
        # load gradient scale length
        Ln  = - R * engine.density.grad_log.profile     # L_n^inv
        Lpi = - R * engine.pressure_i.grad_log.profile  # L_pi^inv
        Lpe = - R *engine.pressure_e.grad_log.profile  # L_pe^inv

        # turbulent flux calls, for each radial flux tube
        idx = np.arange(1, engine.N_radial-1) # drop the first and last point
        for j in idx: 
            rho = rax[j]
            kn  = Ln [j]
            kpi = Lpi[j]
            kpe = Lpe[j]

            self.write_command(j, rho, kn       , kpi       , kpe       )
            self.write_command(j, rho, kn + step, kpi       , kpe       )
            self.write_command(j, rho, kn       , kpi + step, kpe       )
            self.write_command(j, rho, kn       , kpi       , kpe + step)


            self.gx_command(j, rho, kn       , kpi       , kpe       )

    # this prepares the input file for a gx command
    def gx_command(self, r_id, rho, kn, kpi, kpe):
        
        s = rho**2
        kti = kpi - kn
        kte = kpe - kn

        t_id = self.t_id # time integer
        time = self.time # time [s]

        #.format(t_id, r_id, time, rho, s, kti, kn), file=f)
        ft = self.flux_tubes[r_id - 1]
        ft.set_gradients(kn, kti, kte)
        
        # to be specified by Trinity input file, or by time stamp
        root = 'gx-files/'
        path = 'run-dir/' 
        tag  = 't{:}-r{:}-0'.format(t_id, r_id)

        fout  = tag + '.in'
        fsave = tag + '-restart.nc'

        if (t_id > 0):
            fload = 't{:}-r{:}-0-restart.nc'.format(t_id - 1, r_id)
            ft.gx_input.inputs['Restart']['restart_from_file'] = '"{:}"'.format(root + path + fload)
            ft.gx_input.inputs['Restart']['restart'] = 'true'
            ft.gx_input.inputs['Controls']['init_amp'] = '0.0'

        ft.gx_input.inputs['Restart']['restart_to_file'] = '"{:}"'.format(root + path + fsave)
        ft.gx_input.write(root + path + fout)

        ### execute
        self.run_gx(tag, root+path)

    def run_gx(self,tag,path):

        # attempt to call
        cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', path+'./gx', path+tag]

        #print('attempting to call:', cmd)
        print('Calling', tag)
        print_time()
        with open('log.'+tag, 'w') as fp:
        	subprocess.run(cmd, stdout=fp)

        print('slurm gx completed')
        print_time()

        ### attempt to read
        fin = path + tag + '.nc'

        print('attempting to read', fin)
        try:
            qflux = gout.read_GX_output( fin )
            print('  {:} qflux: {:}'.format(tag, qflux))

        except:
            print('  issue reading', fin)
            
        
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
