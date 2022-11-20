import numpy as np
import subprocess
from datetime import datetime
import time as _time

#import Geometry as geo
from Geometry import FluxTube
from GX_io    import GX_Runner
from netCDF4 import Dataset

# read GX output
#import trinity_lib as trl
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

def FluxModelFactory(inputs, grid, time, species):

    model_parameters = inputs.get('model', {})
    my_model = model_parameters.get('model', 'GX')

    models = {
       "GX": GX_FluxModel,
       "ReLU": ReLU_FluxModel,
       "ReLU-particle-only": ReLU_FluxModel,
       "diffusive": Barnes_Model2
    }
    return models[my_model](inputs, grid, time, species)

class FluxModel():

    # base class constructor, should be called in derived class 
    # constructor by super().__init__()
    def __init__(self, inputs, grid, time, species):
        self.N_fluxtubes = len(grid.mid_axis)
        self.rho = grid.mid_axis
        self.grid = grid
        self.time = time

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



'''
analytic flux model based on ReLU + neoclassical
'''
class ReLU_FluxModel():

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
class GX_FluxModel(FluxModel):

    def __init__(self, inputs, grid, time, species):

        # call base class constructor
        super().__init__(inputs, grid, time, species)

        # environment variable containing path to gx executable
        GX_PATH = os.environ.get("GX_PATH") or ""

        model_parameters = inputs.get('model', {})
        gx_template = model_parameters.get('gx_template', 'gx-files/gx-sample.in')
        out_dir = self.out_dir = model_parameters.get('gx_outputs', 'gx-files/out-dir/')
        self.overwrite = model_parameters.get('overwrite', False)

        ### Check file path
        print("  Looking for GX files")
        print("    Expecting GX template:", gx_template)
        print("    Expecting GX executable:", GX_PATH + "gx")
        print("    GX-Trinity output path:", out_dir)

        found_path = os.path.exists(out_dir)
        if (found_path == False):
            print(f"      creating new output dir {out_dir}")
            os.mkdir(out_dir)

        # make a directory for restart files
        restart_dir = out_dir + 'restarts' # this name is hard-coded to agree with that in gx_command(), a better name may be restart_dir/ or a variable naming such
        if os.path.exists(restart_dir) == False:
            print(f"          creating new restart dir {restart_dir}")
            os.mkdir(restart_dir)

        found_gx = os.path.exists(GX_PATH+"gx")
        if (found_gx == False):
            print("  Error: gx executable not found! Make sure the GX_PATH environment variable is set.")
            exit(1)

        print("")

        ###  load an input template
        self.read_input(gx_template)
        self.nstep_gx = self.inputs['Time']['nstep']

        self.processes = []
 
        self.B_ref = 1 # this will be reset 

        # do a dummy init-only GX calculation for each flux tube so that the
        # GX geometry information (B_ref, a_ref, grho, area) can be read
        kn0_sj, kT0_sj = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        out_ncs = self.run_gx_fluxtubes(self.rho, kn0_sj, kT0_sj, species, 'i', init_only=True)
        ### collect parallel runs
        self.wait()
        # read
        _time.sleep(WAIT_TIME)
        self.read_gx_geometry(out_ncs)


        #if self.vmec_wout: # is not blank

        #    # else launch flux tubes from VMEC
        #    f_geo     = self.f_geo
        #    geo_path  = self.gx_root  # this says where the convert executable lives, and where to find the sample .ing file
        #    out_path  = self.path
        #    vmec_path = self.vmec_path

        #    geo_template = gx_io.VMEC_GX_geometry_module( self.engine,
        #                                         f_sample = f_geo,
        #                                         input_path = geo_path,
        #                                         output_path = out_path,
        #                                         tag = vmec[5:-3]
        #                                      )
        #    geo_template.set_vmec( vmec, 
        #                  vmec_path   = vmec_path, 
        #                  output_path = out_path )

        #    geo_files = []
        #    N_fluxtubes = len(self.midpoints)
        #    for j in np.arange(N_fluxtubes):
        #        rho = self.midpoints[j]
        #        f_geometry = geo_template.init_radius(rho,j) 
        #        geo_files.append(out_path + f_geometry)

    # run GX calculations and pass fluxes and flux jacobian to SpeciesDict species
    def get_fluxes(self, species):
        # in the following, A_sjk indicates A[s, j, k] where 
        # s is species index
        # j is rho index
        # k is perturb index
  
        out_ncs_jk = []
        dkap_jk = []
       
        rho_j = self.rho
 
        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        dkap_jk.append(0) # this is a dummy entry, unused
        out_ncs_j = self.run_gx_fluxtubes(rho_j, kn0_sj, kT0_sj, species, pert_id)
        out_ncs_jk.append(out_ncs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_list:
            pert_id = pert_id + 1
            kn_sj, kT_sj = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            dkap = kn_sj - kn0_sj
            dkap_jk.append(dkap[dkap != 0]) # this eliminates elements that are zero,
                                            # so that we are only left with the perturbation in this (stype) species
            out_ncs_for_fluxtubes = self.run_gx_fluxtubes(rho_j, kn_sj, kT_sj, species, pert_id)
            out_ncs_jk.append(out_ncs_for_fluxtubes)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_list:
            pert_id = pert_id + 1
            kn_sj, kT_sj = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            dkap = kT_sj - kT0_sj
            dkap_jk.append(dkap[dkap != 0]) # this eliminates elements that are zero,
                                            # so that we are only left with the perturbation in this (stype) species
            out_ncs_for_fluxtubes = self.run_gx_fluxtubes(rho_j, kn_sj, kT_sj, species, pert_id)
            out_ncs_jk.append(out_ncs_for_fluxtubes)

        ### collect parallel runs
        self.wait()

        # read
        _time.sleep(WAIT_TIME)

        # read GX output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj = self.read_gx_fluxes(out_ncs_jk[pert_id], species)
        species.set_flux(pflux0_sj, qflux0_sj)
        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_list:
            dkn_j = dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_gx_fluxes(out_ncs_jk[pert_id], species)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            species.set_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_list:
            dkT_j = dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_gx_fluxes(out_ncs_jk[pert_id], species)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            species.set_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj)
            pert_id = pert_id + 1

    def wait(self):

        # wait for a list of subprocesses to finish
        #    and reset the list

        exitcodes = [ p.wait() for p in self.processes ]
        print(exitcodes)
        self.processes = [] # reset

        # could add some sort of timer here

    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('#') > -1:
                end = line.find('#')
                line = line[:end]

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key   = key.strip()
            value = value.strip()
            
            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        self.inputs = obj
        self.filename = fin

    def write_input(self, fout='temp.in'):

        # do not overwrite
        if (os.path.exists(fout) and not self.overwrite):
            print( '  input exists, skipping write', fout )
            return

        with open(fout,'w') as f:
        
            for item in self.inputs.items():
                 
                if ( type(item[1]) is not dict ):
                    print('  %s = %s ' % item, file=f)  
                    continue
    
                header, nest = item
                print('\n[%s]' % header, file=f)
    
                longest_key =  max( nest.keys(), key=len) 
                N_space = len(longest_key) 
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)

        print('  wrote input:', fout)

    # run a GX flux tube calculation at each radius, given gradient values kns and kts
    def run_gx_fluxtubes(self, rho, kns, kts, species, pert_id, init_only=False):

        t_id = self.time.t_idx # time integer
        p_id = self.time.p_idx # Newton iteration number
        prev_p_id = self.time.prev_p_id # Newton iteration number

        gx_inputs = self.inputs

        # get profile values on flux grid
        # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True)
        beta_ref = species.ref_species.beta_on_flux_grid(self.B_ref)

        if species.has_adiabatic_species:
            T_adiab = species.adiabatic_species.T().toFluxProfile()
            T_ref = species.ref_species.T().toFluxProfile()
            tau_fac = T_ref/T_adiab
            N_species = species.N_species - 1 # in GX, adiabatic species doesn't count towards nspecies
            if species.adiabatic_species.type == "electron":
                adiab_type = "electron"
            else:
                adiab_type = "ion"
        else:
            N_species = species.N_species

        # loop over flux tubes, one at each rho
        out_ncs = []
        for r_id in np.arange(self.N_fluxtubes):

            gx_inputs['Dimensions']['nspecies'] = N_species

            if init_only:
                gx_inputs['Time']['nstep'] = 1
            else:
                gx_inputs['Time']['nstep'] = self.nstep_gx

            gx_inputs['Geometry']['rhoc'] = rho[r_id]

            # set reference beta
            gx_inputs['Physics']['beta'] = beta_ref[r_id]

            # set species parameters (these are lists)
            gx_inputs['species']['mass'] = list(species.get_masses(normalize=True))
            gx_inputs['species']['z'] = list(species.get_charges(normalize=True))
            gx_inputs['species']['dens'] = list(ns[:, r_id])
            gx_inputs['species']['temp'] = list(Ts[:, r_id])
            gx_inputs['species']['fprim'] = list(kns[:, r_id])
            gx_inputs['species']['tprim'] = list(kts[:, r_id])
            gx_inputs['species']['vnewk'] = list(nu_ss[:, r_id])
            gx_inputs['species']['type'] = species.get_types_ion_electron()

            if species.has_adiabatic_species:
                gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'true'
                gx_inputs['Boltzmann']['tau_fac'] = tau_fac[r_id]
                gx_inputs['Boltzmann']['Boltzmann_type'] = f"'{adiab_type}'"
            else:
                gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'false'

            path = self.out_dir
            tag  = f"t{t_id:02}-p{p_id}-r{r_id}-{pert_id}"

            fout  = tag + '.in'
            f_save = tag + '-restart.nc'

            # Load restart if this is the first trinity timestep,
            # or if the heat flux from the previous GX run was very small
            if (t_id == 0 or species.ref_species.qflux[r_id] < 1e-10): 
                gx_inputs['Restart']['restart'] = 'false'
            else:
                gx_inputs['Restart']['restart'] = 'true'
                f_load = f"restarts/saved-t{t_id-1:02d}-p{prev_p_id}-r{r_id}-{pert_id}.nc" 
                gx_inputs['Restart']['restart_from_file'] = '"{:}"'.format(path + f_load)
                gx_inputs['Initialization']['init_amp'] = '0.0'
                # restart from the same file (prev time step), to ensure parallelizability

            #### save restart file (always)
            gx_inputs['Restart']['save_for_restart'] = 'true'
            f_save = f"restarts/saved-t{t_id:02d}-p{p_id}-r{r_id}-{pert_id}.nc"
            gx_inputs['Restart']['restart_to_file'] = '"{:}"'.format(path + f_save)

            ### execute
            self.write_input(path + fout)
            out_nc = self.submit_gx_job(tag, path) # this returns a file name
            out_ncs.append(out_nc)
        #end flux tube loop

        return out_ncs

    def submit_gx_job(self,tag,path):

        f_nc = path + tag + '.nc'
        if ( os.path.exists(f_nc) == False or self.overwrite ):

            # attempt to call
            system = os.environ['GK_SYSTEM']
            GX_PATH = os.environ.get("GX_PATH") or ""

            cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', '--exclusive', GX_PATH+'gx', path+tag+'.in'] # stellar
            if system == 'traverse':
                # traverse does not recognize path/to/gx as an executable
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gpus-per-task=1', GX_PATH+'gx', path+tag+'.in'] # traverse
            if system == 'satori':
                cmd = ['srun', '-N', '1', '-t', '2:00:00', '--ntasks=1', '--gres=gpu:1', GX_PATH+'gx', path+tag+'.in'] # satori
    
            print('Calling', tag, "with")
            print(">", ' '.join(cmd))
            print_time()
            f_log = path + 'log.' +tag
            with open(f_log, 'w') as fp:

                print('   running:', tag)
                p = subprocess.Popen(cmd, stdout=fp)
                self.processes.append(p)

        else:
            print('  gx output {:} already exists'.format(tag) )

        return f_nc # this is a file name

    def read_gx_fluxes(self, fnc_j, species):
        pflux = np.zeros( (species.N_species, self.N_fluxtubes) )
        qflux = np.zeros( (species.N_species, self.N_fluxtubes) )

        for r_id in np.arange(self.N_fluxtubes):
            fnc = fnc_j[r_id]
            try:
                print('  read_gx_fluxes: reading', fnc)
                f = Dataset(fnc, mode='r')
            except: 
                print('  read_gx_fluxes: could not read', fnc)

            # read qflux[t,s] = time trace of heat flux for each species
            for i, s in enumerate(species.species_dict.values()):
                if s.is_adiabatic:
                    pflux[i, r_id] = 0.0
                    qflux[i, r_id] = 0.0
                else:
                    pflux_t = f.groups['Fluxes'].variables['pflux'][:,i]
                    pflux[i, r_id] = self.median_estimator(pflux_t)

                    qflux_t = f.groups['Fluxes'].variables['qflux'][:,i]
                    qflux[i, r_id] = self.median_estimator(qflux_t)

        return pflux, qflux

    def median_estimator(self, flux):

        N = len(flux)
        med = np.median( [ np.median( flux[::-1][:k] ) for k in np.arange(1,N)] )

        return med

    def read_gx_geometry(self, fnc_j):
        B_ref = []
        a_ref = []
        grho = []
        area = []

        for r_id in np.arange(self.N_fluxtubes):
            fnc = fnc_j[r_id]
            try:
                f = Dataset(fnc, mode='r')
            except: 
                print('  read_gx_geometry: could not read', fnc)

            B_ref.append(f.groups['Geometry']['B_ref'][:])
            a_ref.append(f.groups['Geometry']['a_ref'][:])
            grho.append(f.groups['Geometry']['grhoavg'][:])
            area.append(f.groups['Geometry']['surfarea'][:])

        self.B_ref = pf.FluxProfile(np.asarray(B_ref), self.grid)
        self.a_ref = pf.FluxProfile(np.asarray(a_ref), self.grid)
        self.grho = pf.FluxProfile(np.asarray(grho), self.grid)
        self.area = pf.FluxProfile(np.asarray(area), self.grid)
       

###
def print_time():

    dt = datetime.now()
    print('  time:', dt)
    #ts = datetime.timestamp(dt)
    #print('  time', ts)


