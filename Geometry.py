### this lives as an instance of the Trinity Engine
#   to handle flux tubes and GX's geometry module

from netCDF4 import Dataset
import copy

## for running VMEC
#import vmec as vmec_py
#from mpi4py import MPI
#import f90nml

import numpy as np
import subprocess
import os

class FluxTube():
    ###
    #   This class stores information for one flux tube.
    #   Should it also store the relevant GX input files?

    def __init__(self, fname):
        
        self.load_GX_geometry(fname)
    
    def load_GX_geometry(self, f_geo):
        # loads a single GX flux tube
        #   e.g. gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc
        #        gx_wout_name_geo.nc
        #   there is potentially useful info about radial location and ntheta
        #   but I would prefer to load this from the data within, 
        #   instead of file string
        #   What does gds21 signify? Is this a choice?
       
        print('  Reading GX Flux Tube:', f_geo)
        gf = Dataset(f_geo, mode='r')

        self.ntheta = len(gf['theta'])
        self.shat   = float(gf['shat'][:])
        self.Rmag   = float(gf['Rmaj'][:])

        # store arrays
        self.bmag   = gf['bmag'][:]
        self.grho   = gf['grho'][:] 

        # store netcdf
        self.f_geo  = f_geo
        self.data   = gf

        ## also needs to get surface area from vmec? or is that info already in grho?

    def load_gx_input(self, template):

        # makes a copy of pre-loaded GX input template
        gx = copy.deepcopy(template)

        # modify flux tube specific data
        gx.inputs['Dimensions']['ntheta'] = self.ntheta
        gx.inputs['Geometry']['geofile']  = '"{:}"'.format(self.f_geo)
        gx.inputs['Geometry']['shat']     = self.shat

        # save
        self.gx_input = gx

    def set_gradients(self, kn, kpi, kpe):

        gx = self.gx_input

        tprim = '[ {:.2f},       {:.2f}     ]'.format(kpi, kpe)
        gx.inputs['species']['tprim'] = tprim

        fprim = '[ {:.2f},       {:.2f}     ]'.format(kn, kn)
        gx.inputs['species']['fprim'] = fprim

    def set_dens_temp(self, temp_i, temp_e):

        gx = self.gx_input

        temp = f"[ {temp_i:.2f},       {temp_e:.2f}     ]"
        gx.inputs['species']['temp'] = temp

        # for single species ions ne=ni so dens is [1,1] for now

    def set_fluxtube_scale(self, temp, mass, kts, y0model = 'CBC', y0ionscale = 10): 
        # Moose Set y0 for the fluxtube. See models.py for how this is set.
        gx = self.gx_input
        if y0model == 'CBC': # y0_GX = 4*tprim_GX sqrt(m_{sGX} T_{sGX})
            y0 = 4*kts*sqrt(mass, temp)
            gx.inputs['Domain']['y0'] = y0
        if y0model == 'basic': # y0_GX = y0_ionscale sqrt(m_{sGX} T_{sGX})
            y0 = y0ionscale*sqrt(mass, temp)
            gx.inputs['Domain']['y0'] = y0

class VmecRunner():

    def __init__(self, input_file, engine):

        self.data = f90nml.read(input_file)
        self.input_file = input_file

        self.engine = engine

    def run(self, f_input, ncpu=2):

        self.data.write(f_input, force=True)
        tag = f_input.split('/')[-1][6:]
        vmec_wout = f"wout_{tag}.nc"

        # overwrite previous input file
        self.input_file = f_input
        self.data = f90nml.read(f_input)

        path = self.engine.path
        if os.path.exists( path + vmec_wout ):
            print(f" completed vmec run found: {vmec_wout}, skipping run")
            return

        verbose = True
        fcomm = MPI.COMM_WORLD.py2f()
        reset_file = ''
        ictrl = np.zeros(5, dtype=np.int32)
        ictrl[0] = 1 + 2 + 4 + 8 + 16
        # see VMEC2000/Sources/TimeStep/runvmec.f
        vmec_py.runvmec(ictrl, f_input, verbose, fcomm, reset_file)
        print('slurm vmec completed')

        # right now VMEC writes output to root (input is stored in engine.gx_path)
        #    this cmd manually moves VMEC output to gx_path after vmec_py.run() finishes
        print(f"  moving VMEC files to {path}")
        cmd = f"mv *{tag}* {path}"
        os.system(cmd)




class DescRunner():

    def __init__(self, input_file, engine):

        import desc.io
        eq = desc.io.load(input_file) # loads desc output
        self.input_file = input_file

        self.engine = engine
        self.desc_eq = eq


    def run(self):

        print("DESC RUNNER CALLED")
        eq = self.desc_eq[-1]

        # update profile
        axis = self.engine.rho_axis
        p_SI = self.engine.desc_pressure
        pfit = np.polyfit(axis, p_SI,4)

        from desc.profiles import PowerSeriesProfile
#        from desc.profiles import SplineProfile
        eq.pressure = PowerSeriesProfile(params=pfit)

        # run equilibrium
        eq.solve()

        t_idx = self.engine.t_idx
        path = self.engine.path
        outname = f"{path}desc-t{t_idx:02d}.h5"
        eq.save(outname)
        print("  wrote decs output", outname)





