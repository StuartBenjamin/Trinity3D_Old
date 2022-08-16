### this lives as an instance of the Trinity Engine
#   to handle flux tubes and GX's geometry module

from netCDF4 import Dataset
import copy
import vmec as vmec_py
from mpi4py import MPI

import numpy as np
import subprocess
import f90nml
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

# unused 8/14
#        # read VMEC flux surface radius from file string
#        #    assumes: gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc
#        self.psiN   = float(f_geo.split('/')   [-1]
#                                 .split('psiN')[-1]
#                                 .split('_')   [1])
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


class VmecRunner():

    def __init__(self, input_file, engine):

        self.data = f90nml.read(input_file)
        self.input_file = input_file

        self.engine = engine

    # unused
    def write(self, f_output):
        self.data.write(f_output)

    def run(self, f_input, ncpu=2):

        self.data.write(f_input, force=True)

        #cmd = f"srun -t 2:00:00 -n {ncpu} xvmec2000 {f_input}"
        #os.system(cmd)
        #cmd = ["srun", "-t", "2:00:00", "-n", f"{ncpu}", "xvmec2000", f"{f_input}"] # JFP, can we change this to python vmec?
        verbose = True
        fcomm = MPI.COMM_WORLD.py2f()
        reset_file = ''
        ictrl = np.zeros(5, dtype=np.int32)
        ictrl[0] = 1 + 2 + 4 + 8
        vmec_py.runvmec(ictrl, f_input, verbose, fcomm, reset_file)

        #### Need to figure out how to wait for code to finish...

        #path = self.engine.path
        #tag = "".join( f_input.split('.')[1:] )
        #f_log = path + 'log.' + tag

        #with open(f_log, 'w') as fp:

        #    print('   running:', tag)
        #    p = subprocess.Popen(cmd, stdout=fp)
        #    self.processes.append(p)
   
        ### wait for code to finish
        #exitcodes = [ p.wait() for p in self.processes ]
        ##print(exitcodes)
        #self.processes = [] # reset
        print('slurm vmec completed')
