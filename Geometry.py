### this lives as an instance of the Trinity Engine
#   to handle flux tubes and GX's geometry module

from netCDF4 import Dataset
import copy

import pdb


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

        # read VMEC flux surface radius from file string
        #    assumes: gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc
        self.psiN   = float(f_geo.split('/')   [-1]
                                 .split('psiN')[-1]
                                 .split('_')   [1])
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

