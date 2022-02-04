### this lives as an instance of the Trinity Engine
#   to handle flux tubes and GX's geometry moduleA

from netCDF4 import Dataset
import pdb

class Geometry():

    def __init__(self, N_radius):

        self.N_radius = N_radius # the number of points in Trinity grid

        ## lets start by assuming the number of GX points is the same, 
        #     but potentially it could be fewer 
        #     if want to come up with a flux interpolation scheme

        self.flux_tubes = []
        #gx-files/gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc  
        #gx-files/gx_wout_gonzalez-2021_psiN_0.704_gds21_nt_42_geo.nc
        #gx-files/gx_wout_gonzalez-2021_psiN_0.295_gds21_nt_38_geo.nc  
        #gx-files/gx_wout_gonzalez-2021_psiN_0.897_gds21_nt_42_geo.nc
        #gx-files/gx_wout_gonzalez-2021_psiN_0.500_gds21_nt_40_geo.nc


    def load_fluxtube(self, f_geo):

        ft = FluxTube(f_geo)       # init an instance of flux tube class
        self.flux_tubes.append(ft) # store in list of flux tubes

    def load_VMEC(self, f_vmec):
        # reads a VMEC wout file
        pass

    def run_GX_geometry_module(self):
        # this takes an existing VMEC file, and runs gx.ing to generate a flux tube
        # should store information about the flux tubes
        pass


    def run_VMEC(self):
        # this updates (an existing) VMEC geometry
        #    based on the pressure profile, determined by n,T, and source terms in Trinity
        pass


class FluxTube():

    def __init__(self, fname):
        
        self.load_GX(fname)
    
    def load_GX(self, f_geo):
        # loads a single GX flux tube
        #   e.g. gx_wout_gonzalez-2021_psiN_0.102_gds21_nt_36_geo.nc
        #        gx_wout_name_geo.nc
        #   there is potentially useful info about radial location and ntheta
        #   but I would prefer to load this from the data within, 
        #   instead of file string
        #   What does gds21 signify? Is this a choice?
       
        print('  Reading file:', f_geo)
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
        self.fname  = f_geo
        self.data   = gf

