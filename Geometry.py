### this lives as an instance of the Trinity Engine
#   to handle flux tubes and GX's geometry moduleA

from netCDF4 import Dataset
import copy

import pdb

class Geometry():

    def __init__(self, N_radius):

        self.N_radius = N_radius # the number of points in Trinity grid

        ## lets start by assuming the number of GX points is the same, 
        #     but potentially it could be fewer 
        #     if want to come up with a flux interpolation scheme


        ###  load an input template
        #    later, this should come from Trinity input file
        f_input = 'gx-files/gx-sample.in' 
        self.input_template = GX_Runner(f_input)

        ### list for storing flux tubes
        self.flux_tubes = []


    def load_fluxtube(self, f_geo):

        ft = FluxTube(f_geo)       # init an instance of flux tube class
        ft.load_gx_input(self.input_template)
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
        gx.inputs['Geometry']['geofile']  = self.f_geo
        gx.inputs['Geometry']['shat']     = self.shat

        # save
        self.gx_input = gx


class GX_Runner():

    # This class handles GX input files, and also execution
    #   copied from GX-ready.py

    def __init__(self,template):
        
        self.read_input(template)


    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # skip comments
            if line.find('#') > -1:
                continue

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


    def write(self, fout='temp.in'):

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

        print('  wrote to:', fout)

    def execute(self):

        # assume Trinity is in a salloc environment with GPUs
        # write input file, write batch file, execute srun
        pass

    def pretty_print(self, entry=''):
    # dumps out current input data, in GX input format
    #    if entry, where entry is one of the GX input headers
    #       only print the inputs nested under entry

        for item in self.inputs.items():
        
            # a catch for the debug input, which has no header
            if ( type(item[1]) is not dict ):
                if (entry == ''):
                    print('  %s = %s ' % item)
                    continue
     
            header, nest = item

            # special case
            if (entry != ''):
                header = entry
                nest   = self.inputs[entry]

            print('\n[%s]' % header)
     
            longest_key =  max( nest.keys(), key=len) 
            N_space = len(longest_key) 
            for pair in nest.items():
                s = '  {:%i}  =  {}' % N_space
                print(s.format(*pair))

            # special case
            if (entry != ''):
                break
