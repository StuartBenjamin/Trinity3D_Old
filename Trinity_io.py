import numpy as np
try:
   from netCDF4 import Dataset
except ModuleNotFoundError:
   print("Please install netCDF4")
   exit()
   
try:
   import tomllib
except ModuleNotFoundError:
   try:
       import tomli as tomllib
   except ModuleNotFoundError:
       print("Please install tomli or upgrade to Python >= 3.11")
       exit()
 

class Species():
    pass

class Trinity_Input():

    # This class handles Trinity input files and parameters
    # Parameters are set from input file if they exist, otherwise use defaults

    def __init__(self, fin):
        
        inputs = self.read_input(fin)

#  the following will be moved to the constructors of Geometry, Physics, etc
#        # maybe add a print statement that declares all these settings?
#        debug_parameters = inputs.get('debug', {})
#        self.collisions = debug_parameters.get('collisions', True)
#        self.alpha_heating = debug_parameters.get('alpha_heating', True)
#        self.bremstrahlung = debug_parameters.get('bremstrahlung', True)
#        self.update_equilibrium = debug_parameters.get('update_equilibrium', False)
#        self.turbulent_exchange = debug_parameters.get('turbulent_exchange', False)
#        self.compute_surface_areas = debug_parameters.get('compute_surface_areas', True)
#
#        path_parameters = inputs.get('path', {})
#        self.gx_inputs  = path_parameters.get('gx_inputs', 'gx-files/')
#        self.gx_outputs = path_parameters.get('gx_outputs', 'gx-files/run-dir')
#        self.gx_sample  = path_parameters.get('gx_sample', 'gx-sample.in')
#        self.vmec_path  = path_parameters.get('vmec_path', './')
#
#        geo_parameters = inputs.get('geometry', {})
#        self.vmec_wout = geo_parameters.get('vmec_wout', '')
#        self.R_major   = geo_parameters.get('R_major', 4)
#        self.a_minor   = geo_parameters.get('a_minor', 1)
#        self.Ba   = geo_parameters.get('Ba', 3)
#
#        log_parameters = inputs.get('log', {})
#        self.N_prints = log_parameters.get('N_prints', 10)
#        self.f_save = log_parameters.get('f_save', 'log_trinity')
#
#        # new option
#        eq_parameters = inputs.get('equilibria', {})
#        self.eq_model = eq_parameters.get('eq_model', '')
        
        self.input_dict = inputs

    def read_input(self, fin):

        '''
            Read TOML input file
        '''

        with open(fin, mode="rb") as f:
            obj = tomllib.load(f)

        return obj


    def write(self, fout='temp.in'):

        # do not overwrite
        if (os.path.exists(fout)):
            print( '  input exists, skipping write', fout )
            return

        with open(fout,'w') as f:
        
            for item in self.inputs.items():
                
                if ( type(item[1]) is not dict ):
                    #print('  {} = {} '.format(item, item), file=f)  
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

