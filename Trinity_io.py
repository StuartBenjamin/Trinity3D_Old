import numpy as np
from netCDF4 import Dataset


class Trinity_Input():

    # This class handles Trinity input files

    def __init__(self, fin):
        
        self.read_input(fin)


    def read_input(self, fin):

        '''
            The convention is mostly TOML
            [blocks] organize parameters into subgroups
            # comments out a whole line
            ! comments out everything after the '!'
              blank lines are ignored
        '''

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # remove comments
            line = line.split('!')[0]

            # skip disabled lines
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

