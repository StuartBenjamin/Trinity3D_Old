import numpy as np
from netCDF4 import Dataset

import subprocess
import sys # unused

import os

class GX_Runner():

    # This class handles GX input files, and also execution

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

        # do not overwrite
        if (os.path.exists(fout)):
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


# should this be a class?
def read_GX_output(fname):
    
    try:
        f = Dataset(fname, mode='r')
    except: 
        print('  read_GX_output: could not read', fname)


    t = f.variables['time'][:]
    q = f.groups['Fluxes'].variables['qflux'][:,0]
    #plt.plot(t,q,'.-',label=fins[j])

    # median of a sliding median
    N = len(q)
    med = np.median( [ np.median( q[::-1][:k] ) for k in np.arange(1,N)] )

    #print('  read GX output: qflux = ', med)
    return med # this is the qflux



class VMEC_GX_geometry_module():

    # this class handles VMEC-GX Geometry .ing input files

    def __init__(self, f_sample='gx-geometry-sample.ing',
                       tag  ='default',
                       path = './'
                       ):

        self.path = path

        self.data = self.read(path + f_sample)
        self.tag  = tag


    def read(self,fin):

        with open(fin) as f:
            indata = f.readlines()

        data = {} # create dictionary
        for line in indata:

            # remove comments
            info = line.split('#')[0]

            # skip blanks
            if info.strip() == '':
                continue

            # parse
            key,val = info.split('=')
            key = key.strip()
            val = val.strip()

            # save
            data[key] = val

        return data


    def write(self, fname): # writes a .ing file

        # load
        data = self.data
        path = self.path
        
        # set spacing
        longest_key =  max( data.keys(), key=len) 
        N_space = len(longest_key) 

        # write
        fout = path + fname + '.ing'
        with open(fout,'w') as f:
            for pair in data.items():
                s = '  {:%i}  =  {}' % N_space
                #print(s.format(*pair))   # print to screen for debugging
                print(s.format(*pair), file=f)

    def set_vmec(self,wout):
        self.data['vmec_file'] = wout

    def init_radius(self,rho):

        # set radius
        s = rho**2
        self.data['desired_normalized_toroidal_flux'] = s

        # write input
        fname = self.tag + '-psi-{:.2f}'.format(s)
        self.write(fname)

        # run
        path = self.path
        cmd = ['./{:}convert_VMEC_to_GX'.format(path),  path+fname]

        f_log = path + fname + '.log'
        with open(f_log, 'w') as fp:
            subprocess.call(cmd,stdout=fp)
