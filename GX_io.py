import numpy as np
from netCDF4 import Dataset

import subprocess
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


    def list_inputs(self, header=False):

        if header:
            print( "name, tprim, fprim, temp, dens, taufac" )
            return

        name = self.filename.split('/')[-1]

        def load(label):
            # load from species
            data = self.inputs['species'][label][1:-1].split(',')
            return np.array(data,float)

        tprim = load("tprim")
        fprim = load("fprim")
        dens = load("dens")
        temp = load("temp")

        taufac = float(self.inputs['Boltzmann']['tau_fac'])

        print(name, f"{str(tprim):16}", f"{str(fprim):16}"
                  , f"{str(temp):16}", dens, f"{taufac:.3f}")


    def list_resolution(self, header=False):
        
        if header:
            print( 'name', 'ntheta', 'nx', 'ny', 'nhermite', 'nlaguerre', 'dt', 'nstep')
            return

        ntheta = int(self.inputs['Dimensions']['ntheta'])
        nx     = int(self.inputs['Dimensions']['nx'])
        ny     = int(self.inputs['Dimensions']['ny'])
        nhermite = int(self.inputs['Dimensions']['nhermite'])
        nlaguerre = int(self.inputs['Dimensions']['nlaguerre'])
        
        dt = float(self.inputs['Time']['dt'])
        nstep = int(self.inputs['Time']['nstep'])

        tag = self.filename.split('/')[-1]
        print( tag, ntheta,nx,ny, nhermite, nlaguerre, dt, nstep)

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


# should this be a class? Yes, this is now outdated 8/20
def read_GX_output(fname):
    
    try:
        f = Dataset(fname, mode='r')
    except: 
        print('  read_GX_output: could not read', fname)


    t = f.variables['time'][:]
    q = f.groups['Fluxes'].variables['qflux'][:,0]

    # check for NANs
    if ( np.isnan(q).any() ):
         print('  nans found in', fname)
         q = np.nan_to_num(q)

    # median of a sliding median
    N = len(q)
    med = np.median( [ np.median( q[::-1][:k] ) for k in np.arange(1,N)] )

    #print('  read GX output: qflux = ', med)
    return med # this is the qflux

class GX_Output():

    def __init__(self,fname):

        try:
            f = Dataset(fname, mode='r')
            #f = nc.netcdf_file(fname, 'r') 
        except: 
            print('  read_GX_output: could not read', fname)
            self.pflux = 0.0
            self.qflux_i = 0.0
            self.qflux_e = 0.0
   
        pflux = f.groups['Fluxes'].variables['pflux'][:,0]
        qflux_i = f.groups['Fluxes'].variables['qflux'][:,0]
        try:
            qflux_e = f.groups['Fluxes'].variables['qflux'][:,1]
        except:
            qflux_e = 0.0*qflux_i
    
        # check for NANs
        if ( np.isnan(pflux).any() ):
             print('  nans found in', fname)
             pflux = np.nan_to_num(pflux)
        if ( np.isnan(qflux_i).any() ):
             print('  nans found in', fname)
             qflux_i = np.nan_to_num(qflux_i)
        if ( np.isnan(qflux_e).any() ):
             print('  nans found in', fname)
             qflux_e = np.nan_to_num(qflux_e)

        self.pflux = self.median_estimator(pflux)
        self.qflux_i = self.median_estimator(qflux_i)
        self.qflux_e = self.median_estimator(qflux_e)
        self.time  = f.variables['time'][:]

        self.tprim  = f.groups['Inputs']['Species']['T0_prime'][:]
        self.fprim  = f.groups['Inputs']['Species']['n0_prime'][:]
        self.B_ref = f.groups['Geometry']['B_ref'][:]
        self.a_ref = f.groups['Geometry']['a_ref'][:]
        self.grhoavg = f.groups['Geometry']['grhoavg'][:]
        self.surfarea = f.groups['Geometry']['surfarea'][:]

        self.fname = fname
        self.data = f

    def median_estimator(self,flux):

        N = len(flux)
        med = np.median( [ np.median( flux[::-1][:k] ) for k in np.arange(1,N)] )
        return med

    def exponential_window_estimator(self, tau=100):

        ### todo: separate data from physics. let this take flux as an argument, so it can work for both heat and particle flux

        # load data
        time  = self.time
        qflux = self.qflux

        # initial state
        t0       = time[0]
        qavg     = qflux[0]
        var_qavg = 0
        
        Q_avg = []
        Var_Q_avg = []
        
        # loop through time
        N = len(qflux)
        for k in np.arange(N):
        
            # get q(t)
            q = qflux[k]
            t = time [k]
        
            # compute weights
            gamma = (t - t0)/tau
            alpha = np.e**( - gamma)
            delta = q - qavg
        
            # update averages
            qavg = alpha * qavg + q * (1 - alpha)
            var_qavg = alpha * ( var_qavg + (1-alpha)* delta**2)
            t0 = t
        
            # save
            Q_avg.append(qavg)
            Var_Q_avg.append(var_qavg)

        self.Q_avg = Q_avg
        self.Var_Q_avg = Var_Q_avg

        return Q_avg[-1], Var_Q_avg[-1]

    def check_convergence(self, tau_list=[10,50,100], threshold=0.5):

        # turtle: I don't think this is being used yet, it may not have been tested 11/17

        '''
        Runs the exponential moving average for a list of taus

        Decides whether to halt, by comparing neighboring taus.
        '''
        print("tau_list", tau_list)

        data = [ self.exponential_window_estimator(tau=tau) for tau in tau_list]

        avg,var = np.transpose(data)
        std = np.sqrt(var)

        check = std/avg < threshold
        print(check)
        print(avg)

        N = len(tau_list)
        for j in range(N-1):
            
            c0 = check[j]

            if c0:
               c1 = check[j+1]

               if c1:
                   
                   # execute halt command
                   run_name = self.fname[:-3]
                   cmd = f"touch {run_name}.stop"
                   print(cmd)
                   return
        


class VMEC_GX_geometry_module():

    # this class handles VMEC-GX Geometry .ing input files

    def __init__(self, engine,
                       f_sample    = 'gx-geometry-sample.ing',
                       tag         = 'default',
                       input_path  = 'gx-files/',
                       output_path = './'
                       ):

        self.engine = engine
        self.data = self.read(input_path + f_sample)

        self.output_path = output_path
        self.input_path  = input_path
        self.tag  = tag


    # this function is run at __init__
    #    it parses a sample GX_geometry.ing input file
    #    as a dictionary for future modifications
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
        path = self.output_path
        
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


    def set_vmec(self,wout, vmec_path='./', output_path='./'):

        # copy vmec output from vmec_path to output_path
        #cmd = 'cp {:}{:} {:}'.format(vmec_path, wout, output_path)
        #os.system(cmd)

        self.data['vmec_file'] = '"{:}"'.format(wout)
        self.data['out_path'] = '"{:}"'.format(output_path)
        self.data['vmec_path'] = '"{:}"'.format(vmec_path) 

    def init_radius(self,rho,r_idx):

        # set radius
        s = rho**2
        self.data['desired_normalized_toroidal_flux'] = s

        t_idx = self.engine.t_idx
        file_tag = f"vmec-t{t_idx}-r{r_idx}"
        self.data['file_tag'] = f"\"{file_tag}\""

        # write input
        in_path  = self.input_path
        out_path = self.output_path
        fname = self.tag + '-psi-{:.2f}'.format(s)
        self.write(fname)
        print('  wrote .ing', out_path+fname)

        # run
        system = os.environ['GK_SYSTEM']
        cmd = ['{:}convert_VMEC_to_GX'.format(in_path),  out_path+fname]
        if system == 'traverse':
            cmd = ['convert_VMEC_to_GX',  out_path+fname]

        f_log = out_path + fname + '.log'
        with open(f_log, 'w') as fp:
            subprocess.call(cmd,stdout=fp)

        f_geometry = f"gx_geo_{file_tag}.nc"
        return f_geometry

