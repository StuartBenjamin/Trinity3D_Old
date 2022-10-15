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
        gx.inputs['Geometry']['geo_file']  = '"{:}"'.format(self.f_geo)
        gx.inputs['Geometry']['shat']     = self.shat

        # save
        self.gx_input = gx

    def set_gradients(self, kn, kpi, kpe):

        gx = self.gx_input

        tprim = '[ {:.2f},       {:.2f}     ]'.format(kpi, kpe)
        gx.inputs['species']['tprim'] = tprim

        fprim = '[ {:.2f},       {:.2f}     ]'.format(kn, kn)
        gx.inputs['species']['fprim'] = fprim

    def set_profiles(self, temp_i, temp_e):
    #def set_dens_temp(self, temp_i, temp_e): # renamed 10/13

        gx = self.gx_input

        temp = f"[ {temp_i:.2f},       {temp_e:.2f}     ]"
        gx.inputs['species']['temp'] = temp

        # for adiatibatic electrons ne=ni so dens is [1,1] for now

        # set tau = Te/Ti ratio
        tau = temp_e/temp_i
        gx.inputs['Boltzmann']['tau_fac'] = tau


class VmecReader():
    """
    This Class reads a vmec wout file.

    It computes geometric quantities such as surface area
    """

    def __init__(self,fin):

        f = Dataset(fin, mode='r')
        def get(f,key):
            return f.variables[key][:]

        # 0D array
        self.nfp         = get(f,'nfp')
        self.ns          = get(f,'ns')
        self.mnmax       = get(f,'mnmax')
        self.aminor      = get(f,'Aminor_p')
        self.Rmajor      = get(f,'Rmajor_p')
        self.volume      = get(f,'volume_p')


        # 1D array
        self.xm          = get(f,'xm')
        self.xn          = get(f,'xn')
        self.xm_nyq      = get(f,'xm_nyq')
        self.xn_nyq      = get(f,'xn_nyq')

        self.iotaf       = get(f,'iotaf')
        self.presf       = get(f,'presf')

        # 2D array
        self.rmnc        = get(f,'rmnc')
        self.zmns        = get(f,'zmns')
        self.lmns        = get(f,'lmns')
        self.bmnc        = get(f,'bmnc')
        self.bsupumnc    = get(f,'bsupumnc')
        self.bsupvmnc    = get(f,'bsupvmnc')
        
        # Get B field
        self.phi  = get(f,'phi') # toroidal flux in SI webers (signed)
        phiedge   = self.phi[-1]
        B_norm = phiedge / (np.pi * self.aminor**2) # GX normalizes B_ref = phi/ pi a2
        self.B_GX = np.abs(B_norm) 

        # save
        self.data = f
        self.N_modes = len(self.xm)
        self.filename = fin


    def fourier2space(self, Cmn, tax, pax, s_idx=48, sine=True):
        """
        Taking Fourier modes CMN, selects for flux surface s_idx
        select sine or cosine for array
        input toroidal and poloidal angle axis (tax, pax)
        outputs 2D array Z(p,t)
        """
    
        arr = []
        for j in np.arange(self.N_modes):
    
            m = int( self.xm[j] )
            n = int( self.xn[j] )
    
            c = Cmn[s_idx,j]
    
            if (sine):
                A = [[ c * np.sin( m*p - n*t )  for t in tax] for p in pax ]
            else:
                A = [[ c * np.cos( m*p - n*t )  for t in tax] for p in pax ]
    
            arr.append(A)
    
        return np.sum(arr, axis=0)

    def get_xsection(self, N, phi=0, s=-1):
        '''
        Gets a poloidal cross section at const toroidal angle phi
        '''

        pax = np.linspace(0,np.pi*2,N) # poloidal
        tax = np.array([phi])          # toroidal

        # positions
        R2d = self.fourier2space(self.rmnc, tax,pax, sine=False, s_idx=s)
        Z2d = self.fourier2space(self.zmns, tax,pax, sine=True,  s_idx=s)

        # cartisian coordinates for flux surface
        R = R2d[:,0]
        Z = Z2d[:,0]

        return R,Z

    def get_surface(self, surface, N_zeta=20, N_theta=8, save_cloud=False):
        '''
        Compute area on a single flux surface
        
        using finite difference area elements
        with resolution (N_zeta, N_theta).
        '''

        # get points
        nfp = self.nfp 
        r_arr = []
        for p in np.linspace(0,np.pi*2/nfp,N_zeta):
            r,z = self.get_xsection(N_theta,phi=p,s=surface)

            x = r*np.cos(p)
            y = r*np.sin(p)

            r_arr.append(np.transpose([x,y,z]))

        r_arr = np.transpose(r_arr)
        if save_cloud:
            self.r_cloud.append(r_arr)

        # get displacements
        def uv_space(X_arr,Y_arr,Z_arr):
            # modifying the code such that toroidal (and poloidal) directions need not be closed
            # this enables area computation on a field period for stellarators
            # if this is correct, the previous algoirthm had an (n-1) edge error
            #    yes, I believe that is the case. The previous implementation double counted the edge [0,1]
        
            dXdu = X_arr[1:,:-1] - X_arr[:-1,:-1] 
            dYdu = Y_arr[1:,:-1] - Y_arr[:-1,:-1] 
            dZdu = Z_arr[1:,:-1] - Z_arr[:-1,:-1] 
        
            dXdv = X_arr[:-1,1:] - X_arr[:-1,:-1] 
            dYdv = Y_arr[:-1,1:] - Y_arr[:-1,:-1] 
            dZdv = Z_arr[:-1,1:] - Z_arr[:-1,:-1] 
        
            return dXdu, dYdu, dZdu, dXdv, dYdv, dZdv

        X_arr, Y_arr, Z_arr = r_arr
        dXdu, dYdu, dZdu, dXdv, dYdv, dZdv = uv_space(X_arr,Y_arr,Z_arr)

        # get area
        dRdu = np.array([dXdu, dYdu, dZdu])
        dRdv = np.array([dXdv, dYdv, dZdv])

        # compute cross product and take norm
        dArea = np.linalg.norm( np.cross(dRdu, dRdv,axis=0),axis=0)
        if save_cloud:
            self.A_cloud.append(dArea)

        return np.sum(dArea) * nfp


    def calc_dV(self, radial_grid, N_fine=100):

        from scipy.interpolate import interp1d
        psi_axis = np.linspace(0,1, self.ns)
        areas = self.surface_areas
        
        # switching to radial grid (linear) is easier to interpolate line than psi (sqrt)
        rho_axis = np.sqrt(psi_axis)
        a_of_r = interp1d(rho_axis, areas, kind='cubic')  

        # make a fine grid
        fine_grid = np.linspace(0,1,N_fine)
        midpoints = (fine_grid[1:] + fine_grid[:-1]) / 2
        dr = self.aminor / len(midpoints)
        dV_fine = a_of_r(midpoints) * dr

        # get nearest grid index for each trinity point
        args = [ np.argmin( np.abs(midpoints - r) ) for r in radial_grid ]  
        
        # split dV_fine based on the args, then recombine into dV on a coarse grid
        dV = [ np.sum(segment) for segment in np.split(dV_fine, args) ]
        return np.array(dV)

    def calc_gradrho_area(self,r_axis, N_zeta=16, N_theta=8,):
        '''
        Compute area and < | grad rho | >
        the surface area, of the absolute value, of 3D gradient of rho

        r_axis is a list of (rho = r/a) Trinity grid points
        '''

        self.r_cloud = []
        self.A_cloud = []
        s_list = []

        sax = np.linspace(0,1,self.ns)
        N_points = len(r_axis)
        print(f"    geo: computing {N_points} radial points")
        for r in r_axis:

            ### get s_index just above and below
            s = r**2
            
            s1,s2 = np.argsort( np.abs(sax - s) )[:2]
            self.get_surface(s1, N_zeta=N_zeta, N_theta=N_theta, save_cloud=True) 
            self.get_surface(s2, N_zeta=N_zeta, N_theta=N_theta, save_cloud=True) 
            s_list.append( sax[s1] )
            s_list.append( sax[s2] )

        r_cloud = np.array(self.r_cloud)
        a_cloud = np.array(self.A_cloud)
        r_list = np.sqrt(s_list)
 
        r1 = r_list[1::2] 
        r0 = r_list[0::2]
        drho = r1 - r0

        ### compute < | grad rho | >
        # drop the last theta and zeta points, in order to match the area array downstream
        dr = np.reshape( (r_cloud[1::2] - r_cloud[0::2])[:,:,:-1,:-1], (N_points,3,-1) )
        dx = np.linalg.norm( dr, axis=1)
        abs_grad_rho = np.abs(drho)[:,np.newaxis] / dx

        
        ### compute area
        a1 = a_cloud[1::2] 
        a0 = a_cloud[0::2]
        nax = np.newaxis
        w_list = (r_axis - r0) / (r1 - r0) + 1e-8
        weight = w_list[:,nax,nax]
        dA = np.reshape( a0*(1-weight) + a1*weight, (N_points,-1))

        avg_abs_grad_rho = np.sum(abs_grad_rho*dA,axis=1)/ np.sum(dA,axis=1) 
        areas = np.sum( np.abs(dA),axis=1)

        ### save
        self.avg_abs_grad_rho = avg_abs_grad_rho
        self.surface_areas = areas * self.nfp

    def save_areas(self):

        # strip the path
        fname = self.filename.split('/')[-1] 

        # assume vmec file has the form wout_[tag].nc
        tag = fname[5:-3]

        # save
        fout = f"area_psi_{tag}.npy"
        np.save(fout, self.surface_areas)
        print(f"  Save surface areas to: {fout}")


    def overwrite_simple_torus(self, R=6, a=2):
        '''
        Overwrites VMEC file with concentric circle geometry.

        This is used for testing.
        '''

        sax = np.linspace(0,1,self.ns)
        rax = a * np.sqrt(sax)
        
        # overwrite with simple torus
        self.N_modes = 2
        self.xn = np.array([0,0])
        self.xm = np.array([0,1])

        self.rmnc = np.array( [ [R,r] for r in rax ] )
        self.zmns = np.array( [ [0,r] for r in rax ] )
        
        self.Rmajor = R
        self.aminor = a

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





