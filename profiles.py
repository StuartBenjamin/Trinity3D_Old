import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

'''
This class is called within the Trinity-Engine library

It handles general profiles (n, p, F, gamma, Q, etc)
with options to evaluate half-steps and gradients when initializing new Profile objects
'''
class Profile():
    # should consider capitalizing Profile(), for good python form
    def __init__(self, arr, grid):

        # take a 1D array to be density, for example
        self.profile = np.array(arr) 
        self.length  = len(arr)

        self.axis = grid.rho_axis 

    def plot(self,show=False,new_fig=False,label=''):

        if (new_fig):
            plt.figure(figsize=(4,4))

        #ax = np.linspace(0,1,self.length)
        #plt.plot(ax,self.profile,'.-')

        if (label):
            plt.plot(self.axis,self.profile,'.-',label=label)
        else:
            plt.plot(self.axis,self.profile,'.-')

        if (show):
            plt.show()

    __array_ufunc__ = None

    # operator overloads that automatically dereference the profiles
    def __add__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile + B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            return A.__class__(A.profile + B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__add__")

    def __radd__(A,B):
        return A.__add__(B)

    def __sub__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile - B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            return A.__class__(A.profile - B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__sub__")

    def __rsub__(A,B):
        return -1*(A.__sub__(B))

    def __mul__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile * B.profile, A.grid)
        elif (isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile)) or not hasattr(B, '__len__'):
            return A.__class__(A.profile * B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__mul__")

    def __rmul__(A,B):
        return A.__mul__(B)

    def __truediv__(A,B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile / B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            return A.__class__(A.profile / B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")

    def __rtruediv__(A,B):
        if isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            return A.__class__(B / A.profile, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")


class GridProfile(Profile):

    '''
    Profile class with values on the N_radial rho_axis grid points
    '''
    
    def __init__(self, arr, grid):

        self.profile = arr
        self.grid = grid
        self.axis = grid.rho_axis
        self.length = len(self.axis)
        assert self.length == len(arr), "Error: GridProfile array not same length as rho_axis"

    def gradient(self):
        grad_f = np.zeros(self.length)
        f = self.profile
        dr = self.grid.drho
        N = self.length
   
        # inner boundary
        grad_f[0] = (2.*f[3] - 9.*f[2] + 18.*f[1] - 11.*f[0])/(6.*dr)
        grad_f[1] = (-f[3] + 6.*f[2] - 3.*f[1] -2.*f[0])/(6.*dr)
        
        # interior (4 pt centered)
        for ix in np.arange(2, N-2):
            grad_f[ix] = (f[ix-2] - 8.*f[ix-1] + 8.*f[ix+1] - f[ix+2]) / (12.*dr)

        # outer boundary
        grad_f[N-2] = (f[N-4] - 6.*f[N-3] + 3.*f[N-2] + 2.*f[N-1])/(6.*dr)
        grad_f[N-1] = (-2.*f[N-4] + 9.*f[N-3] - 18.*f[N-2] + 11.*f[N-1])/(6.*dr)

        return GridProfile(grad_f, self.grid)

    def log_gradient(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.gradient()/self

    def gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with gradient values evaluated at midpoints
        '''

        flux_length = self.length - 1
        grad_prof = np.zeros(flux_length)
        prof = self.profile
        dr = self.grid.drho

        # inner boundary (uncentered)
        ix = 0
        grad_prof[ix] = (-prof[ix+3] + 3.*prof[ix+2] + 21.*prof[ix+1] - 23.*prof[ix]) / (24.*dr)

        # outer boundary (uncentered)
        ix = flux_length - 1
        grad_prof[ix] = (prof[ix-2] - 3*prof[ix-1] - 21.*prof[ix] + 23.*prof[ix+1]) / (24.*dr)

        # interior (centered)
        for ix in np.arange(1, flux_length-1):
            grad_prof[ix] = (prof[ix-1] - 27.*prof[ix] + 27.*prof[ix+1] - prof[ix+2]) / (24.*dr)

        return FluxProfile(grad_prof, self.grid)

    def log_gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with log gradient values
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.gradient_as_FluxProfile()/self.toFluxProfile()

    def toFluxProfile(self):
 
        flux_length = self.length - 1
        flux_prof = np.zeros(flux_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        flux_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix+1] - prof[ix+2])

        # outer boundary (third order)
        ix = flux_length-1
        flux_prof[ix] = 0.125*(3.0*prof[ix+1] + 6.0*prof[ix] - prof[ix-1])

        # interior (fourth order)
        for ix in np.arange(1, flux_length-1):
            flux_prof[ix] = 0.0625*(-prof[ix-1] + 9.0*prof[ix] + 9.0*prof[ix+1] - prof[ix+2])

        return FluxProfile(flux_prof, self.grid)

class FluxProfile(Profile):

    '''
    Profile class with values on the N_radial-1 mid_axis midpoints 
    '''

    def __init__(self, arr, grid):

        self.profile = arr
        self.grid = grid
        self.axis = grid.mid_axis
        self.length = len(self.axis)

    def toGridProfile(self, axis_val = 0.0):

        grid_length = self.length + 1
        grid_prof = np.zeros(grid_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        grid_prof[ix] = 0.125*(3.0*axis_val + 6.0*prof[ix] - prof[ix+1])

        ix = 1
        grid_prof[ix] = 0.0625*(-axis_val + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix])

        # outer boundary (third order)
        ix = grid_length-2
        grid_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix-1] - prof[ix-2])

        # this is only second order accurate, but these values are never really needed (outer boundary is fixed)
        ix = grid_length-1
        grid_prof[ix] = 2.0*prof[ix-1]-prof[ix-2]

        # interior (fourth order)
        for ix in np.arange(2, grid_length-2):
            grid_prof[ix] = 0.0625*(-prof[ix-2] + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix+1])

        return GridProfile(grid_prof, self.grid)
    

class Flux_profile(Profile):

    '''
    for N trinity grid points (including both boundaries)
    N-2 points need time evolution, based on fluxes evaluated at
    N-1 flux tubes in between the radial grid points

    This class takes outputs from the N-1 flux tubes
    and prepares (2) profiles of length N on the grid
    the (+) profile has dummy info on the -1 idx
    the (-) profile has dummy info on the 0 idx
    let dummy info be a repeated value (though technically, it could be omitted all together. We include it, to simplify array indexing
    '''

    def __init__(self, arr):

        plus  = np.concatenate( [ arr, [0] ] )  # testing if this edge point is used at all
        minus = np.concatenate( [ [0], arr  ] )       # F- always set to zero

        f_interp = interp1d(mid_axis, arr, kind='cubic', fill_value='extrapolate')
        full = f_interp(rho_axis)

        self.plus  = Profile(plus)
        self.minus = Profile(minus)
        self.full  = Profile(full)

        # save raw data
        self.profile = arr


    def plot(self,show=True,new_fig=True,title=''):

        if new_fig:
            plt.figure(figsize=(4,4))

        plt.plot(mid_axis, self.profile, 'x-', label='Q from GX')
        plt.plot(rho_axis, self.plus.profile,'.-', label=r'$Q_+$')
        plt.plot(rho_axis, self.minus.profile,'.-', label=r'$Q_-$')
        plt.plot(rho_axis, self.full.profile,'o-', label='Q (cubic) interp')
        plt.legend()
        plt.grid()

        if title:
            plt.title(title)

        if show:
            plt.show()

##  deleted 8/12
#    def plot(self,show=False,new_fig=False,label=''):
#
#        if (new_fig):
#            plt.figure(figsize=(4,4))
#
#        if (label):
#            plt.plot(self.axis, self.profile,'.-',label=label)
#        else:
#            plt.plot(self.axis, self.profile,'.-')
#
#        if (show):
#            plt.show()

# maybe this class would be better suited extending Profile()?
# or as an alternate init invocation within Profile?


# the class computes and stores normalized flux F, AB coefficients, and psi for the tridiagonal matrix
# it will need a flux Q, and profiles nT
# it should know whether ions or electrons are being computed, or density...
class Flux_coefficients():

    # x is state vector (n, pi, pe)
    # Y is normalized flux (F,I)
    # Z is dlog flux (d log Gamma / d L_x ), evaluated at +- half step
    def __init__(self,x,Y,Z,dZ,norm):

        self.state   = x
        self.flux    = Y # this is normalized flux F,I
        self.RawFlux = Z # this is Gamma,Q
        self.dRawFlux = dZ # this is Gamma,Q
        self.norm    = norm # normalizlation constant (R/a)/drho

        # plus,minus,zero : these are the A,B coefficients
        self.plus  = self.C_plus()
        self.minus = self.C_minus()
        self.zero  = self.C_zero()


    def C_plus(self):

        norm = self.norm

        x  = self.state.profile
        xp = self.state.plus.profile
        Yp = self.flux.plus.profile
        Zp = self.RawFlux.plus.profile
        dZp = self.dRawFlux.plus.profile

        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZp = np.nan_to_num( dZp / Zp )

        Cp = - norm * (x / xp**2) * Yp * dLogZp
        return Profile(Cp)

    def C_minus(self):

        norm = self.norm
        
        x  = self.state.profile
        xm = self.state.minus.profile
        Ym = self.flux.minus.profile
        Zm = self.RawFlux.minus.profile
        dZm = self.dRawFlux.minus.profile
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZm = np.nan_to_num( dZm / Zm )

        Cm = - norm * (x / xm**2) * Ym * dLogZm
        return Profile(Cm)

    def C_zero(self):

        norm = self.norm

        x  = self.state.profile
        xp = self.state.plus.profile
        xm = self.state.minus.profile
        xp1 = self.state.plus1.profile
        xm1 = self.state.minus1.profile
        
        Yp = self.flux.plus.profile
        Zp = self.RawFlux.plus.profile
        dZp = self.dRawFlux.plus.profile

        Ym = self.flux.minus.profile
        Zm = self.RawFlux.minus.profile
        dZm = self.dRawFlux.minus.profile
        
        with np.errstate(divide='ignore', invalid='ignore'):
            dLogZp = np.nan_to_num( dZp / Zp )
            dLogZm = np.nan_to_num( dZm / Zm )

        cp = xp1 / xp**2 * Yp * dLogZp
        cm = xm1 / xm**2 * Ym * dLogZm
        Cz = norm * ( cp + cm ) 
        return Profile(Cz)


# This class organizes the psi-profiles in tri-diagonal matrix
class Psi_profiles():

    def __init__(self,psi_zero,
                      psi_plus,
                      psi_minus,
                      neumann=False):

        # save profiles
        self.plus  = Profile( psi_plus )
        self.minus = Profile( psi_minus )
        self.zero  = Profile( psi_zero )

        # formulate matrix
        M = tri_diagonal(psi_zero,
                         psi_plus,
                         psi_minus)

        if (neumann):
            # make modification for boundary condition
            # this is no longer used.
            M[0,1] -= psi_minus[0]  

        # save matrix
        self.matrix = M


# make tri-diagonal matrix
def tri_diagonal(a,b,c):
    N = len(a)
    M = np.diag(a)
    for j in np.arange(N-1):
        M[j,j+1] = b[j]   # upper, drop last point
        M[j+1,j] = c[j+1] # lower, drop first 
    return M

# Initialize Trinity profiles
#     with default gradients, half steps, and full steps
def init_profile(x,debug=False):

    #x[0] = x[1] ## removed 7/18 since core boundary condition is relaxed
    X = Profile(x, grad=True, half=True, full=True)
    return X

