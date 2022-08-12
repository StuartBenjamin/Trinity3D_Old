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
    def __init__(self,arr, grad=False, half=False, full=False):

        # take a 1D array to be density, for example
        self.profile = np.array(arr) 
        self.length  = len(arr)

        #global rho_axis
        self.axis    = rho_axis # is this necessary, it is used for plotting, do I still use Profile.plot() anywhere?
        # assumes fixed radial griding, which (if irregular) could also be a profile, defined as a function of index

        if (half): # defines half step
            self.plus  = Profile(self.halfstep_pos())
            self.minus = Profile(self.halfstep_neg())

            # returns the (N-1) linear averages between N grid points
            self.midpoints = self.plus.profile [:-1]
            # this can be improved with multiple stencils

        if (full): # defines full stup
            self.plus1  = Profile(self.fullstep_pos())
            self.minus1 = Profile(self.fullstep_neg())

        # pre-calculate gradients, half steps, or full steps
        if (grad):
            self.grad     =  Profile(self.midpoint_gradient(), full=full)
            self.grad_log =  Profile(self.midpoint_log_gradient(), full=full)
#            self.grad     =  Profile(self.gradient(), half=half, full=full)
#            self.grad_log =  Profile(self.log_gradient(), half=half, full=full)

    # pos/neg are forward and backwards
    def halfstep_neg(self):
        # x_j\pm 1/2 = (x_j + x_j \pm 1) / 2
        xj = self.profile
        x1 = np.roll(xj,1)
#        x1[0] = 0 # new update
        x1[0] = xj[0]
        return (xj + x1) / 2
        # return (xj[1:] + xj[:-1])/2 ## new proposal

    def halfstep_pos(self):
        # x_j\pm 1/2 = (x_j + x_j \pm 1) / 2
        xj = self.profile
        x1 = np.roll(xj,-1)
        x1[-1] = xj[-1]
        return (xj + x1) / 2

    def halfstep_second_order(self):
        # -1 9 9 -1
        pass

    def fullstep_pos(self):
        x0 = self.profile
        x1 = np.roll(x0,-1)
        x1[-1] = x0[-1]
        return x1

    def fullstep_neg(self):
        x0 = self.profile
        x1 = np.roll(x0,1)
        x1[0] = x0[0]
        return x1

    def gradient(self):
        '''
        used less, now that GX is evaluated on the MIDPOINTS
        '''
        print("using old gradient")
        # assume equal spacing
        # 3 point - first deriv: u_j+1 - 2u + u_j-1
        xj = self.profile
        xp = np.roll(xj,-1)
        xm = np.roll(xj, 1)

        dx = 1/len(xj) # assumes spacing is from (0,1) [BUG]
        deriv = (xp - xm) / (2*dx)
        deriv[0]  = 0
        #deriv[0]  = deriv[1]      # should a one-sided stencil be used here too?
                                  # should I set it to 0? in a transport solver, is the 0th point on axis?
                                  # I don't think GX will be run for the 0th point. So should that point be excluded from TRINITY altogether?
                                  #      or should it be included as a ghost point?

        # this is a second order accurate one-sided stencil
        deriv[-1]  = ( 3*xj[-1] -4*xj[-2] + xj[-3])  / (2*dx)

        # Bill, from fortran trinity
        deriv[-1]  = ( 23*xj[-1] -21*xj[-2] - 3*xj[-3] + xj[-4])  / (24*dx)

        return deriv
        # can recursively make gradient also a profile class
        # need to test this

    def log_gradient(self):
        print("calling (old) log gradient")
        # this is actually the gradient of the log...

        with np.errstate(divide='ignore', invalid='ignore'):
            gradlog = np.nan_to_num(self.gradient() / self.profile )

        return gradlog

    def midpoint_gradient(self):
        '''
        Take finite difference of N grid points
        to return gradient on (N-1) half-grid points
        '''
        x = self.profile
        diff = x[1:] - x[:-1]

        ax = self.axis
        dx = ax[1] - ax[0] # assumes equally spaced dx

        grad = diff/ dx
        return grad

    def midpoint_log_gradient(self):
        # this is actually the gradient of the log...

        midpoints = self.profile[1:] - self.profile[:-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            gradlog = np.nan_to_num(self.midpoint_gradient() / self.midpoints )
            #gradlog = np.nan_to_num(self.midpoint_gradient() / midpoints )

        return gradlog

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


    # operator overloads that automatically dereference the profiles
    def __add__(A,B):
        if isinstance(B, A.__class__):
            return A.profile + B.profile
        else:
            return A.profile + B

    def __sub__(A,B):
        if isinstance(B, A.__class__):
            return A.profile - B.profile
        else:
            return A.profile - B

    def __mul__(A,B):
        if isinstance(B, A.__class__):
            return A.profile * B.profile
        else:
            return A.profile * B

    def __truediv__(A,B):
        if isinstance(B, A.__class__):
            return A.profile / B.profile
        else:
            return A.profile / B

    def __rmul__(A,B):
        return A.__mul__(B)


class Flux_profile():

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

