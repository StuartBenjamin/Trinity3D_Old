import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simps
from scipy.constants import e, k as kB

path = 'Fusion_cross_sections/'

# Reactant masses in atomic mass units (u).
u = 1.66053906660e-27
masses = {'D': 2.014, 'T': 3.016, '3He': 3.016, '11B': 11.009305167,
          'p': 1.007276466620409}

### Define a dictionary of available Fusion Cross Sections
xs_names = {'D-T': 'D_T_-_a_n.txt',              # D + T -> a + n
            'D-D_a': 'D_D_-_T_p.txt',            # D + D -> T + p
            'D-D_b': 'D_D_-_3He_n.txt',          # D + D -> 3He + n
            'D-3He': 'D_3He_-_4He_p.txt',        # D + 3He -> a + p
            'p-B': 'p_11B_-_3a.txt',             # p + 11B -> 3a
            'T-T': 'T_T_-_4He_n_n.txt',          # T + T -> 4He + 2n
            'T-3He_a': 'T_3He_-_n_p_4He.txt',    # T + 3He -> 4He + n + p
            'T-3He_b': 'T_3He_-_D_4He.txt',      # T + 3He -> 4He + D
            '3He-3He': '3He_3He_-_p_p_4He.txt',  # 3He + 3He -> 4He + 2p
           }

xs_labels = {'D-T': '$\mathrm{D-T}$',
            'D-D': '$\mathrm{D-D}$',
            'D-3He': '$\mathrm{D-^3He}$',
            'p-B': '$\mathrm{p-^{11}B}$',
            'T-T': '$\mathrm{T-T}$',
            'T-3He': '$\mathrm{T-^3He}$',
            '3He-3He': '$\mathrm{^3He-^3He}$',
            }

# Energy grid, 1 - 1000 keV, evenly spaced in log-space.
Egrid = np.logspace(0, 5, 1000)

class Xsec:
    def __init__(self, m1, m2, xs):
        self.m1, self.m2, self.xs = m1, m2, xs
        self.mr = self.m1 * self.m2 / (self.m1 + self.m2)

    @classmethod
    def read_xsec(cls, filename, path='./', CM=True):
        """
        Read in cross section from filename and interpolate to energy grid.

        """

        #filename = path + filename
        E, xs = np.genfromtxt( path+filename, comments='#', skip_footer=2,
                              unpack=True)
        if CM:
            collider, target = filename.split('_')[:2]
            m1, m2 = masses[target], masses[collider]
            E *= m1 / (m1 + m2)

        xs = np.interp(Egrid, E*1.e3, xs*1.e-28)
        return cls(m1, m2, xs)

    def __add__(self, other):
        return Xsec(self.m1, self.m2, self.xs + other.xs)

    def __mul__(self, n):
        return Xsec(self.m1, self.m2, n * self.xs)
    __rmul__ = __mul__

    def __getitem__(self, i):
        return self.xs[i]

    def __len__(self):
        return len(self.xs)



def get_reactivity(xs, T):
    """Return reactivity, <sigma.v> in cm3.s-1 for temperature T in keV."""

    T = T[:, None]

    fac = 4 * np.pi / np.sqrt(2 * np.pi * xs.mr * u)
    fac /= (1000 * T * e)**1.5
    fac *= (1000 * e)**2 
    func = fac * xs.xs * Egrid * np.exp(-Egrid / T)
    I = np.trapz(func, Egrid, axis=1)
    # Convert from m3.s-1 to cm3.s-1
    return I * 1.e6

def alpha_heating_DT(n_profile, T_profile, f_tritium=0.5): 

    nT = n_profile * f_tritium
    nD = n_profile * (1 - f_tritium)
    T_profile_keV = T_profile / 1e3

    sv = get_reactivity(xs['D-T'], T_profile_keV) / 1e6 # divide by 1e6 to convert from cm^-3 to m^-3. 
    rate = nT * nD * sv  # reaction rate : second^{-1}, m^{-3}.

    E_fusion = 17.6e6 * e     # fusion energy in Joules
    E_alpha = E_fusion / 5    # alpha energy

    P_alpha = rate * E_alpha  # alpha heating power profile in W / m3
    return P_alpha

def alpha_heating_D_T(D_profile, T_profile, Ti_profile): 

    Ti_profile_keV = Ti_profile / 1e3

    sv = get_reactivity(xs['D-T'], Ti_profile_keV) / 1e6 # divide by 1e6 to convert from cm^-3 to m^-3. 
    rate = D_profile * T_profile * sv  # reaction rate : second^{-1}, m^{-3}.

    E_fusion = 17.6e6 * e     # fusion energy in Joules
    E_alpha = E_fusion / 5    # alpha energy

    P_alpha = rate * E_alpha  # alpha heating power profile in W / m3
    return P_alpha

def radiation_bremstrahlung(n_profile, T_profile, Zeff=1,
                                                  n_ref = 1e20, # /m3
                                                  T_ref = 1e3 , # eV
                                                ):

    # P[W/cm3] = 1.69e-3 * Ne[cc] * Te[eV] * \sum( Z**2 N[cc] )
    #    NRL Eq 30 (pg 58, version 2019)

    # Use quasineutrality and assume only species present are deuterium and possibly tritium + impurity:
    # 
    # ne = n_D + n_T + Z_I * n_I
    # also, zeff * ne = n_D + n_T + n_I * Z_I**2

    n = n_profile # units are given by n_ref, T_ref
    T = T_profile #   changing units not yet implemented

    P_brem = 5.34e3 * Zeff * n**2 * np.sqrt(T)  # W/m3
    return P_brem

def radiation_cyclotron(n_profile, T_profile, Zeff=1,
                                                  n_ref = 1e20, # /m3
                                                  T_ref = 1e3 , # eV
                                                  B     = 5   , # T
                                                ):

    # NRL 2019, eq (34)
    # P[W/cc] = 6.21e-28 * B^2[G] * ne[cc] * Te[eV]

    # P[W/m3] = 6.21e-3 * B^2[T] * ne[1e20 m3] * Te[keV]
    #    when do relativistic effects become important?

    P_cyc = 6.21e-3 * B**2 * n_profile * T_profile  # W / m3
    return P_cyc # untested (!)


### Load cross sections (used in computing heating)
xs = {}
for xs_id, xs_name in xs_names.items():
    xs[xs_id] = Xsec.read_xsec(xs_name,path=path)
# Total D + D fusion cross section is due to equal contributions from the above two processes.
xs['D-D']   = xs['D-D_a'] + xs['D-D_b']
xs['T-3He'] = xs['T-3He_a'] + xs['T-3He_b']






