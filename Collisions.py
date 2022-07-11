import numpy as np
import matplotlib.pyplot as plt

m_proton_cgs = 1.67e-24 # mass of proton in grams

'''
The class keeps track of mass, charge, and profiles for each species.
It also computes log Lambda and nu for collisional energy exchange.
'''
class Collision_Model():

    def __init__(self):

        self.n_cc = []
        self.T_eV = []
        self.m_mp = []
        self.Z_e = []
        # it would be better if I defined these lists, using the assumed dimensions
        # right now I'm assuming NRL dimensions (cc, eV, proton mass, proton charge)

        self.isIon = []
        self.species = []

    '''
    For a given species, imports (n,T) profile from Trinity
    Assumes n,T are arrays (radial profiles) of equal length.

    Assumes n_profile_20 is in units of 1e20 / m3
    Assumes p_profile_n20keV is in units of 1e20 / m3 keV
    Assumes mass_p is measured in proton masses
    Assumes charge_p is measured in proton charges

    The ion = T/F boolean is used for {ee, ei, ie, ii} logic.
    '''
    def add_species(self, 
                      n_profile_20,     # density profile   (1e20 / m3)
                      p_profile_n20keV, # pressure profile  (keV)
                      mass_p   = 1,       # mass of species   (proton mass)
                      charge_p = 1,       # charge of species (proton charge)
                      ion    = True,    # bool Ion or Electron
                      name='Hydrogen',  # optional name
                    ):
        
        T_profile_keV = p_profile_n20keV / n_profile_20
        
        self.T_eV.append( T_profile_keV * 1e3 )    # keV     -> eV
        self.n_cc.append( n_profile_20 * 1e14 )    # 1e20/m3 -> cc
        self.m_mp.append( mass_p ) 
        self.Z_e .append( charge_p )

        self.isIon.append( ion )
        self.species.append( name )

    def update_profiles(self, engine):

        pi_20keV = engine.pressure_i.profile
        pe_20keV = engine.pressure_e.profile
        ni_20    = engine.density.profile
        ne_20    = engine.density.profile

        ni_cc = ni_20 * 1e14
        ne_cc = ne_20 * 1e14
        Ti_eV = pi_20keV/ni_20 * 1e3
        Te_eV = pe_20keV/ne_20 * 1e3

        # hard coded to the init convention that (D,e)
        #    how can this be improved, in a way that is generalizable to multiple ions?
        self.T_eV[0] = Ti_eV
        self.T_eV[1] = Te_eV
        self.n_cc[0] = ni_cc
        self.n_cc[1] = ne_cc

    def export_species(self,j):

        Z     = self.Z_e[j]
        m     = self.m_mp[j]
        return Z, m



    # needed for multi-species, but not necessary for two-species
    #   because the anti-symmetric matrix has only one element that matters
    def compute_collision_matrix(self):

        N_species = len(self.species)

        sax = np.arange(N_species)
        nu_matrix = [ [ self.energy_collisions_nrl(s,u) for u in sax] for s in sax]

        # skip self interactions 
        #nu_matrix = []
        #for s in sax:
        #    for u in sax:
        #        if (u == s):
        #            continue
        #        nu_matrix.append( self.energy_collisions(s,u) )

        self.collision_term = np.sum( nu_matrix, axis=1 )

        # for debuging
        self.nu   = np.array(nu_matrix)
        self.lamb = np.array( [ [ self.logLambda_nrl(s,u) for u in sax] for s in sax] )


    def plot_collision_matrix(self):
    
        fig, ax = plt.subplots(1,2,figsize=(8,5))
    
        rax = np.linspace( 0,1, len(self.nu[0,0]) )
    
    
        ax[0].plot( rax, self.nu[0,0], label='ii' )
        ax[0].plot( rax, self.nu[0,1], label='ie' )
        ax[0].plot( rax, self.nu[1,0], label='ei' )
        ax[0].plot( rax, self.nu[1,1], label='ee' )
        ax[0].set_title(r'$\nu_\epsilon$')
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[0].grid()
    
        ax[1].plot( rax, self.lamb[0,0], label='ii' )
        ax[1].plot( rax, self.lamb[0,1], label='ie' )
        ax[1].plot( rax, self.lamb[1,0], label='ei' )
        ax[1].plot( rax, self.lamb[1,1], label='ee' )
        ax[1].set_title(r'$\log \Lambda$')
        ax[1].grid()
        plt.show()

    # identifies collision type (ii, ie, ei, ee)
    def identify_pair(self,s,u):

        pair = ''
        for j in [s,u]:
            if ( self.isIon[j] ):
                pair += 'i'
            else:
                pair += 'e'

        return pair

    ### For test function ###
    def add_species_transp(self, 
                      n_profile_m3,     # density profile from Transp (m3)
                      Temp_profile_eV,  # temperature profile from Transp (eV)
                      mass   = 1,       # mass of species   (proton mass)
                      charge = 1,       # charge of species (proton charge)
                      ion    = True,    # boolean Ion or Electron
                      name='Hydrogen',  # optional name
                    ):
        
        self.T_eV.append( Temp_profile_eV )
        self.n_cc.append( n_profile_m3 / 1e6 )
        self.m_mp.append( mass )
        self.Z_e.append( charge )

        self.isIon.append( ion )
        self.species.append( name ) # should I also save units somehow?
        # maybe I had better save the units in the variable name, 
        # and then call the name with units assumed


    def energy_collisions_nrl(self, s, u):
        #    assume [n] = cc, [T] = eV, [m] = g, [Z] = n charge

        m = np.array( self.m_mp ) * m_proton_cgs  # proton mass -> grams
        n = np.array( self.n_cc )
        T = self.T_eV
        Z = self.Z_e

        nu = 1.8e-19 * np.sqrt( m[s] * m[u]) * (Z[s] * Z[u])**2 * n[u] \
                   * self.logLambda_nrl(s,u) \
                   / ( m[s] * T[u] + m[u] * T[s] )**1.5
    
        return nu

    ## These member functions compute elements N x N matrix using the existing profiles
    def logLambda_nrl(self, s, u):
    
        # compute lamb := log(Lambda)
        #    all logarithms used here are natural logs (base e)
        #    assume [n] = cc, [T] = eV, [m] = amu, [Z] = n charge
    
        n = np.array( self.n_cc ) #/ 1e6          # m-3          -> cc
        T = self.T_eV
        Z = self.Z_e
        m = self.m_mp

        pair = self.identify_pair(s,u) 

        if (pair == 'ii'):
            lamb = 23.0 - np.log(
                 (Z[s] * Z[u]) * (m[s] + m[u]) \
                 / ( m[s] * T[u] + m[u] * T[s] ) \
                 * ( n[s] * Z[s]**2 / T[s] + n[u] * Z[u]**2 / T[u] )**0.5 \
                 )
    
        elif (pair == 'ee'):
            lamb = 23.5 - np.log( n[s]**0.5 / T[s]**1.25 ) \
                - np.sqrt( 1e-5 + ( np.log(1e3 * T[s]) - 2)**2 / 16 )
    
        elif (pair == 'ie'):
            lamb = 24.0 - np.log( n[u]**0.5 / T[u] )
    
        elif (pair == 'ei'):
            lamb = 24.0 - np.log( n[s]**0.5 / T[s] )
    
        else: # not used
            print('  ERROR: in collision operator logLambda() must be given keyword')
            print('            pair == ii ee ie or ei')
            return 0
    
        return lamb


    def compute_nu_ei(self):

        nu_ei = self.energy_collisions_nrl(0,1)

        self.nu_ei = nu_ei
        return nu_ei


