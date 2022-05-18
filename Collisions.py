import numpy as np
import matplotlib.pyplot as plt

m_proton_cgs = 1.67e-24 # mass of proton in grams

# The class keep track of mass, charge, and profiles for each species.
#     It also computes log Lambda and nu for collisional energy exchange.
class Collision_Model():

    def __init__(self):

        self.n_vec = []
        self.T_vec = []
        self.m_vec = []
        self.Z_vec = []
        # it would be better if I defined these lists, using the assumed dimensions
        # right now I'm assuming NRL dimensions (cc, eV, proton mass, proton charge)

        self.isIon = []
        self.species = []

    # imports (n,T) profile from Trinity
    def add_species(self, 
                      n_profile_20,     # density profile   (1e20 / m3)
                      p_profile_n20keV, # pressure profile  (keV)
                      mass   = 1,       # mass of species   (proton mass)
                      charge = 1,       # charge of species (proton charge)
                      ion    = True,    # bool Ion or Electron
                      name='Hydrogen',  # optional name
                    ):
        
        T_profile_keV = p_profile_n20keV / n_profile_20
        
        self.T_vec.append( T_profile_keV * 1e3 )    # keV     -> eV
        self.n_vec.append( n_profile_20 * 1e14 )    # 1e20/m3 -> cc
        self.m_vec.append( mass )
        self.Z_vec.append( charge )

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
        self.T_vec[0] = Ti_eV
        self.T_vec[1] = Te_eV
        self.n_vec[0] = ni_cc
        self.n_vec[1] = ne_cc

    def export_species(self,j):

        #isIon = self.isIon[j]
        Z     = self.Z_vec[j]
        m     = self.m_vec[j]

        #return isIon, Z, m
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

    # collisional heat exchange, as defined in NRL
    def energy_collisions(self, s, u):


        m = np.array( self.m_vec ) * m_proton_cgs  # proton mass -> grams
        n = np.array( self.n_vec ) * 1e14          # 1e20/m3     -> cc
        T = np.array( self.T_vec ) * 1e3           # keV         -> eV
        Z = self.Z_vec

        nu = 1.8e-19 * np.sqrt( m[s] * m[u]) * (Z[s] * Z[u])**2 * n[u] \
                   * self.logLambda(s,u) \
                   / ( m[s] * T[u] + m[u] * T[s] )**1.5
    
        return nu

    ## These member functions compute elements N x N matrix using the existing profiles
    def logLambda(self, s, u):
    
        # compute lamb := log(Lambda)
        #    all logarithms used here are natural logs (base e)
        #    the expressions below assume T_vec is given in keV and n_vec is given in 1e20/m3,
        #    which are the units expected from Trinity.
    
        n = np.array( self.n_vec ) * 1e14      # 1e20/m3     -> cc
        T = np.array( self.T_vec ) * 1e3       # keV         -> eV
        m = self.m_vec       # only appears as dimensionless ratio
        Z = self.Z_vec

        pair = self.identify_pair(s,u) 

        if (pair == 'ii'):
            lamb = 23.0 - np.log(
                 10**2.5 * (Z[s] * Z[u]) * (m[s] + m[u]) \
                 / ( m[s] * T[u] + m[u] * T[s] ) \
                 * np.sqrt( n[s] * Z[s]**2 / T[s] + n[u] * Z[u]**2 / T[u] ) \
                 )
    
        elif (pair == 'ee'):
            lamb = 23.5 - np.log( 10**3.24 * n[s]**0.5 / T[s]**1.25 ) \
                - np.sqrt( 1e-5 + ( np.log(1e3 * T[s]) - 2)**2 / 16 )
    
        elif (pair == 'ie'):
            lamb = 24.0 - np.log( 1e4 * np.sqrt( n[u] ) / T[u] )
    
        elif (pair == 'ei'):
            lamb = 24.0 - np.log( 1e4 * np.sqrt( n[s] ) / T[s] )
    
        else: # not used
            print('  ERROR: in collision operator logLambda() must be given keyword')
            print('            pair == ii ee ie or ei')
            return 0
    
        return lamb
    
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
        
        self.T_vec.append( Temp_profile_eV )
        self.n_vec.append( n_profile_m3 )
        self.m_vec.append( mass )
        self.Z_vec.append( charge )

        self.isIon.append( ion )
        self.species.append( name ) # should I also save units somehow?
        # maybe I had better save the units in the variable name, 
        # and then call the name with units assumed


    # assume inputs are in TRANSP units
    #    m3, eV, proton mass
    def energy_collisions_nrl(self, s, u):

        m = np.array( self.m_vec ) * m_proton_cgs  # proton mass -> grams
        n = np.array( self.n_vec ) / 1e6           # m-3         -> cc
        T = self.T_vec 
        Z = self.Z_vec

        nu = 1.8e-19 * np.sqrt( m[s] * m[u]) * (Z[s] * Z[u])**2 * n[u] \
                   * self.logLambda_nrl(s,u) \
                   / ( m[s] * T[u] + m[u] * T[s] )**1.5
    
        return nu

    ## These member functions compute elements N x N matrix using the existing profiles
    def logLambda_nrl(self, s, u):
    
        # compute lamb := log(Lambda)
        #    all logarithms used here are natural logs (base e)
        #    assime [n] = /m3, [T] = eV, [m] = amu, [Z] = n charge
    
        n = np.array( self.n_vec ) / 1e6          # m-3          -> cc
        T = self.T_vec  
        Z = self.Z_vec
        m = self.m_vec

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





