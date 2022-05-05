import numpy as np
import matplotlib.pyplot as plt

m_proton_cgs = 1.67e-24

# We need a species class to keep track of mass, charge, and collisions
#     or we can hard code it for now
class Species():

    def __init__(self):


        self.m_vec = []
        self.Z_vec = []
        self.T_vec = []
        self.n_vec = []

        self.isIon = []
        self.species = []

    def add_species(self, 
                      n_profile_20,     # density profile from Trinity (1e20 / m3)
                      Temp_profile_keV, # temperature profile from Trinity (keV)
                      mass   = 1,       # mass of species   (proton mass)
                      charge = 1,       # charge of species (proton charge)
                      ion    = True,    # boolean Ion or Electron
                      name='Hydrogen',  # optional name
                    ):
        
        self.T_vec.append( Temp_profile_keV )
        self.n_vec.append( n_profile_20 )
        self.m_vec.append( mass )
        self.Z_vec.append( charge )

        self.isIon.append( ion )
        self.species.append( name )

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


    def add_species_transp(self, 
                      n_profile_m3,     # density profile from Trinity (1e20 / m3)
                      Temp_profile_eV,  # temperature profile from Trinity (keV)
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
        self.species.append( name )


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
        #    the expressions below assume T_vec is given in keV and n_vec is given in 1e20/m3,
        #    which are the units expected from Trinity.
    
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


