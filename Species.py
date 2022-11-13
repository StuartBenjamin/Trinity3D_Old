import numpy as np
from profiles import GridProfile
from collections import OrderedDict

class SpeciesDict():
    
    def __init__(self, inputs, grid):
        species_list_params = inputs.get('species', {})
        self.N_species = len(species_list_params)
        self.N_radial = grid.N_radial
        self.grid = grid

        # create dictionary of species objects, keyed by species type (e.g. 'deuterium', 'electron', etc)
        self.species = OrderedDict()
        self.has_adiabatic_species = False
        reference_species_count = 0
        adiabatic_species_count = 0
        first_type = None

        self.n_evolve = OrderedDict()
        self.T_evolve = OrderedDict()
        for sp in species_list_params:
            # initialize a species object
            s = Species(sp, grid)
            # store species object in dictionary keyed by species type (e.g. 'deuterium', 'electron', etc)
            self.species[s.type] = s

            # check for adiabatic species
            if s.is_adiabatic:
                adiabatic_species_count = adiabatic_species_count + 1
                adiabatic_type = s.type
                s.evolve_density = False
                s.evolve_temperature = False

            # check for reference species
            if s.use_as_reference:
                reference_species_count = reference_species_count + 1
                reference_type = s.type
            
            # save the type of the first listed species
            if first_type == None:
                first_type = s.type

            if s.evolve_density:
                self.n_evolve["n_"+s.type] = s.n.profile
            if s.temperature_equal_to != None:
                s.evolve_temperature = False
            if s.evolve_temperature: 
                self.T_evolve["T_"+s.type] = s.T.profile

        # remove the last n_evolve because it can be set by quasineutrality
        if len(self.n_evolve) > 0:
            self.n_evolve.popitem()

        # create dictionary of evolved densities and temperatures
        self.nT_dict = OrderedDict(**self.n_evolve, **self.T_evolve)

        # number of evolved profiles
        self.N_profiles = len(self.nT_dict)

        # copy nT_dict values into a stacked numpy vector
        self.nT_vec = self.nT_dict_copyto_vec() 

        # sanity checks
        assert adiabatic_species_count <= 1, "Error: cannot have more than one adiabatic species"
        assert reference_species_count <= 1, "Error: cannot have more than one species set as reference species"

        # label adiabatic species in dictionary
        if adiabatic_species_count == 1:
            self.species['adiabatic'] = self.species[adiabatic_type]

        # label reference species in dictionary
        if reference_species_count == 0:
            self.species['reference'] = self.species[first_type]
        else:
            self.species['reference'] = self.species[reference_type]

        print(f"Using {self.species['reference'].type} as reference species for flux tube calculations.")

    def nT_dict_copyto_vec(self):
        '''
        copy values from nT_dict into a numpy vector nT_vec
        '''
        nT_vec = np.concatenate(list(self.nT_dict.values()))
        return nT_vec

    def nT_dict_copyfrom_vec(self, vec):
        '''
        copy values from nT_vec vector into the ordered nT_dict and species dictionary
        '''
        offset = 0
        for k, v in self.nT_dict.items():
            self.nT_dict[k] = vec[offset:offset+self.N_radial]
            if k[0] == 'n':
                self.species[k[2:]].n = GridProfile(self.nT_dict[k], self.grid)
            if k[0] == 'T':
                self.species[k[2:]].T = GridProfile(self.nT_dict[k], self.grid)
            offset = offset + self.N_radial
        

class Species():

    def __init__(self, sp, grid):

        self.type = sp.get('type', "deuterium")
        self.is_adiabatic = sp.get('adiabatic', False)
        self.use_as_reference = sp.get('use_as_reference', False)
        density_parameters = sp.get('density', {})
        temperature_parameters = sp.get('temperature', {})

        self.evolve_density = density_parameters.get('evolve', True)
        self.evolve_temperature = temperature_parameters.get('evolve', True)
        self.temperature_equal_to = temperature_parameters.get('equal_to', None)
        
        # initial profiles
        self.n_core = density_parameters.get('core', 4)
        self.n_edge = density_parameters.get('edge', 4)
        self.T_core = temperature_parameters.get('core', 4)
        self.T_edge = temperature_parameters.get('edge', 4)
        self.p_core = self.n_core * self.T_core
        self.p_edge = self.n_edge * self.T_edge
        
        # sources
        density_source = sp.get('density_source', {})
        pressure_source = sp.get('pressure_source', {})
        self.Sn_height = density_source.get('height', 0)
        self.Sn_width  = density_source.get('width', 0.1)
        self.Sn_center = density_source.get('center', 0)
        self.Sp_height = pressure_source.get('height', 0)
        self.Sp_width  = pressure_source.get('width', 0.1)
        self.Sp_center = pressure_source.get('center', 0)
        
        # set initial profiles and sources
        self.init_profiles(grid)
        self.init_sources(grid)

    def init_profiles(self, grid):
        self.n = GridProfile((self.n_core - self.n_edge)*(1 - (grid.rho_axis/grid.rho_edge)**2) + self.n_edge, grid)
        self.T = GridProfile((self.T_core - self.T_edge)*(1 - (grid.rho_axis/grid.rho_edge)**2) + self.T_edge, grid)
        self.p = self.n * self.T

        # save a copy of initial profiles separately
        self.n_init = self.n
        self.T_init = self.T
        self.p_init = self.p

    def init_sources(self, grid):
        ### sources
        # temp, Gaussian model. Later this should be adjustable
        Gaussian = np.vectorize(self.Gaussian)
        self.aux_source_n = GridProfile(Gaussian(grid.rho_axis, A=self.Sn_height, sigma=self.Sn_width, x0=self.Sn_center), grid)
        self.aux_source_p = GridProfile(Gaussian(grid.rho_axis, A=self.Sp_height, sigma=self.Sp_width, x0=self.Sp_center), grid)
        
        self.source_model = 'Gaussian'
        
    def update_profiles(self):
        pass

    # for a particle and heat sources
    def Gaussian(self, x, A=2, sigma=.3, x0=0):
        exp = - ( (x - x0) / sigma)**2  / 2
        return A * np.e ** exp

