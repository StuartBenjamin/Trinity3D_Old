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

        self.N_dens_profiles = 0
        self.N_temp_profiles = 0
        self.N_profiles = 0
        self.n_evolve_list = []
        self.T_evolve_list = []
        self.T_equalto_list = []

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
                self.N_dens_profiles = self.N_dens_profiles + 1
                self.n_evolve_list.append(s.type)
            if s.temperature_equal_to != None:
                s.evolve_temperature = False
                self.T_equalto_list.append(s.type)
            if s.evolve_temperature: 
                self.N_temp_profiles = self.N_temp_profiles + 1
                self.T_evolve_list.append(s.type)

        # the last density profile can be set by quasineutrality,
        # so don't include it in count of evolved profiles
        if self.N_dens_profiles > 0:
            self.N_dens_profiles = self.N_dens_profiles - 1
            self.qneut_species = self.species[self.n_evolve_list[-1]]
            self.n_evolve_list = self.n_evolve_list[:-1]
        else:
            self.qneut_species = None
        # number of evolved profiles
        self.N_profiles = self.N_dens_profiles + self.N_temp_profiles

        self.nt_vec = self.get_vec_from_profs()

        # sanity checks
        assert adiabatic_species_count <= 1, "Error: cannot have more than one adiabatic species"
        assert reference_species_count <= 1, "Error: cannot have more than one species set as reference species"
        for t in self.T_equalto_list:
            assert self.species[t].temperature_equal_to in self.species.keys(), f"Error: cannot set '{t}' temperature equal to non-existent species '{self.species[t].temperature_equal_to}'"

        # label adiabatic species in dictionary
        if adiabatic_species_count == 1:
            self.adiabatic_species = self.species[adiabatic_type]
        else:
            self.adiabatic_species = None

        # label reference species in dictionary
        if reference_species_count == 0:
            self.ref_species = self.species[first_type]
        else:
            self.ref_species = self.species[reference_type]

        print(f"This calculation contains {[s for s in self.species]} species.")
        print(f"Evolving densities: {self.n_evolve_list}")
        if self.qneut_species:
            print(f"The '{self.qneut_species.type}' density will be set by quasineutrality.")
        print(f"Evolving temperatures: {self.T_evolve_list}")
        for t in self.T_equalto_list:
            print(f"The '{t}' temperature will be set equal to the {self.species[t].temperature_equal_to} temperature.")

        if self.adiabatic_species:
            print(f"The '{self.adiabatic_species.type}' species will be treated adiabatically.")

        print(f"Using '{self.ref_species.type}' as the reference species for turbulence calculations.")

        #print("Base profiles:")
        #kn, kt = self.get_perturbed_fluxgrads()
        #print([l for l in kn])
        #print([l for l in kt])

        #print("Perturbed profiles:")
        #for t in self.n_evolve_list:
        #    print("perturb", t, "density")
        #    kn, kt = self.get_perturbed_fluxgrads(pert_n=t, pert_T=None)
        #    print([l for l in kn])
        #    print([l for l in kt])

        #for t in self.T_evolve_list:
        #    print("perturb", t, "temp")
        #    kn, kt = self.get_perturbed_fluxgrads(pert_T=t, pert_n=None)
        #    print([l for l in kn])
        #    print([l for l in kt])

    def get_vec_from_profs(self):
        '''
        copy and concatenate values from species profiles into a numpy vector
        '''
        nt_vec = []

        for t in self.n_evolve_list:
            nt_vec.append(self.species[t].n().profile)

        for t in self.T_evolve_list:
            nt_vec.append(self.species[t].p().profile)
            
        nt_vec = np.concatenate(nt_vec)

        return nt_vec

    def get_profs_from_vec(self, nt_vec):
        '''
        copy values from nt_vec vector into the species profiles
        '''
        offset = 0
        ndens = 0
        charge_density = np.zeros(self.N_radial)
        for s in self.species.values():
            if s.evolve_density:
                ndens = ndens + 1
                if ndens <= self.N_dens_profiles:
                    s.n().profile = nt_vec[offset:offset+self.N_radial]
                    charge_density = charge_density + s.Z*s.n()
                    offset = offset + self.N_radial
            else:
                # non-evolved species still count towards charge density
                charge_density = charge_density + s.Z*s.n()

        # use quasineutrality to set density of last evolved species.
        # this needs to happen outside the above loop to ensure charge_density
        # has been computed correctly.
        self.qneut_species.set_n(-charge_density/s.Z)

        for s in self.species.values():
            if s.evolve_temperature:
                s.p().profile = nt_vec[offset:offset+self.N_radial]
                offset = offset + self.N_radial

    def get_perturbed_fluxgrads(self, pert_n=None, pert_T=None, rel_step = 0.2, abs_step = 0.3):

        kns = []
        kts = []
        for s in self.species.values():
            kn, kt = s.get_fluxgrads()
            for j in np.arange(len(kn)):
                if pert_n == s.type:
                    # perturb density gradient at fixed pressure gradient
                    kn[j] = kn[j] + abs_step
                    kt[j] = kt[j] - abs_step
                if pert_T == s.type:
                    kt[j] = max(kt[j]*(1+rel_step), kt[j] + abs_step)

                # if perturbing density, maintain quasineutrality of density gradient in species evolved by quasineutrality
                if self.qneut_species and s.type == self.qneut_species.type and pert_n != None:
                    # perturb density gradient at fixed pressure gradient
                    kn[j] = kn[j] - self.species[pert_n].Z*abs_step/s.Z
                    kt[j] = kt[j] + self.species[pert_n].Z*abs_step/s.Z
                
            kns.append(kn)
            kts.append(kt)

        return kns, kts

class Species():

    def __init__(self, sp, grid):

        self.type = sp.get('type', "deuterium")
        self.is_adiabatic = sp.get('adiabatic', False)
        self.use_as_reference = sp.get('use_as_reference', False)
        density_parameters = sp.get('density', {})
        temperature_parameters = sp.get('temperature', {})

        # flags controlling density and temperature evolution
        self.evolve_density = density_parameters.get('evolve', True)
        self.evolve_temperature = temperature_parameters.get('evolve', True)
        self.temperature_equal_to = temperature_parameters.get('equal_to', None)

        # physical parameters
        self.mass = sp.get('mass') or self.get_mass()  # mass in units of proton mass
        self.Z = sp.get('Z') or self.get_charge()      # charge in units of e
        
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
        self.n_prof = GridProfile((self.n_core - self.n_edge)*(1 - (grid.rho_axis/grid.rho_edge)**2) + self.n_edge, grid)
        T_prof = GridProfile((self.T_core - self.T_edge)*(1 - (grid.rho_axis/grid.rho_edge)**2) + self.T_edge, grid)
        self.p_prof = self.n_prof * T_prof

        # save a copy of initial profiles separately
        self.n_init = self.n_prof
        self.p_init = self.p_prof

    def n(self):
        return self.n_prof

    def p(self):
        return self.p_prof

    def T(self):
        return self.p_prof / self.n_prof

    def set_n(self, n):
        self.n_prof = n

    def set_p(self, p):
        self.p_prof = p

    def get_fluxgrads(self):
        kap_n = -1*self.n_prof.log_gradient_as_FluxProfile()
        T_prof = self.p_prof / self.n_prof
        kap_t = -1*T_prof.log_gradient_as_FluxProfile()
        return kap_n, kap_t

    def init_sources(self, grid):
        ### sources
        # temp, Gaussian model. Later this should be adjustable
        Gaussian = np.vectorize(self.Gaussian)
        self.aux_source_n = GridProfile(Gaussian(grid.rho_axis, A=self.Sn_height, sigma=self.Sn_width, x0=self.Sn_center), grid)
        self.aux_source_p = GridProfile(Gaussian(grid.rho_axis, A=self.Sp_height, sigma=self.Sp_width, x0=self.Sp_center), grid)
        
        self.source_model = 'Gaussian'
        
    # for a particle and heat sources
    def Gaussian(self, x, A=2, sigma=.3, x0=0):
        exp = - ( (x - x0) / sigma)**2  / 2
        return A * np.e ** exp

    def get_mass(self):
        ''' 
        Look-up table for species mass in units of proton mass.
        '''
        if self.type == "hydrogen":
            return 1.0
        if self.type == "deuterium":
            return 2.0
        elif self.type == "tritium":
            return 3.0
        elif self.type == "electron":
            return 0.000544617021
        else:
            assert False, f"species '{self.type}' has unknown mass. use mass parameter in input file (with units of proton mass)."

    def get_charge(self):
        ''' 
        Look-up table for species charge in units of e.
        '''
        if self.type == "hydrogen":
            return 1.0
        if self.type == "deuterium":
            return 1.0
        elif self.type == "tritium":
            return 1.0
        elif self.type == "electron":
            return -1.0
        else:
            assert False, f"species '{self.type}' has unknown charge. use Z parameter in input file (with units of e)."


