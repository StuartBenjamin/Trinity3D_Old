import numpy as np
from profiles import GridProfile, FluxProfile
from collections import OrderedDict

m_proton_cgs = 1.67e-24 # mass of proton in grams

class SpeciesDict():
    
    def __init__(self, inputs, grid):
        species_list_params = inputs.get('species', {})
        self.N_species = len(species_list_params)
        self.N_radial = grid.N_radial
        self.grid = grid

        # create dictionary of species objects, keyed by species type (e.g. 'deuterium', 'electron', etc)
        self.species_dict = OrderedDict()
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
            self.species_dict[s.type] = s

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
                # make a list of species types that will evolve density
                self.n_evolve_list.append(s.type)
            if s.temperature_equal_to != None:
                s.evolve_temperature = False
                self.T_equalto_list.append(s.type)
            if s.evolve_temperature: 
                self.N_temp_profiles = self.N_temp_profiles + 1
                # make a list of species types that will evolve temperature
                self.T_evolve_list.append(s.type)

        # the last density profile can be set by quasineutrality,
        # so don't include it in count of evolved profiles
        if self.N_dens_profiles > 0:
            self.N_dens_profiles = self.N_dens_profiles - 1
            self.qneut_species = self.species_dict[self.n_evolve_list[-1]]
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
            assert self.species_dict[t].temperature_equal_to in self.species_dict.keys(), f"Error: cannot set '{t}' temperature equal to non-existent species '{self.species_dict[t].temperature_equal_to}'"

        # label adiabatic species in dictionary
        if adiabatic_species_count == 1:
            self.adiabatic_species = self.species_dict[adiabatic_type]
            self.has_adiabatic_species = True

        # label reference species in dictionary
        if reference_species_count == 0:
            self.ref_species = self.species_dict[first_type]
        else:
            self.ref_species = self.species_dict[reference_type]

        print(f"This calculation contains {[s for s in self.species_dict]} species.")
        print(f"Evolving densities: {self.n_evolve_list}")
        if self.qneut_species:
            print(f"The '{self.qneut_species.type}' density will be set by quasineutrality.")
        print(f"Evolving temperatures: {self.T_evolve_list}")
        for t in self.T_equalto_list:
            print(f"The '{t}' temperature will be set equal to the {self.species_dict[t].temperature_equal_to} temperature.")

        if self.has_adiabatic_species:
            print(f"The '{self.adiabatic_species.type}' species will be treated adiabatically.")

        print(f"Using '{self.ref_species.type}' as the reference species for turbulence calculations.")

        print(f"Total number of (parallelizable) flux tube calculations per step = {(self.N_radial-1)*(1 + len(self.n_evolve_list) + len(self.T_evolve_list))}.")

        #print("Base profiles:")
        #kns, kts = self.get_perturbed_fluxgrads()
        #print(kns[:,0])
        #print(kts)

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
            nt_vec.append(self.species_dict[t].n().profile)

        for t in self.T_evolve_list:
            nt_vec.append(self.species_dict[t].p().profile)
            
        nt_vec = np.concatenate(nt_vec)

        return nt_vec

    def get_profs_from_vec(self, nt_vec):
        '''
        copy values from nt_vec vector into the species profiles
        '''
        offset = 0
        ndens = 0
        charge_density = np.zeros(self.N_radial)
        for s in self.species_dict.values():
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

        for s in self.species_dict.values():
            if s.evolve_temperature:
                s.p().profile = nt_vec[offset:offset+self.N_radial]
                offset = offset + self.N_radial

    def get_profiles_on_flux_grid(self, normalize=False, a_ref=1.0, vt_sqrt_2=False):
        
        ns = np.zeros( (self.N_species, self.N_radial-1) )
        Ts = np.zeros( (self.N_species, self.N_radial-1) )
        nus = np.zeros( (self.N_species, self.N_radial-1) )

        if normalize:
            n_ref = self.ref_species.n().toFluxProfile()
            T_ref = self.ref_species.T().toFluxProfile()
            vt_ref = 9.79e3*(1e3*T_ref/self.ref_species.mass)**0.5 # m/s
            if vt_sqrt_2:
                vt_ref = vt_ref*np.sqrt(2.0)

        for i, s in enumerate(self.species_dict.values()):
            n = s.n().toFluxProfile()
            T = s.T().toFluxProfile()
            nu_ss = s.collision_frequency(s).toFluxProfile()

            if normalize:
                n = n/n_ref
                T = T/T_ref
                nu_ss = nu_ss*a_ref/vt_ref

            ns[i,:] = n
            Ts[i,:] = T
            nus[i,:] = nu_ss

        return ns, Ts, nus

    def get_grads_on_flux_grid(self, pert_n=None, pert_T=None, rel_step = 0.2, abs_step = 0.3):

        kns = np.zeros( (self.N_species, self.N_radial-1) )
        kts = np.zeros( (self.N_species, self.N_radial-1) )

        for i, s in enumerate(self.species_dict.values()):
            kn, kt = s.get_fluxgrads()
            for j in np.arange(len(kn)):
                kns[i,j] = kn[j]
                kts[i,j] = kt[j]
                if pert_n == s.type:
                    # perturb density gradient at fixed pressure gradient
                    kns[i,j] = kn[j] + abs_step
                    kts[i,j] = kt[j] - abs_step
                if pert_T == s.type:
                    kts[i,j] = max(kt[j]*(1+rel_step), kt[j] + abs_step)

                # if perturbing density, maintain quasineutrality of density gradient in species evolved by quasineutrality
                if self.qneut_species and s.type == self.qneut_species.type and pert_n != None:
                    # perturb density gradient at fixed pressure gradient
                    kns[i,j] = kn[j] - self.species_dict[pert_n].Z*abs_step/s.Z
                    kts[i,j] = kt[j] + self.species_dict[pert_n].Z*abs_step/s.Z
                
        return kns, kts

    def get_masses(self, normalize=False):
        ms = np.zeros(self.N_species)
        if normalize:
            m_ref = self.ref_species.mass
        else:
            m_ref = 1.0

        for i, s in enumerate(self.species_dict.values()):
            ms[i] = s.mass/m_ref
      
        return ms

    def get_charges(self, normalize=False):
        Zs = np.zeros(self.N_species)
        if normalize:
            Z_ref = self.ref_species.Z
        else:
            Z_ref = 1.0

        for i, s in enumerate(self.species_dict.values()):
            Zs[i] = s.Z/Z_ref
      
        return Zs
        
    def get_types_ion_electron(self):
        ie_type = []
        for s in self.species_dict.values():
            if s.type == "electron":
                ie_type.append("electron")
            else:
                ie_type.append("ion")

        return ie_type

    def set_flux(self, pflux_sj, qflux_sj):
        for i, s in enumerate(self.species_dict.values()):
            s.pflux = FluxProfile(pflux_sj[i, :], self.grid)
            s.qflux = FluxProfile(qflux_sj[i, :], self.grid)

    def set_dflux_dkn(self, stype, dpflux_dkn_sj, dqflux_dkn_sj):
        for i, s in enumerate(self.species_dict.values()):
            s.dpflux_dkn[stype] = FluxProfile(dpflux_dkn_sj[i, :], self.grid)
            s.dqflux_dkn[stype] = FluxProfile(dqflux_dkn_sj[i, :], self.grid)

    def set_dflux_dkT(self, stype, dpflux_dkT_sj, dqflux_dkT_sj):
        for i, s in enumerate(self.species_dict.values()):
            s.dpflux_dkT[stype] = FluxProfile(dpflux_dkT_sj[i, :], self.grid)
            s.dqflux_dkT[stype] = FluxProfile(dqflux_dkT_sj[i, :], self.grid)

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

        # init flux profiles with zeros. these will be set by SpeciesDict.set_flux
        self.pflux = FluxProfile(0, grid)
        self.qflux = FluxProfile(0, grid)
        # init flux jacobians as empty dictionaries. these will be set by SpeciesDict.set_dflux_dk*
        self.dpflux_dkn = {}
        self.dqflux_dkn = {}
        self.dpflux_dkT = {}
        self.dqflux_dkT = {}

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

    def beta_on_flux_grid(self, B_ref):
        # compute beta
        # p_prof is in units of 10^20 m^-3*keV
        # B_ref is in units of T

        # convert p and B to cgs
        p_cgs = self.p_prof*1e17
        B_cgs = B_ref*1e4
        return 4.03e-11*p_cgs.toFluxProfile()/(B_cgs*B_cgs)

    def collision_frequency(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV
        # the below formula is from the NRL formulary, which is in cgs units (except for temperatures in eV)

        Z_s = self.Z
        m_s = self.mass*m_proton_cgs
        n_s = self.n()*1e14  # 10^20 m^-3 -> cm^-3
        T_s = self.T()*1e3    # keV -> eV

        Z_u = other.Z
        m_u = other.mass*m_proton_cgs
        n_u = other.n()*1e14  # 10^20 m^-3 -> cm^-3
        T_u = other.T()*1e3    # keV -> eV

        logLambda = self.logLambda(other)

        nu = 1.8e-19 * np.sqrt( m_s * m_u) * (Z_s * Z_u)**2 * n_u \
                   * logLambda / ( m_s * T_u + m_u * T_s )**1.5

        return nu

    def logLambda(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV
        # the below formula is from the NRL formulary, which is in cgs units (except for temperatures in eV)

        Z_s = self.Z
        m_s = self.mass*m_proton_cgs
        n_s = self.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_s = self.T().profile*1e3    # keV -> eV

        Z_u = other.Z
        m_u = other.mass*m_proton_cgs
        n_u = other.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_u = other.T().profile*1e3    # keV -> eV

        if self.type == "electron" and other.type == "electron": # ee
            lamb = 23.5 - np.log( n_s**0.5 / T_s**1.25 ) \
                - np.sqrt( 1e-5 + ( np.log(T_s) - 2)**2 / 16 )
        elif self.type == "electron": # ei
            lamb = 24.0 - np.log( n_s**0.5 / T_s )
        elif other.type == "electron": # ie
            lamb = 24.0 - np.log( n_u**0.5 / T_u )
        else: # ii
            lamb = 23.0 - np.log(
                 (Z_s * Z_u) * (m_s + m_u) \
                 / ( m_s * T_u + m_u * T_s ) \
                 * ( n_s * Z_s**2 / T_s + n_u * Z_u**2 / T_u )**0.5 \
                 )
        return lamb

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


