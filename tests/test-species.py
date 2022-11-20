import unittest
import numpy as np
from Trinity_io import Trinity_Input
from Species import SpeciesDict
from Grid import Grid

class TestSpecies(unittest.TestCase):

    def test_1(self):
        inputs = {}
        grid = Grid(inputs)

        inputs = Trinity_Input('tests/test-species.in').input_dict
        species = SpeciesDict(inputs, grid)
        self.assertEqual(species.N_species, 3)

        self.assertEqual(species.has_adiabatic_species, False)
        self.assertEqual(species.ref_species.type, "deuterium")
        self.assertEqual(species.n_evolve_list, ['deuterium', 'electron'])
        self.assertEqual(species.qneut_species.type, "tritium")
        self.assertEqual(species.T_evolve_list, ['deuterium', 'tritium'])
        self.assertEqual(species.species_dict['electron'].temperature_equal_to, 'deuterium')
        self.assertEqual(species.species_dict['tritium'].temperature_equal_to, None)

    def test_2(self):
        inputs = {}
        grid = Grid(inputs)

        inputs = Trinity_Input('tests/test-species-2.in').input_dict
        species = SpeciesDict(inputs, grid)
        self.assertEqual(species.N_species, 3)

        self.assertEqual(species.has_adiabatic_species, True)
        self.assertEqual(species.ref_species.type, "tritium")

if __name__ == '__main__':
    unittest.main()
