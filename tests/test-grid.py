import unittest
import numpy as np
from Grid import Grid

class TestGrid(unittest.TestCase):

    def test_defaults(self):
        inputs = {}
        grid = Grid(inputs)
        self.assertEqual(grid.N_radial, 10)
        self.assertEqual(grid.rho_edge, 0.8)

        self.assertEqual(len(grid.rho_axis), grid.N_radial)
        self.assertEqual(len(grid.mid_axis), grid.N_radial-1)

        assert(isinstance(grid.rho_axis, np.ndarray))
        assert(isinstance(grid.mid_axis, np.ndarray))

    def test_nondefault(self):
        inputs = {'grid': {'N_radial' : 5, 'rho_edge' : 0.7}}
        grid = Grid(inputs)
        self.assertEqual(grid.N_radial, 5)
        self.assertEqual(grid.rho_edge, 0.7)

if __name__ == '__main__':
    unittest.main()
