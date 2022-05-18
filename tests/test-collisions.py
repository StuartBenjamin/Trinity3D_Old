import numpy as np
import matplotlib.pyplot as plt


import Collisions as clib


svec = clib.Collision_Model()


Ti = np.array([ 8313.189,  8198.816,  8091.17 ,  7994.321,  7914.864,  7867.481,
        7821.454,  7779.283,  7713.844,  7625.038,  7508.127,  7365.848,
        7186.943,  6960.574,  6734.024,  6498.964,  6241.515,  5986.755,
        5735.176,  5497.67 ,  5272.698,  5051.919,  4841.107,  4651.061,
        4464.896,  4282.391,  4104.022,  3934.803,  3781.287,  3632.684,
        3500.889,  3394.745,  3293.195,  3159.308,  2988.603,  2817.899,
        2647.195,  2483.69 ,  2336.518,  2201.841])

Te = np.array([ 7710.615,  7596.242,  7488.595,  7391.746,  7312.289,  7264.906,
        7218.879,  7176.708,  7111.269,  7022.462,  6905.552,  6763.273,
        6605.096,  6434.115,  6258.391,  6078.535,  5894.457,  5707.133,
        5518.803,  5327.799,  5140.474,  4957.849,  4780.883,  4631.766,
        4490.764,  4365.162,  4256.135,  4150.083,  4047.599,  3946.246,
        3834.933,  3696.768,  3563.54 ,  3410.122,  3247.951,  3086.895,
        2933.328,  2759.656,  2596.131,  2446.49 ])

Ne = np.array([  7.86138100e+19,   7.86367500e+19,   7.84880200e+19,
         7.80470600e+19,   7.74094500e+19,   7.64706000e+19,
         7.54969300e+19,   7.44791100e+19,   7.35900300e+19,
         7.28335600e+19,   7.22909000e+19,   7.19466000e+19,
         7.16479100e+19,   7.13736200e+19,   7.10058300e+19,
         7.05410200e+19,   6.99372400e+19,   6.92023300e+19,
         6.84440600e+19,   6.76470800e+19,   6.68504700e+19,
         6.60535700e+19,   6.52679400e+19,   6.45119700e+19,
         6.37637800e+19,   6.29698400e+19,   6.21313100e+19,
         6.12480900e+19,   6.02377800e+19,   5.91881100e+19,
         5.80323700e+19,   5.65910300e+19,   5.52011900e+19,
         5.35616400e+19,   5.18171500e+19,   5.00747000e+19,
         4.84021000e+19,   4.63733500e+19,   4.44607900e+19,
         4.27106200e+19])

Ni = Ne

svec.add_species_transp( Ni, Te, mass=2, charge=1, ion=True, name='Deuterium')
svec.add_species_transp( Ne, Ti, mass=1/1800, charge=-1, ion=False, name='electrons')

svec.compute_collision_matrix()
def show_debug(self):

    fig, ax = plt.subplots(1,2,figsize=(8,5))

    rax = np.linspace( 0,1, len(Ne) )


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

def show_profiles():

    fig, ax = plt.subplots(2,1,figsize=(5,3) )

    rax = np.linspace( 0,1, len(Ne) )
    ax[0].plot(rax, Ne, label='ne')
    ax[0].plot(rax, Ni, label='ni')
    ax[1].plot(rax, Te, label='Te')
    ax[1].plot(rax, Ti, label='Ti')

    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    plt.show()

show_debug(svec)
show_profiles()

import pdb
pdb.set_trace()
