import numpy as np
import matplotlib.pyplot as plt

import fusion_lib as flb

# This tests the fusion profiles produced by JET profiles (shot 42982)
#      taken when PFUSION peaks at t_idx=292 (16.61 s)
#      6.075 MW (6075418.0)


Ti = np.array([ 8313.189,  8198.816,  8091.17 ,  7994.321,  7914.864,  7867.481,
        7821.454,  7779.283,  7713.844,  7625.038,  7508.127,  7365.848,
        7186.943,  6960.574,  6734.024,  6498.964,  6241.515,  5986.755,
        5735.176,  5497.67 ,  5272.698,  5051.919,  4841.107,  4651.061,
        4464.896,  4282.391,  4104.022,  3934.803,  3781.287,  3632.684,
        3500.889,  3394.745,  3293.195,  3159.308,  2988.603,  2817.899,
        2647.195,  2483.69 ,  2336.518,  2201.841])

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

# Deuterium
Nm1 = np.array([  3.18689100e+19,   3.19812500e+19,   3.20168000e+19,
         3.18938700e+19,   3.18162100e+19,   3.15786700e+19,
         3.13071500e+19,   3.10104200e+19,   3.07902900e+19,
         3.06022900e+19,   3.04252800e+19,   3.02206800e+19,
         3.00238400e+19,   2.98484800e+19,   2.96563700e+19,
         2.93891500e+19,   2.90105200e+19,   2.86407200e+19,
         2.82421500e+19,   2.78544800e+19,   2.75264500e+19,
         2.71993300e+19,   2.68870400e+19,   2.65961400e+19,
         2.62951900e+19,   2.59585400e+19,   2.55653400e+19,
         2.51203900e+19,   2.45696600e+19,   2.39511600e+19,
         2.32364900e+19,   2.24064600e+19,   2.16081500e+19,
         2.06955300e+19,   1.97823400e+19,   1.88901000e+19,
         1.80323700e+19,   1.69819900e+19,   1.59591400e+19,
         1.49725500e+19])

# Tritium
Nm2 = np.array([  2.43585400e+19,   2.44202500e+19,   2.44334700e+19,
         2.43420000e+19,   2.42890500e+19,   2.41150100e+19,
         2.39232100e+19,   2.37217400e+19,   2.35799200e+19,
         2.34595100e+19,   2.33484600e+19,   2.32251500e+19,
         2.31213800e+19,   2.30438300e+19,   2.29535900e+19,
         2.28045100e+19,   2.25744900e+19,   2.23567400e+19,
         2.21199000e+19,   2.18992200e+19,   2.17362700e+19,
         2.15804700e+19,   2.14410100e+19,   2.13273900e+19,
         2.12159700e+19,   2.10846500e+19,   2.09130000e+19,
         2.07031200e+19,   2.04162400e+19,   2.00884600e+19,
         1.96916800e+19,   1.92006000e+19,   1.87418000e+19,
         1.81938300e+19,   1.76537100e+19,   1.71389800e+19,
         1.66566900e+19,   1.59918900e+19,   1.53426800e+19,
         1.47454400e+19])

# Surface Area
Area = np.array([   3.690838,    7.380161,   11.06828 ,   14.75417 ,   18.4388  ,
         22.12032 ,   25.79927 ,   29.47423 ,   33.14775 ,   36.81909 ,
         40.48758 ,   44.1503  ,   47.80921 ,   51.46573 ,   55.121   ,
         58.77137 ,   62.41607 ,   66.05138 ,   69.67905 ,   73.29583 ,
         76.90294 ,   80.49858 ,   84.08411 ,   87.65808 ,   91.22311 ,
         94.77758 ,   98.32429 ,  101.862   ,  105.3933  ,  108.9184  ,
        112.4412  ,  115.9635  ,  119.4911  ,  123.029   ,  126.5876  ,
        130.1814  ,  133.8356  ,  137.5802  ,  141.4491  ,  145.4804  ])




# Set up geometry
R_major_m = 2.905
a_minor_m = 0.936

#area_profile_m2 = (2 * np.pi * R_major_m) * (2 * np.pi * radial_grid * a_minor_m)
#dr = a_minor_m / (N_radial-1)

radial_grid = np.linspace(0,1, len(Area) )
r_grid_m = np.linspace(0,a_minor_m, len(Area) )
dr = a_minor_m / len(Area) # approximate, what is a better way to get radial spacing?
V_profile_m3  = Area * dr 
P_fusion_Wm3 = flb.alpha_heating_D_T(Nm1, Nm2, Ti)
P_fusion_Wm  = P_fusion_Wm3 * Area


# compute fusion
#P_fusion_Wm3 = flb.alpha_heating_DT( n_profile_m3, T_profile_eV )
#P_fusion_MWm3 = P_fusion_Wm3/1e6
#P_fusion_MW   = P_fusion_MWm3 * V_profile_m3
#P_fusion_MWm  = P_fusion_MWm3 * area_profile_m2
#print(' Total fusion power (integral Simpson): {} MW'.format( flb.simps(P_fusion_MWm, a_minor_m*radial_grid, dx=dr) ) ) # this one looks accurate, but off by factor 1/2
#print(' Total fusion power (Pfus * Vol): {} MW'.format( np.sum(P_fusion_MW) ) )


#Total_Volume = (2*np.pi*R_major_m) * (np.pi * a_minor_m**2)
#print('Total Volume: ', Total_Volume)
#print('Fusion power density: ', P_fusion_MWm3[0])
#print('Total Fusion: ', Total_Volume * P_fusion_MWm3[0])

fig,axs = plt.subplots(2,3, figsize=(8,5))

axs[0,0].plot( radial_grid, Nm1 , '.-' )
axs[0,0].plot( radial_grid, Nm2 , '.-' )
axs[1,0].plot( radial_grid, Ti, '.-' )
axs[0,1].plot( radial_grid, P_fusion_Wm3 , '.-' )
axs[1,1].plot( radial_grid, Area , '.-' )
axs[0,2].plot( r_grid_m, P_fusion_Wm , '.-' )
#axs[1,1].plot( radial_grid, P_brem_MWm3         , '.-' )
#axs[1,2].plot( radial_grid, P_cyc_MWm3         , '.-' )
#axs[1,0].plot( radial_grid, P_fusion_MWm3 , '.-' )
#axs[4].plot( radial_grid, P_fusion_MW , '.-' )

axs[0,0].set_title('density [m-3]')
axs[1,0].set_title('Ion Temperature [eV]')
#axs[1,1].set_title('Bremstrahlung Power Density')
axs[0,1].set_title('Fusion power density [W/m3]')
axs[1,1].set_title('Area [m2]')
axs[0,2].set_title('Fusion power density [W/m]')
#axs[1,2].set_title('Cyclotron power density')


plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()
