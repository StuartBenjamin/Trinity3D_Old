import numpy as np
import matplotlib.pyplot as plt

import fusion_lib as flb


# Make profiles
N_radial = 11
radial_grid = np.linspace(0,1,N_radial)
n_profile_m3  = np.ones(N_radial) * 1e20
edge_temp_keV = 0.1
T_profile_keV = (1 - radial_grid**2) * 15 + edge_temp_keV
#T_profile_keV = np.ones(N_radial) * 15 ## constant for debugging
T_profile_eV = T_profile_keV*1e3

# Set up geometry
R_major_m = 6
a_minor_m = 3
area_profile_m2 = (2 * np.pi * R_major_m) * (2 * np.pi * radial_grid * a_minor_m)
dr = a_minor_m / (N_radial-1)
V_profile_m3  = area_profile_m2 * dr 


# compute fusion
P_fusion_Wm3 = flb.alpha_heating_DT( n_profile_m3, T_profile_eV )
P_fusion_MWm3 = P_fusion_Wm3/1e6
P_fusion_MW   = P_fusion_MWm3 * V_profile_m3
P_fusion_MWm  = P_fusion_MWm3 * area_profile_m2
print(' Total fusion power (integral Simpson): {} MW'.format( flb.simps(P_fusion_MWm, a_minor_m*radial_grid, dx=dr) ) ) # this one looks accurate, but off by factor 1/2
print(' Total fusion power (Pfus * Vol): {} MW'.format( np.sum(P_fusion_MW) ) )

# compute Bremstrahlung
P_brem_Wm3 = flb.radiation_bremstrahlung(n_profile_m3/1e20, T_profile_keV) 
P_brem_MWm3 = P_brem_Wm3 / 1e6

# copmute Cyclotron
P_cyc_Wm3 = flb.radiation_cyclotron(n_profile_m3/1e20, T_profile_keV)
P_cyc_MWm3 = P_cyc_Wm3 / 1e6

Total_Volume = (2*np.pi*R_major_m) * (np.pi * a_minor_m**2)
print('Total Volume: ', Total_Volume)
print('Fusion power density: ', P_fusion_MWm3[0])
print('Total Fusion: ', Total_Volume * P_fusion_MWm3[0])

fig,axs = plt.subplots(2,3, figsize=(12,3))

axs[0,0].plot( radial_grid, n_profile_m3 , '.-' )
axs[0,1].plot( radial_grid, T_profile_keV, '.-' )
#axs[2].plot( radial_grid, sv           , '.-' )
#axs[3].plot( radial_grid, rate         , '.-' )
axs[1,0].plot( radial_grid, P_fusion_MWm3 , '.-' )
axs[1,1].plot( radial_grid, P_brem_MWm3         , '.-' )
axs[1,2].plot( radial_grid, P_cyc_MWm3         , '.-' )
axs[1,0].plot( radial_grid, P_fusion_MWm3 , '.-' )
#axs[4].plot( radial_grid, P_fusion_MW , '.-' )

axs[0,0].set_title('density')
axs[0,1].set_title('temperature')
axs[1,1].set_title('Bremstrahlung Power Density')
axs[1,0].set_title('Fusion power density')
axs[1,2].set_title('Cyclotron power density')


plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()
