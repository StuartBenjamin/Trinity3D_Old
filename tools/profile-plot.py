import numpy as np
import sys
import matplotlib.pyplot as plt

import profiles as pf

'''
    This program plots TRINITY .npy output
    Usage: 
        python plot-trinity.py [trinity-log.npy]

    It reads data from TRINITY stored in a python dictionary,
    and makes a number of plots showing profiles and fluxes,
    all in one pannel.

    Updated 1 April 2022, T. M. Qian
'''

fin = sys.argv[1]
data = np.load(fin, allow_pickle=True).tolist()

time   =      np.array( data['time'  ] ) 
n      =      np.array( data['n'     ] ) 
pi     =      np.array( data['pi'    ] ) 
pe     =      np.array( data['pe'    ] ) 
Gamma  =      np.array( data['Gamma' ] ) 
Qi     =      np.array( data['Qi'    ] ) 
Qe     =      np.array( data['Qe'    ] ) 
aLn  =      np.array( data['aLn' ] ) 
aLpi =      np.array( data['aLpi'] ) 
aLpe =      np.array( data['aLpe'] ) 

P_fusion_Wm3 = np.array( data['P_fusion_Wm3'] )
P_brems_Wm3 = np.array( data['P_brems_Wm3'] )
fusion_rate = np.array( data['fusion_rate'] )
nu_ei_Hz = np.array( data['nu_ei_Hz'] )

# write minor radius
#        major radius, and other geoemtry info?

Ti = pi / n
Te = pe / n

settings = data['system']
profile_data = data['profiles']
N_rho    = profile_data['N_radial']
rho_edge = profile_data['rho_edge']
try:
    axis = profile_data['rho_axis']
except:
    print(" backwards compatibility: rho_axis not found on log (file was created prior to 7/17) defaulting rho_inner to 0 in plots")
    axis        = np.linspace(0,rho_edge,N_rho) # radial axis
mid_axis    = (axis[1:] + axis[:-1])/2
pf.rho_axis = axis

# sources
source_n  = profile_data['source_n' ] 
source_pi = profile_data['source_pi']
source_pe = profile_data['source_pe']

def init_profile(x,debug=False):

    X = pf.Profile(x, grad=True, half=True, full=True)
    return X


N = len(time)

fig,axs = plt.subplots( 2, 6, figsize=(15,8) )

# run settings
alpha = settings['alpha']
dtau  = settings['dtau']
N_steps = settings['N_steps']
rlabel = rf'$\alpha = {alpha} : d\tau = {dtau:.3e} : N_\tau = {N_steps} : \Delta\tau = {dtau*N_steps:.1e}$'
plt.suptitle(rlabel)


## set up color
import matplotlib.pylab as pl
warm_map = pl.cm.autumn(np.linspace(1,0,N))
cool_map = pl.cm.Blues(np.linspace(0,1,N))
green_map = pl.cm.YlGn(np.linspace(0,1,N))
purple_map = pl.cm.Purples(np.linspace(0,1,N))

# time evolution
for t in np.arange(N):

    # plot profiles
    axs[0,0].plot(axis,n [t] ,'.-', color=green_map[t])

    # plot fluxes
    axs[0,1].plot(axis,Te[t] ,'.-', color=cool_map[t])
    axs[0,1].plot(axis,Ti[t] ,'.-', color=warm_map[t])

    axs[0,2].plot(axis,pe[t] ,'.-', color=cool_map[t])
    axs[0,2].plot(axis,pi[t] ,'.-', color=warm_map[t])
           
    # plot diffusivity
    axs[0,4].plot(mid_axis,Qe[t] ,'x-', color=cool_map[t])
    axs[0,4].plot(mid_axis,Qi[t] ,'x-', color=warm_map[t])
    axs[0,5].plot(mid_axis,Gamma[t] ,'x-', color=green_map[t])

    axs[0,3].plot(axis, source_pe, 'C0.-')
    axs[0,3].plot(axis, source_pi, 'C1.-')
    axs[0,3].plot(axis, source_n , 'C2.-')
    #axs[1,4].plot(axis, source_pi * data['norms']['pressure_source_scale'], 'C1.-')

    axs[1,4].plot( aLpi[t] - aLn[t], Qi[t] ,'.-', color=warm_map[t])
    axs[1,4].plot( aLpe[t] - aLn[t], Qe[t] ,'.-', color=cool_map[t])
    axs[1,5].plot( aLn[t],Gamma[t] ,'.-', color=green_map[t])

 #   axs[1,0].plot(axis, fusion_rate[t], '.-', color=purple_map[t])
    axs[1,0].plot(axis, P_fusion_Wm3[t]/1e6, '.-', color=purple_map[t])
    axs[1,1].plot(axis, P_brems_Wm3[t]/1e6, '.-', color=purple_map[t])
    axs[1,2].plot(axis, nu_ei_Hz[t], '.-', color=cool_map[t])

#axs[0,0].set_ylim( bottom=0 )
#axs[1,0].set_ylim( bottom=0 )
#axs[2,0].set_ylim( bottom=0 )

axs[0,0].set_title(r'density [10$^{20}$ m$^{-3}$]')
axs[0,1].set_title('temperature [keV]')
axs[0,2].set_title(r'pressure [10$^{20}$m$^{-3}$ keV]')
axs[0,4].set_title('heat flux')
axs[0,5].set_title('particle flux')
axs[0,3].set_title('sources')
axs[1,4].set_title(r'$Q_i(L_{T_i})$')
axs[1,5].set_title(r'$\Gamma(L_n)$')
#axs[1,4].set_title(r'sources [MW/m$^{-3}$]')

#axs[1,0].set_title('fusion rate')
axs[1,0].set_title(r'fusion power density [MW/m$^{-3}$]')
axs[1,1].set_title(r'bremstralung radiation [MW/m$^{-3}$]')
axs[1,2].set_title('collisional heat exchange [Hz]')

plt.tight_layout()

plt.show()

