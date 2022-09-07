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

p_source_scale = data['norms']['pressure_source_scale']


def init_profile(x,debug=False):

    X = pf.Profile(x, grad=True, half=True, full=True)
    return X


N = len(time)

fig,axs = plt.subplots( 2, 6, figsize=(65,8) )

# run settings
alpha = settings['alpha']
dtau  = settings['dtau']
N_steps = settings['N_steps']
rlabel = rf'$\alpha = {alpha} : d\tau = {dtau:.3e} : N_\tau = {N_steps} : \Delta\tau = {dtau*N_steps:.1e}$'
plt.suptitle(rlabel)


## set up color
import matplotlib.pylab as pl
warm_map = pl.cm.autumn(np.linspace(1,0.25,N))
cool_map = pl.cm.Blues(np.linspace(0.25,1,N))
green_map = pl.cm.YlGn(np.linspace(0.25,1,N))
purple_map = pl.cm.Purples(np.linspace(0.25,1,N))

n_leg = 5
t_plot_freq = int(np.rint(N/n_leg)) # Ensures we only plot a maximum of n_leg timesteps in the legend.

# time evolution
for t in np.arange(N):

    # plot profiles
    if t%t_plot_freq == 0:
        axs[0,0].plot(axis,n [t] ,'.-', color=green_map[t], label = '{:.2f}'.format(time[t]))
    else:
        axs[0,0].plot(axis,n [t] ,'.-', color=green_map[t])

    # plot fluxes
    if t == 0:
        axs[0,1].plot(axis,Te[t] ,'.-', color=cool_map[t], label = '$T_e$, {:.2f}'.format(time[t]))
        axs[0,1].plot(axis,Ti[t] ,'.:', color=warm_map[t], label = '$T_i$, {:.2f}'.format(time[t]))
    elif t == N-1:
        axs[0,1].plot(axis,Te[t] ,'.-', color=cool_map[t], label = '$T_e$, {:.2f}'.format(time[t]))
        axs[0,1].plot(axis,Ti[t] ,'.:', color=warm_map[t], label = '$T_i$, {:.2f}'.format(time[t]))
    else:
        axs[0,1].plot(axis,Te[t] ,'.-', color=cool_map[t])
        axs[0,1].plot(axis,Ti[t] ,'.:', color=warm_map[t])

    axs[0,2].plot(axis,pe[t] ,'.-', color=cool_map[t])
    axs[0,2].plot(axis,pi[t] ,'.:', color=warm_map[t])

    # plot diffusivity
    axs[0,4].plot(mid_axis,Qe[t] ,'x-', color=cool_map[t])
    axs[0,4].plot(mid_axis,Qi[t] ,'x:', color=warm_map[t])
    axs[0,5].plot(mid_axis,Gamma[t] ,'x-', color=green_map[t])

    axs[1,4].plot( aLpi[t] - aLn[t], Qi[t] ,'.:', color=warm_map[t])
    axs[1,4].plot( aLpe[t] - aLn[t], Qe[t] ,'.-', color=cool_map[t])
    axs[1,5].plot( aLn[t],Gamma[t] ,'.-', color=green_map[t])

 #   axs[1,0].plot(axis, fusion_rate[t], '.-', color=purple_map[t])
    axs[1,0].plot(axis, P_fusion_Wm3[t]/1e6, '.-', color=purple_map[t])
    axs[1,1].plot(axis, P_brems_Wm3[t]/1e6, '.-', color=purple_map[t])
    axs[1,2].plot(axis, nu_ei_Hz[t], '.-', color=cool_map[t])

axs[0,3].plot(axis, source_pe, 'C0.-', label = '$S_{p_e}$')
axs[0,3].plot(axis, source_pi, 'C1.:', label = '$S_{p_i}$')
axs[0,3].plot(axis, source_n , 'C2.-', label = '$S_{n}$')

# convert from Trinity units to MW/m3
axs[1,3].plot(axis, source_pe / p_source_scale * 1e-6, 'C0.-', label = '$S_{p_e}$')
axs[1,3].plot(axis, source_pi / p_source_scale * 1e-6, 'C1.:', label = '$S_{p_i}$')

#axs[0,0].set_ylim( bottom=0 )
#axs[1,0].set_ylim( bottom=0 )
#axs[2,0].set_ylim( bottom=0 )

axs[0,0].set_xlabel('$r/a$')
axs[0,1].set_xlabel('$r/a$')
axs[0,2].set_xlabel('$r/a$')
axs[0,3].set_xlabel('$r/a$')
axs[0,4].set_xlabel('$r/a$')
axs[0,5].set_xlabel('$r/a$')
axs[1,0].set_xlabel('$r/a$')
axs[1,1].set_xlabel('$r/a$')
axs[1,2].set_xlabel('$r/a$')
axs[1,3].set_xlabel('$r/a$')
axs[1,4].set_xlabel('$a/L_{{T_i}}$')
axs[1,5].set_xlabel('$a/L_{{T_i}}$')


axs[0,0].set_title(r'density [10$^{20}$ m$^{-3}$]')
axs[0,1].set_title('temperature [keV]')
axs[0,2].set_title(r'pressure [10$^{20}$m$^{-3}$ keV]')
axs[0,4].set_title('heat flux')
axs[0,5].set_title('particle flux')
axs[0,3].set_title('sources')
axs[1,4].set_title(r'$Q_i(L_{T_i})$')
axs[1,5].set_title(r'$\Gamma(L_n)$')
axs[1,3].set_title(r'sources [MW/m$^{-3}$]')

#axs[1,0].set_title('fusion rate')
axs[1,0].set_title('fusion power density \n [MW/m$^{-3}$]')
axs[1,1].set_title('bremstralung radiation \n [MW/m$^{-3}$]')
axs[1,2].set_title('collisional heat \n exchange [Hz]')

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

#plt.subplots_adjust(wspace = 0.4, hspace = 0.5)

#Legends
leg = axs[0,0].legend(loc='best', title = '$t v_{ti}/a$', fancybox=False, shadow=False,ncol=1)
plt.setp(leg.get_title())
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.65)
leg2 = axs[0,1].legend(loc='best', title = '$t v_{ti}/a$', fancybox=False, shadow=False,ncol=1)
plt.setp(leg2.get_title())
leg2.get_frame().set_edgecolor('k')
leg2.get_frame().set_linewidth(0.65)
leg4 = axs[0,3].legend(loc='best', fancybox=False, shadow=False,ncol=1)
leg4.get_frame().set_edgecolor('k')
leg4.get_frame().set_linewidth(0.65)
#plt.tight_layout()

plt.show()

