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

    Updated 16 October 2022, T. M. Qian
'''

fin = sys.argv[1]
data = np.load(fin, allow_pickle=True).tolist()

time  =      np.array( data['time'  ] ) 
n     =      np.array( data['n'     ] ) 
pi    =      np.array( data['pi'    ] ) 
pe    =      np.array( data['pe'    ] ) 

t_idx  =      np.array( data['t_idx'] ) 
p_idx  =      np.array( data['p_idx'] ) 

N_profiles = len(time)

Ti = pi/n
Te = pe/n

profile_data = data['profiles']
N_rho    = profile_data['N_radial']
rho_edge = profile_data['rho_edge']
axis        = np.linspace(0,rho_edge,N_rho) # radial axis

# unused
#P_fusion_Wm3 = np.array( data['P_fusion_Wm3'] )
#P_brems_Wm3 = np.array( data['P_brems_Wm3'] )
#fusion_rate = np.array( data['fusion_rate'] )
#nu_ei_Hz = np.array( data['nu_ei_Hz'] )

## set up color
import matplotlib.pylab as pl

N_steps = np.max(t_idx)
warm_map = pl.cm.autumn(np.linspace(1,0.25,N_steps))

###
pb = data['power balance']
force_n       =  np.array( pb['force_n']       )  
force_pi      =  np.array( pb['force_pi']      ) 
force_pe      =  np.array( pb['force_pe']      ) 
Ei            =  np.array( pb['Ei']            ) 
Ee            =  np.array( pb['Ee']            ) 
Gi            =  np.array( pb['Gi']            ) 
Ge            =  np.array( pb['Ge']            ) 
Hi            =  np.array( pb['Hi']            ) 
He            =  np.array( pb['He']            ) 
P_fusion      =  np.array( pb['P_fusion']      ) 
P_brems       =  np.array( pb['P_brems']       ) 
aux_source_n  =  np.array( pb['aux_source_n']  ) 
aux_source_pi =  np.array( pb['aux_source_pi'] )
aux_source_pe =  np.array( pb['aux_source_pe'] )

alpha_ion_frac = np.array( data['alpha_ion_heating_fraction'] )
alpha_ion = P_fusion * alpha_ion_frac

def plot_power_balance(t,axs):

    # sanity check: do all of these terms have the same units?

    #fig,axs = plt.subplots(1,2, figsize=(12,5) )
    axs[0].clear()
    rax = axis[:-1]


    axs[0].plot(rax, force_pi     [t]     , '.-', label='turbulent heat flux')
    axs[0].plot(rax, Ei           [t][:-1], '.-', label='collisional heat exchange')
#    axs[0].plot(rax, Gi           [t][:-1], '.-', label='G term')
#    axs[0].plot(rax, Hi           [t][:-1], '.-', label='H term')
    axs[0].plot(rax, aux_source_pi[t][:-1], '.-', label='auxiliary ion heating')
    axs[0].plot(rax, alpha_ion[t][:-1], '.-', label='alpha heating')

    dn  = force_n[t]  + aux_source_n [t][:-1]                              
    dpi = force_pi[t] + aux_source_pi[t][:-1] + P_fusion[t][:-1] + Ei[t][:-1]
    dpe = force_pe[t] + aux_source_pe[t][:-1] + P_brems [t][:-1] + Ee[t][:-1] 
    axs[0].plot(rax, dpi, '--', color='gray', label=r'$\Delta p_i$')
    axs[1].plot(axis, Ti[t] , '.-', color=warm_map[t_idx[t]])

    axs[0].set_title('pi power balance')
    axs[1].set_title('Ti profile')

    axs[0].legend(frameon=False)
    plt.suptitle(rf"(t,p) = {t_idx[t]}, {p_idx[t]} :: $\tau$ = {time[t]:.3f}")

## plot
path = "tmp/"
import os
if not os.path.exists(path):
    os.makedirs(path)

fig,axs = plt.subplots(1,2, figsize=(12,5) )

for t in np.arange(N_profiles):
    plot_power_balance(t,axs)
    fout = f'{path}t={t:03d}.png'
    plt.savefig(fout)
    print(f"saved: {fout}")

