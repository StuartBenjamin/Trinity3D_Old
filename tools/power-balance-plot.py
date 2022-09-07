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

time  =      np.array( data['time'  ] ) 
n     =      np.array( data['n'     ] ) 
pi    =      np.array( data['pi'    ] ) 
pe    =      np.array( data['pe'    ] ) 

profile_data = data['profiles']
N_rho    = profile_data['N_radial']
rho_edge = profile_data['rho_edge']
axis        = np.linspace(0,rho_edge,N_rho) # radial axis

# unused
P_fusion_Wm3 = np.array( data['P_fusion_Wm3'] )
P_brems_Wm3 = np.array( data['P_brems_Wm3'] )
fusion_rate = np.array( data['fusion_rate'] )
nu_ei_Hz = np.array( data['nu_ei_Hz'] )


###
pb = data['power balance']
force_n       =  pb['force_n']  
force_pi      =  pb['force_pi'] 
force_pe      =  pb['force_pe'] 
Ei            =  pb['Ei']       
Ee            =  pb['Ee']       
Gi            =  pb['Gi']       
Ge            =  pb['Ge']       
Hi            =  pb['Hi']       
He            =  pb['He']       
P_fusion      =  pb['P_fusion'] 
P_brems       =  pb['P_brems']  
aux_source_n  =  pb['aux_source_n']  
aux_source_pi =  pb['aux_source_pi']
aux_source_pe =  pb['aux_source_pe']

def plot_power_balance(t=0):

    # sanity check: do all of these terms have the same units?

    rax = axis[:-1]

    fig,axs = plt.subplots(1,3, figsize=(12,4) )
    axs[0].plot(rax, force_n      [t]     , '.-', label='turbulent particle flux')
    axs[0].plot(rax, aux_source_n [t][:-1], '.-', label='auxiliary particle source')

    axs[1].plot(rax, force_pi     [t]     , '.-', label='turbulent heat flux')
    axs[1].plot(rax, Ei           [t][:-1], '.-', label='collisional heat exchange')
    axs[1].plot(rax, Gi           [t][:-1], '.-', label='G term')
    axs[1].plot(rax, Hi           [t][:-1], '.-', label='H term')
    axs[1].plot(rax, P_fusion     [t][:-1], '.-', label='alpha heating')
    axs[1].plot(rax, aux_source_pi[t][:-1], '.-', label='auxiliary ion heating')

    axs[2].plot(rax, force_pe     [t]     , '.-', label='turbulent heat flux')
    axs[2].plot(rax, Ee           [t][:-1], '.-', label='collisional heat exchange')
    axs[2].plot(rax, Ge           [t][:-1], '.-', label='G term')
    axs[2].plot(rax, He           [t][:-1], '.-', label='H term')
    axs[2].plot(rax, P_brems      [t][:-1], '.-', label='bremstrahlung radiation') 
    axs[2].plot(rax, aux_source_pe[t][:-1], '.-', label='auxiliary electron heating')

    dn  = force_n[t]  + aux_source_n [t][:-1]                              
    dpi = force_pi[t] + aux_source_pi[t][:-1] + P_fusion[t][:-1] + Ei[t][:-1]
    dpe = force_pe[t] + aux_source_pe[t][:-1] + P_brems [t][:-1] + Ee[t][:-1] 
    axs[0].plot(rax, dn , '--', color='gray', label=r'$\Delta n$')
    axs[1].plot(rax, dpi, '--', color='gray', label=r'$\Delta p_i$')
    axs[2].plot(rax, dpe, '--', color='gray', label=r'$\Delta p_e$')

    axs[0].set_title('n')
    axs[1].set_title('pi')
    axs[2].set_title('pe')

    plt.suptitle('t = {}'.format(t))

    axs[0].legend(frameon=False)
    axs[1].legend(frameon=False)
    axs[2].legend(frameon=False)


## plot
path = "tmp/"
import os
if not os.path.exists(path):
    os.makedirs(path)

for t in np.arange(len(force_n)):
    plot_power_balance(t=t)
    fout = f'{path}t={t:03d}.png'
    plt.savefig(fout)
    plt.clf()
    print('saved', fout)

