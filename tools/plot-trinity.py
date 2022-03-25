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

    Updated 15 March 2022, T. M. Qian
'''

fin = sys.argv[1]
data = np.load(fin, allow_pickle=True).tolist()

time  =      np.array( data['time'  ] ) 
n     =      np.array( data['n'     ] ) 
pi    =      np.array( data['pi'    ] ) 
pe    =      np.array( data['pe'    ] ) 
Gamma =      np.array( data['Gamma' ] ) 
Qi    =      np.array( data['Qi'    ] ) 
Qe    =      np.array( data['Qe'    ] ) 

# write minor radius
#        major radius, and other geoemtry info?

settings = data['system']
N_rho    = settings['N_radial']
rho_edge = settings['rho_edge']
axis        = np.linspace(0,rho_edge,N_rho) # radial axis
mid_axis    = (axis[1:] + axis[:-1])/2
pf.rho_axis = axis


def init_profile(x,debug=False):

    X = pf.Profile(x, grad=True, half=True, full=True)
    return X

N = len(time)

fig,axs = plt.subplots( 3, 5, figsize=(13,9) )

for t in np.arange(N):

    axs[0,0].plot(axis,n [t] ,'.-')
    axs[1,0].plot(axis,pi[t] ,'.-')
    axs[2,0].plot(axis,pe[t] ,'.-')

    axs[0,1].plot(mid_axis,Gamma [t] ,'.-')
    axs[1,1].plot(mid_axis,Qi[t]     ,'.-')
    axs[2,1].plot(mid_axis,Qe[t]     ,'.-')

    density     = init_profile(n[t] )    # try doing this outside the loop, it worked accidentally
    pressure_i  = init_profile(pi[t])
    pressure_e  = init_profile(pe[t])

    axs[0,2].plot(axis,-density.grad_log.profile    ,'.-')
    axs[1,2].plot(axis,-pressure_i.grad_log.profile ,'.-')
    axs[2,2].plot(axis,-pressure_e.grad_log.profile ,'.-')
           
    axs[0,3].plot(axis,-density   .grad.profile ,'.-')
    axs[1,3].plot(axis,-pressure_i.grad.profile ,'.-')
    axs[2,3].plot(axis,-pressure_e.grad.profile ,'.-')
           
    axs[0,4].plot(mid_axis,Gamma[t] / -density.grad.midpoints ,'.-')
    axs[1,4].plot(mid_axis,Qi[t] / -pressure_i.grad.midpoints ,'.-')
    axs[2,4].plot(mid_axis,Qe[t] / -pressure_e.grad.midpoints ,'.-')

axs[0,0].set_ylabel('n' , rotation='horizontal', labelpad=15)
axs[1,0].set_ylabel('pi', rotation='horizontal', labelpad=15)
axs[2,0].set_ylabel('pe', rotation='horizontal', labelpad=15)

axs[0,0].set_title('profile')
axs[0,1].set_title('flux')
axs[0,2].set_title(r'$\nabla$')
axs[0,3].set_title(r'$L^{-1}$')
axs[0,4].set_title('diffusivity')

plt.tight_layout()

import pdb
pdb.set_trace()

plt.show()
