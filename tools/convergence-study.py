import numpy as np
import sys
import matplotlib.pyplot as plt

import profiles as pf

'''
    This tools shows Trinity convergence.
    Usage: 
        python plot-trinity.py [trinity-log.npy]

    Updated 13 October 2022, T. M. Qian
'''

fin = sys.argv[1]
data = np.load(fin, allow_pickle=True).tolist()

#time   =      np.array( data['time'  ] ) 
#n      =      np.array( data['n'     ] ) 
#pi     =      np.array( data['pi'    ] ) 
#pe     =      np.array( data['pe'    ] ) 
#Gamma  =      np.array( data['Gamma' ] ) 
#Qi     =      np.array( data['Qi'    ] ) 
#Qe     =      np.array( data['Qe'    ] ) 
#aLn  =      np.array( data['aLn' ] ) 
#aLpi =      np.array( data['aLpi'] ) 
#aLpe =      np.array( data['aLpe'] ) 

t_idx = np.array( data['t_idx'] )
p_idx = np.array( data['p_idx'] )

y_hist  = np.array( data['y_hist'] )
y_error = np.array( data['y_error'] )
chi_err = np.array( data['chi_error'] )

N_profiles = len(chi_err)
plt.figure()

# scatter plot
for j in np.arange(N_profiles):
    plt.plot(t_idx[j], chi_err[j], f'C{p_idx[j]}o')

# labels
max_iter = data['system']['max_newton_iter']
for j in np.arange(max_iter):
    plt.plot([],[], f'C{j}o', label=f'iter {j}')

plt.legend()
plt.grid()
plt.show()

import pdb
pdb.set_trace()
