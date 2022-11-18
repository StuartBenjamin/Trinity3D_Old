import numpy as np
import sys
import matplotlib.pyplot as plt

import profiles as pf

'''
    Plot information about fluxes

    15 October 2022, T. M. Qian
'''

fin = sys.argv[1]
data = np.load(fin, allow_pickle=True).tolist()

rec = data['flux_record']
Q0 =      rec['Q0']
Q1 =      rec['Q1']
dQ =      rec['dQ']
kT =      rec['kT']
dk =      rec['dk']
N = len(Q0)

## set up color
import matplotlib.pylab as pl
warm_map = pl.cm.autumn(np.linspace(1,0.25,N))
cool_map = pl.cm.Blues(np.linspace(0.25,1,N))
green_map = pl.cm.YlGn(np.linspace(0.25,1,N))
purple_map = pl.cm.Purples(np.linspace(0.25,1,N))

fig,axs = plt.subplots(1,5, figsize=(13,3.5))
for t in np.arange(N):

    axs[0].plot( Q0[t], '.-' ,color=green_map[t] )
    axs[1].plot( Q1[t], '.-' ,color=green_map[t] )
    axs[2].plot( kT[t], '.-' ,color=green_map[t] )
    axs[3].plot( dk[t], '.-' ,color=green_map[t] )
    axs[4].plot( dQ[t], '.-' ,color=green_map[t] )

axs[0].set_title(r"$Q_0$")
axs[1].set_title(r"$Q_1$")
axs[2].set_title(r"$L_T$")
axs[3].set_title(r"$\Delta = 0.1 L_T$")
axs[4].set_title(r"$Q' = (Q_1 - Q_0)/ \Delta$")

#axs[0].set_yscale('log')
#axs[1].set_yscale('log')
#axs[4].set_yscale('log')

plt.tight_layout()
plt.show()

