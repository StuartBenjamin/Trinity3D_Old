import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset

'''
    This script takes a list of GX outputs,
    and plots Q(time) overlaying curves if multiple inputs are given.

    usage: python qflux.py [list of .nc files]')

    Updated 21 Jan 2022
    tqian@pppl.gov
'''

data = []
fins = []
for fname in sys.argv[1:]:

    if (fname.find('nonZonal') > 0): continue

    try:
        data.append(Dataset(fname, mode='r'))
        fins.append(fname)
        print(' read', fname)
    
    except:
        print(' usage: python qflux.py [list of .nc files]')



plt.figure()
cols = ['C0','C1','C2','C3','C4','C5','C6','C7']

j = 0
flux = []
for f in data:

    t = f.variables['time'][:]
    q = f.groups['Fluxes'].variables['qflux'][:,0]
    # check for nans
    if ( np.isnan(q).any() ):
         print('  nans found')
         q = np.nan_to_num(q)
    # fix nan_to_num here
    plt.plot(t,q,'.-',label=fins[j])

    # median of a sliding median
    N = len(q)
    med = np.median( [ np.median( q[::-1][:k] ) for k in np.arange(1,N)] )
    plt.axhline(med, color=cols[j%8], ls='--')
    flux.append(med)
    j+=1

print(flux)

#plt.yscale('log')
plt.ylabel('qflux')
plt.xlabel('time')
plt.grid()

N_files = len(data)
if N_files < 20:
    plt.legend()
else:
    plt.title(f"Showing {N_files} GX runs")
plt.show()
