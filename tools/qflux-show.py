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

    try:
        data.append(Dataset(fname, mode='r'))
        fins.append(fname)
        print(' read', fname)
    
    except:
        print(' usage: python qflux.py [list of .nc files]')



plt.figure()

j = 0
for f in data:
    t = f.variables['time'][:]
    q = f.groups['Fluxes'].variables['qflux'][:,0]

    #plt.plot(t,'.-',label=fins[j]) # plot time
    #plt.plot(q,'.-',label=fins[j]) # plot index
    plt.plot(t,q,'.-',label=fins[j])
    j+=1

plt.yscale('log')
plt.ylabel('qflux')
plt.xlabel('time')
plt.legend()
plt.grid()
plt.show()
