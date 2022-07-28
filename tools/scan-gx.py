import numpy as np
from netCDF4 import Dataset
import sys

import matplotlib.pyplot as plt

'''
This program is a tool for scanning over a set of gx outputs generated by Trinity.

    usage: python tools/scan-gx.py t*nc
    (assuming files have the form t00-r00-1.nc)

It will loop over all files and produce two scatter plots, qflux( tprim ) and nprim( tprim )
The points are color grouped by radial fluxtube index

Updated 26 July 2022
'''

file_list = sys.argv[1:]

#data_list = [ Dataset(f,mode='r') for f in file_list]

def avg_flux(q):
    med = np.median( [ np.median( q[::-1][:k] ) for k in np.arange( 1,len(q) )] )
    return med

data = []

for fname in file_list:

    f = Dataset(fname,mode='r')

    tprim = f.groups['Inputs']['Species']['T0_prime'][:]
    nprim = f.groups['Inputs']['Species']['n0_prime'][:]
    uprim = f.groups['Inputs']['Species']['u0_prime'][:]

    time = f.variables['time'][:]
    qflux = f.groups['Fluxes'].variables['qflux'][:,0]
    # check for nans
    if ( np.isnan(qflux).any() ):
         print('  nans found in', fname)
         qflux = np.nan_to_num(qflux)

    entry = {}
    entry['file_name'] = fname
    entry['tprim']     = tprim
    entry['nprim']     = uprim
    entry['uprim']     = nprim
    entry['qflux']     = qflux
    entry['time']      = time

    entry['avg_q']     = avg_flux(qflux)

    data.append(entry)


tprim_list = np.array( [ d['tprim'][0] for d in data] )
nprim_list = np.array( [ d['nprim'][0] for d in data] )
qavg_list = np.array( [ d['avg_q'] for d in data] )

qavg_list = np.nan_to_num(qavg_list) # set nans to zero

### make a two panel plot
#   add ability to color code radii?
#    would need to read radius from file name, then sort the filesA

'''
assume file has the form 'gx-files/JET/t00-r0-2.nc'
   t00-r0-2.nc
   r0
   0
'''
radius_index = np.array( [ f.split('/')[-1].split('-')[1][1:] for f in file_list], int )

fig,ax = plt.subplots(1,2, figsize=(9,5))

for j in np.arange(len(tprim_list)):
    ax[0].plot(tprim_list[j],qavg_list[j], f'C{radius_index[j]}.')
    ax[1].plot(tprim_list[j],nprim_list[j],f'C{radius_index[j]}.')

ax[0].set_xlabel('tprim')
ax[0].set_ylabel('qavg')
ax[1].set_xlabel('tprim')
ax[1].set_ylabel('nprim')

plt.suptitle("colors group radial flux tubes")
plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()