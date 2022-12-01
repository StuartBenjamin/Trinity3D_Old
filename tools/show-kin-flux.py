import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset

from GX_io import GX_Output

'''
    This script takes a list of GX outputs,
    and plots Q(time) overlaying curves if multiple inputs are given.

    usage: python qflux.py [list of .nc files]')

    Updated 1 December 2022
    tqian@pppl.gov
'''

data = []
fins = []
for fname in sys.argv[1:]:

    if (fname.find('nonZonal') > 0): continue

    try:
        gxdata = GX_Output(fname)
        data.append( gxdata )
        fins.append(fname)
        print(' read', fname)
    
    except:
        print(' usage: python qflux.py [list of .nc files]')



fig, axs = plt.subplots(2,2, figsize=(12,8))
#plt.figure(figsize=(12,8))
cols = ['C0','C1','C2','C3','C4','C5','C6','C7']

j = 0
flux = []
for f in data:

    t = f.time
    Qflux_i = f.qflux_i_arr
    Qflux_e = f.qflux_e_arr
    Pflux   = f.pflux_arr

    axs[0,0].plot(t,Qflux_i,'.-')
    axs[0,1].plot(t,Qflux_e,'.-')
    axs[1,0].plot(t,Pflux  ,'.-')
    axs[1,1].plot(0,0  ,'.',label=fins[j])

    qflux_i = f.qflux_i
    flux.append(qflux_i)
    j+=1

print(flux)

#plt.yscale('log')
axs[0,0].set_title('Qi')
axs[0,1].set_title('Qe')
axs[1,0].set_title('Gamma')

axs[1,0].set_xlabel('time')
axs[1,1].set_xlabel('time')

axs[0,0].grid()
axs[0,1].grid()
axs[1,0].grid()

axs[1,1].legend()

N_files = len(data)
if N_files < 28:
    plt.legend()
else:
    plt.title(f"Showing {N_files} GX runs")
plt.show()
