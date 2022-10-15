
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

files = sys.argv[1:]

data = [np.load(f, allow_pickle=True).tolist() for f in files]

fig,axs = plt.subplots(1,3)

for j in range(len(files)):
    drho = data[j]['profiles']['drho']
    try:
        grho = data[j]['profiles']['grho'].profile
    except:
        grho = data[j]['profiles']['grho']
    area = data[j]['profiles']['area']

    gfac = grho/area/drho
    axs[0].plot(area,'.-')
    axs[1].plot(grho,'.-')
    axs[2].plot(gfac,'.-')

plt.show()
