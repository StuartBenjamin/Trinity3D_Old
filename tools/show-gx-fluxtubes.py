from gx_fluxtube import VMEC_Geometry

'''
Copied from simsopt-gx tool library

27 July 2022
'''

import matplotlib.pyplot as plt
import sys

f_list = sys.argv[1:]
fin = f_list[0]

fig, axs= plt.subplots( 3,4, figsize=(15,7) )

gx_geo = [ VMEC_Geometry(f, axs=axs) for f in f_list]
[g.plot() for g in gx_geo]

plt.tight_layout()
plt.show()

