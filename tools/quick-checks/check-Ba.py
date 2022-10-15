from Geometry import VmecReader
import numpy as np
import sys

import matplotlib.pyplot as plt

'''
Checks the magnetic field data in a VMEC wout file.

15 October 2022
'''

fin = sys.argv[1]
vmec = VmecReader(fin)
print("read input:", fin)

R = vmec.Rmajor
a = vmec.aminor
volavgB = vmec.volavgB
b0 = vmec.data.variables['b0'][:]
rbtor = vmec.data.variables['rbtor'][:]
rbtor0 = vmec.data.variables['rbtor0'][:]
phiedge = vmec.data.variables['phi'][:][-1]

print("R         = ", R)
print("rbtor     = ", rbtor)
print("rbtor0    = ", rbtor0)
print("phiedge   = ", phiedge)
print("")
print("b0        = ", b0)
print("volavgB   = ", volavgB)
print("rbtor / R = ", rbtor/R)
print("rbtor0/ R = ", rbtor0/R)
print("phi/pi/a2 = ", phiedge/np.pi/a**2)

Bref = phiedge/np.pi/a**2
print("\n normalizing by Bref = phi/(pi a2)")
print("b0        = ", b0/Bref)
print("volavgB   = ", volavgB/Bref)
print("rbtor / R = ", rbtor/R/Bref)
print("rbtor0/ R = ", rbtor0/R/Bref)
print("phi/pi/a2 = ", phiedge/np.pi/a**2/Bref)

import pdb
pdb.set_trace()


