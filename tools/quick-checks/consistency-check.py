'''
    This script reads a GX input file and checks for consistency conditions
    (1) Quasineutrality
    (2) Pressure Profile

    For (1) we need to identify (n,T,n',T',q) for each species.
    For (2) we additionally need to identify (beta') for the flux tube. 
    This can be read from the geometry file. If igeo=2 vmec is used for GX,
    then the pressure profile can be read from VMEC, and the flux surface index is read from geometry file.

    This program should be exportable outside of Trinity, for standalone use in GX.
    5 July 2022
'''

from GX_io import GX_Runner as gio
from Geometry import FluxTube
#from Trinity_io import Trinity_Input

import numpy as np
import sys

fin = sys.argv[1]
gx = gio(fin)


sp = gx.inputs['species']
def _unpack(s):
    '''
         unpacks the string
         [1  ,   2,]
         which contains species info
         returns an np float array.
    '''
    return np.array( s[1:-1].split(','), float)

qs = _unpack( sp['z'] )
ns = _unpack( sp['dens'] )
ts = _unpack( sp['temp'] )
dn = _unpack( sp['fprim'] )
dT = _unpack( sp['tprim'] )


# quasineutrality
print(" (1) quasineutrality condition")
print(f"      \sum qs ns  =   {np.sum(ns*qs)}                 (should be 0)")

# quasineutrality gradient
print(" (2) quasineutrality gradient condition")
print(f"      \sum qs ns n'  =   {np.sum(ns*qs*dn)}              (should be 0)")

# total pressure
print(" (3) total pressure condition")
print(f"      \sum (n' + t') ns ts  =   {np.sum( (dn+dT)*ns*ts)}       (should be beta')")

###  get beta'
geo = gx.inputs['Geometry']

igeo =  int(geo['igeo']) 
print(f" igeo = {igeo}   (must be 2 for vmec)")
gfile = geo['geofile'][1:-1] # remove quotations

# get path
ss = len(fin.split('/')[-1])
path = fin[:-ss]

#fl = FluxTube(path+gfile)
'''
 the problem here is the GFILE is the netcdf output containing fluxtube geometry.
 This doesn't have pressure profile information.

 We need to go to the original VMEC output. That is stored in flux tube INPUT, but the gx input only has flux tube OUTPUT.
'''

# magically get vmec output
#wout = sys.argv[2]
# import netCDF library to read it

import pdb
pdb.set_trace()
