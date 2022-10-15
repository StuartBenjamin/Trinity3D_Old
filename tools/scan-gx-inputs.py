from GX_io import GX_Runner
import sys

'''
This script reads a list of GX inputs (*.in)

It prints two sets of columns: grid resolutions and plasma parameters
15 October 2022
'''

f_list = sys.argv[1:]

gx_ins = [GX_Runner(f) for f in f_list]

gx_ins[0].list_resolution(header=True)
[ g.list_resolution() for g in gx_ins ]

print("")

gx_ins[0].list_inputs(header=True)
[ g.list_inputs() for g in gx_ins ]


import pdb
pdb.set_trace()
