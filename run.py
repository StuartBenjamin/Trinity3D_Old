import numpy as np
import matplotlib.pyplot as plt

import trinity_lib as trl
import diagnostics as dgn
import models      as mf

import os, sys, time

print("\nWelcome to Trinity3D")

# check that path is set
if os.environ.get("TRINITY_PATH") == None:
    print("\n  Environment Variable $TRINITY_PATH does not appear to be set.")
    print("  Try running (source setup.sh)\n")
    sys.exit()

try:
    fin = sys.argv[1]
except:
    fin = 'trinity.in'
print("\n  Loading input file:", fin, "\n")
sys.stdout.flush()


### Run Trinity!
start_time = time.time()

engine = trl.Trinity_Engine(fin)

writer = dgn.ProfileSaver()
#writer.store_system(engine) # 10/14


### Set up time controls
N_prints = engine.N_prints
N_steps  = engine.N_steps
#N_step_print = engine.N_steps // engine.N_prints   # how often to print 
if N_prints > N_steps:
    N_step_print = N_steps
    # guard against a bug, when more prints are demanded than steps, int-division gives 0
else:
    N_step_print = N_steps // N_prints   # how often to print 

# Put this into "Trinity Runner" class
#    "better to have functions than scripts"
while (engine.t_idx < engine.N_steps):
#while (engine.gx_idx < engine.N_steps):

    '''
    shift from counting time, to counting gx_calls?

    decison: not doing this later, it will change the nature of tests
    and different run modes (electron scale) might use more GX calls than others

    For now, I will just have the Newton method NOT increment this while loop (turtle 9/27)
    '''

    engine.get_flux()
    engine.calc_flux_coefficients()

    # I think collisions and turb. heat exchange should be added here
    engine.calc_collisions()

    engine.calc_psi_n()
    engine.calc_psi_pi() 
    engine.calc_psi_pe() 

    engine.calc_sources()

    if not engine.newton_mode:
       engine.calc_y_next()
    else:
       engine.calc_y_iter()

    engine.update()

###  turtle: why not save every step? Its not expensive to write. Is it expensive to read?
#    if not ( engine.t_idx % N_step_print):
#        
#        print(f"  Saving: t = {engine.t_idx}")
#        writer.save(engine)
    writer.save(engine)
    writer.store_system(engine) # 10/9

    writer.temp_record(engine) # 10/15
    writer.export(engine.f_save)

    engine.reset_fluxtubes()

    sys.stdout.flush()

end_time = time.time()
delta_t = end_time - start_time
def print_time(dt):
    h = int(dt // 3600)
    m = int( (dt-3600*h) // 60 )
    s = dt - 3600*h - 60*m
    print(f"  Total time: {h:d}h {m:d}m {s:.1f}s")

print('\nTRINITY Complete. Exiting normally')
print(f"  Total gx calls: {engine.gx_idx}")
print_time(delta_t)

root = os.environ.get("TRINITY_PATH") 
cmd = f"python {root}/tools/profile-plot.py {engine.f_save}.npy"
print('\nCalling plot function:')
print('  ',cmd)
os.system(cmd)


