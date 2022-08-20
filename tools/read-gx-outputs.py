#from turbulence.GX_io import GX_Runner
#from simsopt.turbulence import GX_io
from simsopt.turbulence.GX_io import GX_Output

import numpy as np
import matplotlib.pyplot as plt
import sys

f_list = []

for f in sys.argv[1:]:

    if f.find('restart') > 0:
        continue

    f_list.append(f)



gx_outs = [GX_Output(f) for f in f_list]
q_med = np.array([ g.median_estimator() for g in gx_outs ])
q_avg, dq_avg = np.transpose( [ g.exponential_window_estimator() for g in gx_outs ] )

X = np.linspace(-1,1.5,20)

#X = np.array( [-0.10426373683811282, -0.0449543072758396, -0.0004785094882797103, 0.03287363372805269, 0.05788421279480711, 0.07663950120901777, 0.09070398338519448, 0.1012508571263689, 0.10915989667143614, 0.11509083962766346, 0.11953841940641945, 0.1228736337280527, 0.1328736337280527, 0.1428736337280527, 0.14620884804968592, 0.1506564278284419, 0.15658737078466925, 0.1644964103297365, 0.1750432840709109, 0.1891077662470876, 0.20786305466129829, 0.2662257769443851, 0.310701574731945, 0.2328736337280527, 0.3700110042942182] )

import pdb
pdb.set_trace()

plt.figure(); 
plt.errorbar(X,q_avg,yerr=dq_avg,fmt='.-',label='exponential moving average');
plt.plot(X,q_med,'x',label='median of medians'); 

plt.ylabel('GX Nonlinear Heat Flux')
plt.xlabel('Boundary ZBS (m=0, n=1)')
plt.legend(fontsize=10)
plt.show()

