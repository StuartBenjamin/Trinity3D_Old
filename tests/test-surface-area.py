from Geometry import VmecReader
import numpy as np
import sys

import matplotlib.pyplot as plt


try:
    # load custom input
    fin = sys.argv[1]
    vmec = VmecReader(fin)
    print("read input:", fin)

except:
    # use default and overwrite with simple torus
    fin = "gx-geometry/wout_JET-256.nc"
    vmec = VmecReader(fin)
    vmec.overwrite_simple_torus()
    fin = "Default Circular Torus (R,a) = 6,2"
    print("loaded default")



R = vmec.Rmajor
a = vmec.aminor
sax = np.linspace(0,1,vmec.ns)
rax = np.sqrt(sax)

# make a comparison
N_skip = 10
rho_ax = rax[::N_skip] 

fig, axs = plt.subplots(1,2, figsize=(11,6) )
axs[0].plot(rho_ax, 4*np.pi**2*R*a*rho_ax,'o-',label=r"$A = (2\pi R)(2\pi r)$")
axs[1].plot(rho_ax, 1/a * np.ones(len(rho_ax)),'o-',label=r"$|\nabla \rho| = 1/a$")

for nz in [10,20,40]:
    for nt in [4,8,16]:
        vmec.calc_gradrho_area( rho_ax, N_theta=nt, N_zeta=nz )
        axs[0].plot(rho_ax,vmec.surface_areas,'.',label=r'$(N_\varphi, N_\theta)$ = {}, {}'.format(nz,nt))
        axs[1].plot(rho_ax,vmec.avg_abs_grad_rho,'.')

axs[0].set_title("surface area")
axs[1].set_title("< | grad rho | >")
axs[0].set_xlabel("r/a")
axs[1].set_xlabel("r/a")
axs[1].set_ylim(0,2/a)
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()

plt.suptitle(fin)
plt.show()
