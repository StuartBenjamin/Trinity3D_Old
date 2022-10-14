from Geometry import VmecReader
import numpy as np

import matplotlib.pyplot as plt

fin = "gx-geometry/wout_JET-256.nc"

vmec = VmecReader(fin)


# overwrite with simple torus
vmec.xn = np.array([0,0])
vmec.xm = np.array([0,1])

R = vmec.Rmajor
a = vmec.aminor

R = 6
a = 2

sax = np.linspace(0,1,vmec.ns)
rax = a * np.sqrt(sax)

vmec.rmnc = np.array( [ [R,r] for r in rax ] )
vmec.zmns = np.array( [ [0,r] for r in rax ] )
vmec.N_modes = 2

# make a comparison
N_skip = 10
ax = rax[::N_skip]

fig, axs = plt.subplots(1,2, figsize=(11,6) )
axs[0].plot(ax, 4*np.pi**2*R * ax,'o-',label=r"$A = (2\pi R)(2\pi r)$")
axs[1].plot(ax, 1/a * np.ones(len(ax)),'o-',label=r"$|\nabla \rho| = 1/a$")

for nz in [10,20,40]:
    for nt in [4,8,16]:
        vmec.calc_gradrho_area( ax/a, N_theta=nt, N_zeta=nz )
        axs[0].plot(ax,vmec.surface_areas,'.',label=r'$(N_\varphi, N_\theta)$ = {}, {}'.format(nz,nt))
        axs[1].plot(ax,vmec.avg_abs_grad_rho,'.')

axs[0].set_title("surface area")
axs[1].set_title("< | grad rho | >")
axs[1].set_ylim(0,2/a)
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()
plt.show()
