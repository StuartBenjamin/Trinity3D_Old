# Tests profiles with different parameterisations: quadratic, linear, and exponential, with a given core and edge specified.
# Compares exact results with calculations in Trinity profiles.py, to ensure profiles.py is working correctly.
# Saves images in tests directory.
# To run: python test-profiles.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("..")
import profiles as pf

##### PROFILE INPUT PARAMETERS
n_core  = 10
n_edge  = 1
Ti_core = 12
Ti_edge = 0.5
Te_core = 14
Te_edge = 1
N_radial = 20
rho_edge = 0.9
#####

plot_out_dir = 'tmp/'

if not os.path.exists(plot_out_dir):
    os.makedirs(plot_out_dir)


###### 1 - rho^2 Profiles #######
def quadratic_profile_and_deriv(fun_core,fun_edge,rho_axis,rho_edge):
	#### Function has form:
	# fun = (fun_core - fun_edge)*(1 - (rho_axis/rho_edge)**2) + fun_edge	

	#### We can take analytic derivatives of these functions.
	# fun = a (1-(x/b)^2) + c
	# dfun/dx = -2ax/(b^2)
	# dln(fun)/ dx = (-2ax/(b^2)) / (a (1-(x/b)^2) + c)
	profile_fun = (fun_core  - fun_edge)*(1 - (rho_axis/rho_edge)**2) + fun_edge
	profile_fun_deriv = (-2*(fun_core  - fun_edge)*rho_axis/(rho_edge**2))/profile_fun

	return profile_fun, profile_fun_deriv

# Standard procedure in profiles.py
rho_axis = np.linspace(0,rho_edge,N_radial)
pf.rho_axis = rho_axis

n, n_deriv = quadratic_profile_and_deriv(n_core,n_edge,rho_axis,rho_edge)
Ti, Ti_deriv = quadratic_profile_and_deriv(Ti_core,Ti_edge,rho_axis,rho_edge)
Te, Te_deriv = quadratic_profile_and_deriv(Te_core,Te_edge,rho_axis,rho_edge)

n_profiles = pf.Profile(n, grad=True, half=True, full=True)
Ti_profiles = pf.Profile(Ti, grad=True, half=True, full=True)
Te_profiles = pf.Profile(Te, grad=True, half=True, full=True)

fluxtube_axis = n_profiles.radial_midpoints()
n_log_grad_midpoint = n_profiles.midpoint_log_gradient()
Ti_log_grad_midpoint = Ti_profiles.midpoint_log_gradient()
Te_log_grad_midpoint = Te_profiles.midpoint_log_gradient()

fig = plt.figure(figsize=(15,8),dpi=300,facecolor='w')
ax1 = plt.subplot(231) #
ax2 = plt.subplot(232) #
ax3 = plt.subplot(233) #
ax4 = plt.subplot(234) #
ax5 = plt.subplot(235) #
ax6 = plt.subplot(236) #
ax1.plot(rho_axis,n,'kx-',label = '$n$')
ax2.plot(rho_axis,Ti,'ro--',label = '$T_i$')
ax3.plot(rho_axis,Te,'bd--',label = '$T_e$')
ax4.plot(fluxtube_axis,-n_log_grad_midpoint,'ko-',label = '$a/L_{{n}}$, TRINITY')
ax4.plot(rho_axis,-n_deriv,'kx--',label = '$a/L_{{n}}$, Exact')
ax5.plot(fluxtube_axis,-Ti_log_grad_midpoint,'ro-',label = '$a/L_{{n}}$, TRINITY')
ax5.plot(rho_axis,-Ti_deriv,'rx--',label = '$a/L_{{Ti}}$, Exact')
ax6.plot(fluxtube_axis,-Te_log_grad_midpoint,'bo--',label = '$a/L_{{Te}}$, TRINITY')
ax6.plot(rho_axis,-Te_deriv,'bx--',label = '$a/L_{{Te}}$, Exact')
ax1.set_xlabel('$\\rho$')
ax2.set_xlabel('$\\rho$')
ax3.set_xlabel('$\\rho$')
ax4.set_xlabel('$\\rho$')
ax5.set_xlabel('$\\rho$')
ax6.set_xlabel('$\\rho$')
ax1.set_ylabel('Profile [arb. units]')
ax2.set_ylabel('Profile [arb. units]')
ax3.set_ylabel('Profile [arb. units]')
ax4.set_ylabel('Profile [arb. units]')
ax5.set_ylabel('Profile [arb. units]')
ax6.set_ylabel('Profile [arb. units]')
ax1.set_xlim(0,1)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)
ax4.set_xlim(0,1)
ax5.set_xlim(0,1)
ax6.set_xlim(0,1)
leg1 = ax1.legend(loc = 'best')
leg1.get_frame().set_edgecolor('k')
leg1.get_frame().set_linewidth(0.4)
leg2 = ax2.legend(loc = 'best')
leg2.get_frame().set_edgecolor('k')
leg2.get_frame().set_linewidth(0.4)
leg3 = ax3.legend(loc = 'best')
leg3.get_frame().set_edgecolor('k')
leg3.get_frame().set_linewidth(0.4)
leg4 = ax4.legend(loc = 'best')
leg4.get_frame().set_edgecolor('k')
leg4.get_frame().set_linewidth(0.4)
leg5 = ax5.legend(loc = 'best')
leg5.get_frame().set_edgecolor('k')
leg5.get_frame().set_linewidth(0.4)
leg6 = ax6.legend(loc = 'best')
leg6.get_frame().set_edgecolor('k')
leg6.get_frame().set_linewidth(0.4)
plt.suptitle('$1-\\rho^2$ Equilibrium Profiles')
plt.savefig(plot_out_dir + 'profile_plot_gradients_quadratic.png',bbox_inches='tight', pad_inches = 0.1)
plt.clf()
plt.close(fig)


##### Linear Profile Case
###### 1-rho Profiles #######
def linear_profile_and_deriv(fun_core,fun_edge,rho_axis,rho_edge):
	#### Function has form:
	# fun = (fun_core - fun_edge)*(1 - (rho_axis/rho_edge)) + fun_edge	

	#### We can take analytic derivatives of these functions.
	# fun = a (1-(x/b)) + c
	# dfun/dx = -ax/b
	# dln(fun)/ dx = (-a/b) / (a (1-(x/b)) + c)
	profile_fun = (fun_core  - fun_edge)*(1 - (rho_axis/rho_edge)) + fun_edge
	profile_fun_deriv = (-(fun_core  - fun_edge)/rho_edge)/profile_fun

	return profile_fun, profile_fun_deriv

# Standard procedure in profiles.py
rho_axis = np.linspace(0,rho_edge,N_radial)

n, n_deriv = linear_profile_and_deriv(n_core,n_edge,rho_axis,rho_edge)
Ti, Ti_deriv = linear_profile_and_deriv(Ti_core,Ti_edge,rho_axis,rho_edge)
Te, Te_deriv = linear_profile_and_deriv(Te_core,Te_edge,rho_axis,rho_edge)

n_profiles = pf.Profile(n, grad=True, half=True, full=True)
Ti_profiles = pf.Profile(Ti, grad=True, half=True, full=True)
Te_profiles = pf.Profile(Te, grad=True, half=True, full=True)

fluxtube_axis = n_profiles.radial_midpoints()
n_log_grad_midpoint = n_profiles.midpoint_log_gradient()
Ti_log_grad_midpoint = Ti_profiles.midpoint_log_gradient()
Te_log_grad_midpoint = Te_profiles.midpoint_log_gradient()

#### Profiles
fig = plt.figure(figsize=(15,8),dpi=300,facecolor='w')
ax1 = plt.subplot(231) #
ax2 = plt.subplot(232) #
ax3 = plt.subplot(233) #
ax4 = plt.subplot(234) #
ax5 = plt.subplot(235) #
ax6 = plt.subplot(236) #
ax1.plot(rho_axis,n,'kx-',label = '$n$')
ax2.plot(rho_axis,Ti,'ro--',label = '$T_i$')
ax3.plot(rho_axis,Te,'bd--',label = '$T_e$')
ax4.plot(fluxtube_axis,-n_log_grad_midpoint,'ko-',label = '$a/L_{{n}}$, TRINITY')
ax4.plot(rho_axis,-n_deriv,'kx--',label = '$a/L_{{n}}$, Exact')
ax5.plot(fluxtube_axis,-Ti_log_grad_midpoint,'ro-',label = '$a/L_{{n}}$, TRINITY')
ax5.plot(rho_axis,-Ti_deriv,'rx--',label = '$a/L_{{Ti}}$, Exact')
ax6.plot(fluxtube_axis,-Te_log_grad_midpoint,'bo--',label = '$a/L_{{Te}}$, TRINITY')
ax6.plot(rho_axis,-Te_deriv,'bx--',label = '$a/L_{{Te}}$, Exact')
ax1.set_xlabel('$\\rho$')
ax2.set_xlabel('$\\rho$')
ax3.set_xlabel('$\\rho$')
ax4.set_xlabel('$\\rho$')
ax5.set_xlabel('$\\rho$')
ax6.set_xlabel('$\\rho$')
ax1.set_ylabel('Profile [arb. units]')
ax2.set_ylabel('Profile [arb. units]')
ax3.set_ylabel('Profile [arb. units]')
ax4.set_ylabel('Profile [arb. units]')
ax5.set_ylabel('Profile [arb. units]')
ax6.set_ylabel('Profile [arb. units]')
ax1.set_xlim(0,1)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)
ax4.set_xlim(0,1)
ax5.set_xlim(0,1)
ax6.set_xlim(0,1)
leg1 = ax1.legend(loc = 'best')
leg1.get_frame().set_edgecolor('k')
leg1.get_frame().set_linewidth(0.4)
leg2 = ax2.legend(loc = 'best')
leg2.get_frame().set_edgecolor('k')
leg2.get_frame().set_linewidth(0.4)
leg3 = ax3.legend(loc = 'best')
leg3.get_frame().set_edgecolor('k')
leg3.get_frame().set_linewidth(0.4)
leg4 = ax4.legend(loc = 'best')
leg4.get_frame().set_edgecolor('k')
leg4.get_frame().set_linewidth(0.4)
leg5 = ax5.legend(loc = 'best')
leg5.get_frame().set_edgecolor('k')
leg5.get_frame().set_linewidth(0.4)
leg6 = ax6.legend(loc = 'best')
leg6.get_frame().set_edgecolor('k')
leg6.get_frame().set_linewidth(0.4)
plt.suptitle('$1-\\rho$ Equilibrium Profiles')
plt.savefig(plot_out_dir + 'profile_plot_gradients_linear.png',bbox_inches='tight', pad_inches = 0.1)
plt.clf()
plt.close(fig)

##### Exponential Profile Case
###### exp(-rho) Profiles #######
def exp_profile_and_deriv(fun_core,rho_axis,rho_edge,exp_fac=2):
	#### Function has form:
	# fun = (fun_core)*exp(-(rho_axis/rho_edge))	

	#### We can take analytic derivatives of these functions.
	# fun = a (exp-(x/b)) + c
	# dfun/dx = -(a/b)(exp-(x/b))
	# dln(fun)/ dx = (-(a/b)(exp-(x/b))) / (a (exp-(x/b)) + c)
	profile_fun = fun_core*np.exp(-exp_fac*(rho_axis/rho_edge))
	profile_fun_deriv = -(fun_core*exp_fac/rho_edge)*np.exp(-exp_fac*(rho_axis/rho_edge))/profile_fun

	return profile_fun, profile_fun_deriv

# Standard procedure in profiles.py
rho_axis = np.linspace(0,rho_edge,N_radial)

n, n_deriv = exp_profile_and_deriv(n_core,rho_axis,rho_edge)
Ti, Ti_deriv = exp_profile_and_deriv(Ti_core,rho_axis,rho_edge)
Te, Te_deriv = exp_profile_and_deriv(Te_core,rho_axis,rho_edge)

n_profiles = pf.Profile(n, grad=True, half=True, full=True)
Ti_profiles = pf.Profile(Ti, grad=True, half=True, full=True)
Te_profiles = pf.Profile(Te, grad=True, half=True, full=True)

fluxtube_axis = n_profiles.radial_midpoints()
n_log_grad_midpoint = n_profiles.midpoint_log_gradient()
Ti_log_grad_midpoint = Ti_profiles.midpoint_log_gradient()
Te_log_grad_midpoint = Te_profiles.midpoint_log_gradient()

fig = plt.figure(figsize=(15,8),dpi=300,facecolor='w')
ax1 = plt.subplot(231) #
ax2 = plt.subplot(232) #
ax3 = plt.subplot(233) #
ax4 = plt.subplot(234) #
ax5 = plt.subplot(235) #
ax6 = plt.subplot(236) #
ax1.plot(rho_axis,n,'kx-',label = '$n$')
ax2.plot(rho_axis,Ti,'ro--',label = '$T_i$')
ax3.plot(rho_axis,Te,'bd--',label = '$T_e$')
ax4.plot(fluxtube_axis,-n_log_grad_midpoint,'ko-',label = '$a/L_{{n}}$, TRINITY')
ax4.plot(rho_axis,-n_deriv,'kx--',label = '$a/L_{{n}}$, Exact')
ax5.plot(fluxtube_axis,-Ti_log_grad_midpoint,'ro-',label = '$a/L_{{n}}$, TRINITY')
ax5.plot(rho_axis,-Ti_deriv,'rx--',label = '$a/L_{{Ti}}$, Exact')
ax6.plot(fluxtube_axis,-Te_log_grad_midpoint,'bo--',label = '$a/L_{{Te}}$, TRINITY')
ax6.plot(rho_axis,-Te_deriv,'bx--',label = '$a/L_{{Te}}$, Exact')
ax1.set_xlabel('$\\rho$')
ax2.set_xlabel('$\\rho$')
ax3.set_xlabel('$\\rho$')
ax4.set_xlabel('$\\rho$')
ax5.set_xlabel('$\\rho$')
ax6.set_xlabel('$\\rho$')
ax1.set_ylabel('Profile [arb. units]')
ax2.set_ylabel('Profile [arb. units]')
ax3.set_ylabel('Profile [arb. units]')
ax4.set_ylabel('Profile [arb. units]')
ax5.set_ylabel('Profile [arb. units]')
ax6.set_ylabel('Profile [arb. units]')
ax1.set_xlim(0,1)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)
ax4.set_xlim(0,1)
ax5.set_xlim(0,1)
ax6.set_xlim(0,1)
leg1 = ax1.legend(loc = 'best')
leg1.get_frame().set_edgecolor('k')
leg1.get_frame().set_linewidth(0.4)
leg2 = ax2.legend(loc = 'best')
leg2.get_frame().set_edgecolor('k')
leg2.get_frame().set_linewidth(0.4)
leg3 = ax3.legend(loc = 'best')
leg3.get_frame().set_edgecolor('k')
leg3.get_frame().set_linewidth(0.4)
leg4 = ax4.legend(loc = 'best')
leg4.get_frame().set_edgecolor('k')
leg4.get_frame().set_linewidth(0.4)
leg5 = ax5.legend(loc = 'best')
leg5.get_frame().set_edgecolor('k')
leg5.get_frame().set_linewidth(0.4)
leg6 = ax6.legend(loc = 'best')
leg6.get_frame().set_edgecolor('k')
leg6.get_frame().set_linewidth(0.4)
plt.suptitle('$exp(-\\rho)$ Equilibrium Profiles')
plt.savefig(plot_out_dir + 'profile_plot_gradients_exponential.png',bbox_inches='tight', pad_inches = 0.1)
plt.clf()
plt.close(fig)


