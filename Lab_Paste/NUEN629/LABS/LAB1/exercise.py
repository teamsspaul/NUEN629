import numpy as np #loads
#open total cross-section
sigma_t = np.genfromtxt('u235_total.csv', delimiter=",") #creats a two D array (energy in MEV X Sec in barns)
#open scattering cross-section
sigma_s = np.genfromtxt('u235_elastic.csv', delimiter=",", skip_header=1) #skip text

#get the union of the energy grids
energies = np.union1d(sigma_t[:,0], sigma_s[:,0])

#create the fission spectrum
chi = lambda E: 0.4865*np.sinh(np.sqrt(2*E))*np.exp(-E)

#make interpolation functions
from scipy import interpolate
sig_t_interp = interpolate.interp1d(sigma_t[:,0], sigma_t[:,1],bounds_error=False, fill_value=sigma_t[-1,1])
sig_s_interp = interpolate.interp1d(sigma_s[:,0], sigma_s[:,1],bounds_error=False, fill_value=sigma_s[-1,1])

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
#font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc')
#def hide_spines(intx=False,inty=False):
#	"""Hides the top and rightmost axis spines from view for all active
#	figures and their respective axes."""
#
#	# Retrieve a list of all current figures.
#	figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
#	if (plt.gca().get_legend()):
#		plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font)
#	for figure in figures:
#		# Get all Axis instances related to the figure.
#		for ax in figure.canvas.figure.get_axes():
#			# Disable spines.
#			ax.spines['right'].set_color('none')
#			ax.spines['top'].set_color('none')
#			# Disable ticks.
#			ax.xaxis.set_ticks_position('bottom')
#			ax.yaxis.set_ticks_position('left')
#			# ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10))))
#			for label in ax.get_xticklabels() :
#				label.set_fontproperties(font)
#			for label in ax.get_yticklabels() :
#				label.set_fontproperties(font)
#			#ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
#			ax.set_xlabel(ax.get_xlabel(), fontproperties = font)
#			ax.set_ylabel(ax.get_ylabel(), fontproperties = font)
#			ax.set_title(ax.get_title(), fontproperties = font)
#			if (inty):
#				ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
#			if (intx):
#				ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
#def show(nm=0,a=0,b=0):
#	hide_spines(a,b)
#	#ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
#	#plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
#	#ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
#	if (nm != 0):
#		plt.savefig(nm);
#	plt.show()

fig = plt.figure(figsize=(8,6), dpi=1600)
plt.loglog(energies, sig_t_interp(energies), label="$\sigma_\mathrm{t}$")
plt.loglog(energies, sig_s_interp(energies), label="$\sigma_\mathrm{s}$")
plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel("$\sigma$ (barns)")
plt.xlabel("E (MeV)")
plt.savefig("U-235_xsect.pdf")
#show("U-235_xsect.pdf")

fig = plt.figure(figsize=(8,6), dpi=1600)
plt.semilogx(energies,chi(energies))
plt.xlabel("E (MeV)")
plt.ylabel("Probability (MeV$^{-1}$)")  #Gives the nice -1
plt.savefig("U-235_chi.pdf")

#Now do 1 iteration
fig = plt.figure(figsize=(8,6), dpi=1600)
phi = interpolate.interp1d(energies,chi(energies)/sig_t_interp(energies),fill_value=1e-10,bounds_error=False)
plt.loglog(energies,phi(energies))
plt.xlabel("E (MeV)")
plt.ylabel("$\phi(E)$ (MeV$^{-1}$)")
plt.savefig("U-235_Spectrum_Iteration_1.pdf")


#converge the spectrum
A = 235
phi_iteration = lambda E: phi(E)
converged = 0
tolerance = 1.0e-6
iteration = 0
change_factor = (A+1)**2/(A**2+1)
while not(converged):
	phi_prev = interpolate.interp1d(energies,phi_iteration(energies),fill_value=0,bounds_error=False)
	phi_iteration= lambda E: (phi_prev(E*change_factor)*sig_s_interp(E*change_factor) + chi(E))/sig_t_interp(E)
	converged = (np.linalg.norm(phi_prev(energies) - phi_iteration(energies))/
				np.linalg.norm(phi_iteration(energies)) < tolerance)
	iteration += 1
print("Number of iterations",iteration)

fig = plt.figure(figsize=(8,6), dpi=1600)
plt.loglog(energies,phi_iteration(energies)/np.sum(phi_iteration(energies)), label="Scattering Adjusted")
plt.loglog(energies,phi(energies)/np.sum(phi_iteration(energies)), label="Uncollided")
plt.xlabel("E (MeV)")
plt.ylabel("$\phi(E)$ (MeV$^{-1}$)")
plt.legend(loc=4)
show("SpectrumComparison.pdf")



