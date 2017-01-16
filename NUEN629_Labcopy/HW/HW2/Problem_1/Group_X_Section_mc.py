import numpy as np
import sys
import matplotlib.pyplot as plt
#open total cross-sections
sigma_t_235 = np.genfromtxt('u235_total.csv', delimiter=",")
sigma_t_238 = np.genfromtxt('u238_total.csv', delimiter=",")
sigma_t_12 = np.genfromtxt('c12_total.csv', delimiter=",")
#open scattering cross-section
sigma_s_235 = np.genfromtxt('u235_elastic.csv', delimiter=",", skip_header=1)
sigma_s_238 = np.genfromtxt('u238_elastic.csv', delimiter=",")
sigma_s_12 = np.genfromtxt('c12_elastic.csv', delimiter=",")

#create the fission spectrum 
chi = lambda E:  0.4865*np.sinh(np.sqrt(2*E))*np.exp(-E)

#make interpolation functions
from scipy import interpolate
sig_t_235_interp = interpolate.interp1d(sigma_t_235[:,0], sigma_t_235[:,1],bounds_error=False, fill_value=sigma_t_235[-1,1])
sig_s_235_interp = interpolate.interp1d(sigma_s_235[:,0], sigma_s_235[:,1],bounds_error=False, fill_value=sigma_s_235[-1,1])
sig_t_238_interp = interpolate.interp1d(sigma_t_238[:,0], sigma_t_238[:,1],bounds_error=False, fill_value=sigma_t_238[-1,1])
sig_s_238_interp = interpolate.interp1d(sigma_s_238[:,0], sigma_s_238[:,1],bounds_error=False, fill_value=sigma_s_238[-1,1])
sig_t_12_interp = interpolate.interp1d(sigma_t_12[:,0], sigma_t_12[:,1],bounds_error=False, fill_value=sigma_t_12[-1,1])
sig_s_12_interp = interpolate.interp1d(sigma_s_12[:,0], sigma_s_12[:,1],bounds_error=False, fill_value=sigma_s_12[-1,1])

#get the union of the energy grids
energies = np.union1d(sigma_t_235[:,0], sigma_s_235[:,0])

#I am not sure what these Commands do; print(len(energies))
energies_new = np.union1d(energies,sigma_t_238[:,0])
energies = energies_new
energies_new = np.union1d(energies,sigma_s_238[:,0])
energies = energies_new
energies_new = np.union1d(energies,sigma_t_12[:,0])
energies = energies_new
energies_new = np.union1d(energies,sigma_s_12[:,0])
energies = energies_new
#But Dr. MC did them...
#energies = np.union1d(energies,sigma_t_238[:,0])
#If I want to append energies I Should do this print(len(energies))

#Now do 1 iteration
phi = interpolate.interp1d(energies,chi(energies)/(0.00657*sig_t_238_interp(energies) + \
4.768e-5 * sig_t_235_interp(energies) + 0.993377 * sig_t_12_interp(energies)),fill_value=1.0e-11,bounds_error=False)
#converge the spectrum
phi_iteration = lambda E: phi(E)
converged = 0
tolerance = 1.0e-6
iteration = 0
max_iterations = 600
ff = lambda A: (A+1)**2/(A**2+1)
#first solve without 238-U scattering
while not(converged):
    phi_prev = interpolate.interp1d(energies,phi_iteration(energies),fill_value=0,bounds_error=False)
    phi_iteration= lambda E: (0*phi_prev(E*ff(238))*sig_s_238_interp(E*ff(238)) \
                                + 4.768e-5 * phi_prev(E*ff(235))*sig_s_235_interp(E*ff(235)) \
                               + 0.993377 * phi_prev(E*ff(12))*sig_s_12_interp(E*ff(12))\
                                + chi(E))/(0.00657*sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    converged = (np.linalg.norm(phi_prev(energies) - phi_iteration(energies))/\
                 np.linalg.norm(phi_iteration(energies)) < tolerance) or (iteration >= max_iterations)
    iteration += 1
print("Number of iterations",iteration)
#using that as initial guess, now solve the entire thing
converged=0
while not(converged):
    phi_prev = interpolate.interp1d(energies,phi_iteration(energies),fill_value=0,bounds_error=False)
    phi_iteration= lambda E: (0.00657*phi_prev(E*ff(238))*sig_s_238_interp(E*ff(238)) \
                                + 4.768e-5 * phi_prev(E*ff(235))*sig_s_235_interp(E*ff(235)) \
                               + 0.993377 * phi_prev(E*ff(12))*sig_s_12_interp(E*ff(12))\
                                + chi(E))/(0.00657*sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    converged = (np.linalg.norm(phi_prev(energies) - phi_iteration(energies))/\
                 np.linalg.norm(phi_iteration(energies)) < tolerance) or (iteration >= max_iterations)
    iteration += 1
print("Number of iterations",iteration)

#Perform Integration
from scipy import integrate
from scipy.integrate import trapz
#Make Functions of things I want to integrate.
X_t_235_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_t_235_interp(energies),fill_value=0,bounds_error=False)
X_s_235_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_s_235_interp(energies),fill_value=0,bounds_error=False)
X_t_238_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_t_238_interp(energies),fill_value=0,bounds_error=False)
X_s_238_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_s_238_interp(energies),fill_value=0,bounds_error=False)
X_t_12_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_t_12_interp(energies),fill_value=0,bounds_error=False)
X_s_12_phi=interpolate.interp1d(energies,phi_iteration(energies)*sig_s_12_interp(energies),fill_value=0,bounds_error=False)

#Perform the integration For Group 1
EE=1e-6
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index=xx
phi_int_g1=integrate.trapz(phi_iteration(energies[0:index]),energies[0:index])
X_t_235_g1=integrate.trapz(X_t_235_phi(energies[0:index]),energies[0:index])
X_t_238_g1=integrate.trapz(X_t_238_phi(energies[0:index]),energies[0:index])
X_t_12_g1=integrate.trapz(X_t_12_phi(energies[0:index]),energies[0:index])
X_s_235_g1=integrate.trapz(X_s_235_phi(energies[0:index]),energies[0:index])
X_s_238_g1=integrate.trapz(X_s_238_phi(energies[0:index]),energies[0:index])
X_s_12_g1=integrate.trapz(X_s_12_phi(energies[0:index]),energies[0:index])

#Perform the integration for group 2
EE=0.1
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index2=xx
phi_int_g2=integrate.trapz(phi_iteration(energies[index:index2]),energies[index:index2])
X_t_235_g2=integrate.trapz(X_t_235_phi(energies[index:index2]),energies[index:index2])
X_t_238_g2=integrate.trapz(X_t_238_phi(energies[index:index2]),energies[index:index2])
X_t_12_g2=integrate.trapz(X_t_12_phi(energies[index:index2]),energies[index:index2])
X_s_235_g2=integrate.trapz(X_s_235_phi(energies[index:index2]),energies[index:index2])
X_s_238_g2=integrate.trapz(X_s_238_phi(energies[index:index2]),energies[index:index2])
X_s_12_g2=integrate.trapz(X_s_12_phi(energies[index:index2]),energies[index:index2])

#Perform the integration for group 3
EE=20
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index3=xx
phi_int_g3=integrate.trapz(phi_iteration(energies[index2:index3]),energies[index2:index3])
X_t_235_g3=integrate.trapz(X_t_235_phi(energies[index2:index3]),energies[index2:index3])
X_t_238_g3=integrate.trapz(X_t_238_phi(energies[index2:index3]),energies[index2:index3])
X_t_12_g3=integrate.trapz(X_t_12_phi(energies[index2:index3]),energies[index2:index3])
X_s_235_g3=integrate.trapz(X_s_235_phi(energies[index2:index3]),energies[index2:index3])
X_s_238_g3=integrate.trapz(X_s_238_phi(energies[index2:index3]),energies[index2:index3])
X_s_12_g3=integrate.trapz(X_s_12_phi(energies[index2:index3]),energies[index2:index3])


print (X_s_235_g1/phi_int_g1)
print (X_s_235_g2/phi_int_g2)
print (X_s_235_g3/phi_int_g3)
print (X_s_238_g1/phi_int_g1)
print (X_s_238_g2/phi_int_g2)
print (X_s_238_g3/phi_int_g3)
print (X_s_12_g1/phi_int_g1)
print (X_s_12_g2/phi_int_g2)
print (X_s_12_g3/phi_int_g3)

#Plot if I want to
if (sys.argv[1]=="t"):
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.loglog(energies, sig_t_235_interp(energies), label="$\sigma_\mathrm{t}$")
	plt.loglog(energies, sig_s_235_interp(energies), label="$\sigma_\mathrm{s}$")
	plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.ylabel("$\sigma$ (barns)")
	plt.xlabel("E (MeV)")
	plt.savefig("U-235_xsect.pdf")
	#plot cross section 238
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.loglog(energies, sig_t_238_interp(energies), label="$\sigma_\mathrm{t}$")
	plt.loglog(energies, sig_s_238_interp(energies), label="$\sigma_\mathrm{s}$")
	plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.ylabel("$\sigma$ (barns)")
	plt.xlabel("E (MeV)")
	plt.savefig("U-238_xsect.pdf")
	#plot cross section 12
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.loglog(energies, sig_t_12_interp(energies), label="$\sigma_\mathrm{t}$")
	plt.loglog(energies, sig_s_12_interp(energies), label="$\sigma_\mathrm{s}$")
	plt.legend(loc=3) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.ylabel("$\sigma$ (barns)")
	plt.xlabel("E (MeV)")
	plt.savefig("c-12_xsect.pdf")
	#Plot the fission spectrum
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.semilogx(energies,chi(energies))
	plt.xlabel("E (MeV)")
	plt.ylabel("Probability (MeV$^{-1}$)")
	plt.savefig("U-235chi.pdf")
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.loglog(energies,phi(energies)/np.sum(phi(energies)), label="Uncollided")
	plt.loglog(energies,phi_iteration(energies)/np.sum(phi_iteration(energies)), label="Scattering Adjusted")
	plt.xlabel("E (MeV)")
	plt.ylabel("$\phi(E)$ (MeV$^{-1}$)")
	plt.legend(loc=4)
	plt.savefig("SpectrumComparison_mc.pdf")