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
energies = energies.copy()
#But Dr. MC did them...
#energies = np.union1d(energies,sigma_t_238[:,0])
#If I want to append energies I Should do this print(len(energies))
#Find point in energies that we go over 2.53e-8
EE=2.53e-8
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index=xx
#Now do 1 iteration
phi = interpolate.interp1d(energies,chi(energies)/(0.00657*sig_t_238_interp(energies) + \
4.768e-5 * sig_t_235_interp(energies) + 0.993377 * sig_t_12_interp(energies)),fill_value=1.0e-11,bounds_error=False)
#converge the spectrum
phi_iteration = lambda E: phi(E)
converged = 0
tolerance = 1.0e-6
iteration = 0
max_iterations = 100
ff = lambda A: (A+1)**2/(A**2+1)
fff = lambda A: (A-1)**2/(A**2+1)
#first solve without 238-U scattering
while not(converged):
    phi_prev = interpolate.interp1d(energies,phi_iteration(energies),fill_value=0,bounds_error=False)
    phi_i1= lambda E: (0*phi_prev(E*(fff(238)))*sig_s_238_interp(E*(fff(238))) \
                                + 4.768e-5 * phi_prev(E*(fff(235)))*sig_s_235_interp(E*(fff(235))) \
                               + 0.993377 * phi_prev(E*(fff(16)))*sig_s_12_interp(E*(fff(16)))\
                                + chi(E))/(sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    phi_i2= (0*phi_prev(EE*(fff(238)))*sig_s_238_interp(EE*(fff(238))) \
                                + 4.768e-5 * phi_prev(EE*(fff(235)))*sig_s_235_interp(EE*(fff(235))) \
                               + 0.993377 * phi_prev(EE*(fff(16)))*sig_s_12_interp(EE*(fff(16)))+\
            0*phi_prev(EE*(ff(238)))*sig_s_238_interp(EE*(ff(238))) \
                                + 4.768e-5 * phi_prev(EE*(ff(235)))*sig_s_235_interp(EE*(ff(235))) \
                               + 0.993377 * phi_prev(EE*(ff(16)))*sig_s_12_interp(EE*(ff(16)))\
                                + chi(EE))/(sig_t_238_interp(EE) + \
                                           4.768e-5 * sig_t_235_interp(EE) + \
                                           0.993377 * sig_t_12_interp(EE))
    phi_i3= lambda E:(0*phi_prev(E*ff(238))*sig_s_238_interp(E*ff(238)) \
                                + 4.768e-5 * phi_prev(E*ff(235))*sig_s_235_interp(E*ff(235)) \
                               + 0.993377 * phi_prev(E*ff(16))*sig_s_12_interp(E*ff(16))\
                                + chi(E))/(sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    phi_i1_np=np.array(phi_i1(energies[0:index]))
    phi_i2_np=np.array([phi_i2])
    phi_i3_np=np.array(phi_i3(energies[index+1:len(energies)]))
    arraydawg=np.concatenate([phi_i1_np,phi_i2_np,phi_i3_np])
    phi_iteration=interpolate.interp1d(energies,arraydawg)
    converged = (np.linalg.norm(phi_prev(energies) - phi_iteration(energies))/\
                 np.linalg.norm(phi_iteration(energies)) < tolerance) or (iteration >= max_iterations)
    iteration += 1
print("Number of iterations",iteration)
#using that as initial guess, now solve the entire thing
converged=0
while not(converged):
    phi_prev = interpolate.interp1d(energies,phi_iteration(energies),fill_value=0,bounds_error=False)
    phi_i1= lambda E: (0.00657*phi_prev(E*(fff(238)))*sig_s_238_interp(E*(fff(238))) \
                                + 4.768e-5 * phi_prev(E*(fff(235)))*sig_s_235_interp(E*(fff(235))) \
                               + 0.993377 * phi_prev(E*(fff(16)))*sig_s_12_interp(E*(fff(16)))\
                                + chi(E))/(sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    phi_i2= (0.00657*phi_prev(EE*(fff(238)))*sig_s_238_interp(EE*(fff(238))) \
                                + 4.768e-5 * phi_prev(EE*(fff(235)))*sig_s_235_interp(EE*(fff(235))) \
                               + 0.993377 * phi_prev(EE*(fff(16)))*sig_s_12_interp(EE*(fff(16)))+\
            0.00657*phi_prev(EE*(ff(238)))*sig_s_238_interp(EE*(ff(238))) \
                                + 4.768e-5 * phi_prev(EE*(ff(235)))*sig_s_235_interp(EE*(ff(235))) \
                               + 0.993377 * phi_prev(EE*(ff(16)))*sig_s_12_interp(EE*(ff(16)))\
                                + chi(EE))/(sig_t_238_interp(EE) + \
                                           4.768e-5 * sig_t_235_interp(EE) + \
                                           0.993377 * sig_t_12_interp(EE))
    phi_i3= lambda E:(0.00657*phi_prev(E*ff(238))*sig_s_238_interp(E*ff(238)) \
                                + 4.768e-5 * phi_prev(E*ff(235))*sig_s_235_interp(E*ff(235)) \
                               + 0.993377 * phi_prev(E*ff(16))*sig_s_12_interp(E*ff(16))\
                                + chi(E))/(sig_t_238_interp(E) + \
                                           4.768e-5 * sig_t_235_interp(E) + \
                                           0.993377 * sig_t_12_interp(E))
    phi_i1_np=np.array(phi_i1(energies[0:index]))
    phi_i2_np=np.array([phi_i2])
    phi_i3_np=np.array(phi_i3(energies[index+1:len(energies)]))
    arraydawg=np.concatenate([phi_i1_np,phi_i2_np,phi_i3_np])
    phi_iteration=interpolate.interp1d(energies,arraydawg)
    converged = (np.linalg.norm(phi_prev(energies) - phi_iteration(energies))/\
                 np.linalg.norm(phi_iteration(energies)) < tolerance) or (iteration >= max_iterations)
    iteration += 1
print("Number of iterations",iteration)
#Lets Plot if we want to.
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
	#Plot the Fluxes
	fig = plt.figure(figsize=(8,6), dpi=1600)
	plt.loglog(energies,phi(energies)/np.sum(phi(energies)), label="Uncollided")
	plt.loglog(energies,phi_iteration(energies)/np.sum(phi_iteration(energies)), label="Scattering Adjusted")
	plt.xlabel("E (MeV)")
	plt.ylabel("$\phi(E)$ (MeV$^{-1}$)")
	plt.legend(loc=4)
	plt.savefig("SpectrumComparison.pdf")
