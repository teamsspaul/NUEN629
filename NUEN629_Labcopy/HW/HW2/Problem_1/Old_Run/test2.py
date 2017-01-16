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
for xx in range(0, len(energies)):
	if (energies[xx]<=2.53e-8):
		index=xx
print(energies[index:index+5])

