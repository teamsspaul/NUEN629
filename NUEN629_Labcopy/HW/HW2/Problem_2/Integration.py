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

#Perform Integration
from scipy import integrate
from scipy.integrate import trapz

#Perform the integration For Group 1
EE=1e-9
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index=xx
g1=integrate.trapz(chi(energies[0:index]),energies[0:index])


#Perform the integration for group 2
EE=1e-7
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index2=xx
g2=integrate.trapz(chi(energies[index:index2]),energies[index:index2])


#Perform the integration for group 3
EE=0.00001
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index3=xx
g3=integrate.trapz(chi(energies[index2:index3]),energies[index2:index3])
#Perform the integration for group 4
EE=0.1
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index4=xx
g4=integrate.trapz(chi(energies[index3:index4]),energies[index3:index4])
#Perform the integration for group 5
EE=20
for xx in range(0, len(energies)):
    if (energies[xx]<=EE):
        index5=xx
g5=integrate.trapz(chi(energies[index4:index5]),energies[index4:index5])

print (g1)
print (g2)
print (g3)
print (g4)
print (g5)

