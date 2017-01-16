import numpy as np #loads
#open total cross-section
sigma_t = np.genfromtxt('u235_total.csv', delimiter=",") #creats a two D array (energy in MEV X Sec in barns)
#open scattering cross-section
sigma_s = np.genfromtxt('u235_elastic.csv', delimiter=",", skip_header=1) #skip text