import numpy as np
#open total cross-section
sigma_t = np.genfromtxt('u235_total.csv', delimiter=",")
#open scattering cross-section
sigma_s = np.genfromtxt('u235_elastic.csv', delimiter=",", skip_header=1)



