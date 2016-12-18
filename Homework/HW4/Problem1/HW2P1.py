import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

coord=[0]*2500
mu=[0]*2500

for i in range(0,50):
    for j in range(0,50):
        coord[i+j*50] = [i/50,j/50]

k = np.zeros((50, 50, 50, 50))
k[0,0,0,0] = 5
k2= np.zeros((2500,2500))

print(k[0,0,0,0])

for i in range(0,50):
    print(i)
    for j in range(0,50):
        for l in range(0,50):
            for m in range(0,50):
                k[i,j,l,m] = np.exp(-np.abs(i/50.0-j/50.0)-np.abs(l/50.0-m/50.0))
                k2[i*50+l,j*50+m] = np.exp(-np.abs(i/50.0-j/50.0)-np.abs(l/50.0-m/50.0))

for m in range(0,4):
    U=multivariate_normal.rvs(mu,k2)
    Z = [[U[50*i+j] for j in range(0,50)] for i in range(0,50)]
    plt.imshow(Z, extent=(0,1,0,1))
    plt.savefig("HW2P1realization"+str(m)+".pdf")
