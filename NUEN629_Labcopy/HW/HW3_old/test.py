import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
import sys
import os
import os

Geometry = np.genfromtxt('Geometry.txt', delimiter=",")

if(os.path.isfile('phi_out.txt')):
	os.remove('phi_out.txt')
np.savetxt('phi_out.txt',Geometry,delimiter=',')

if(os.path.isfile('k_effective_out.txt')):
	os.remove('k_effective_out.txt')

	
k=1.00232
k2=1.2231
with open("k_effective_out.txt","a") as myfile:
	myfile.write(str(k)+"\n")
with open("k_effective_out.txt","a") as myfile:
	myfile.write(str(k2)+"\n")
	
k_old_file=np.genfromtxt('k_effective_out.txt',delimiter=",")
print (k_old_file[len(k_old_file[:])-1])

print(len(np.genfromtxt('Geometry.txt', delimiter=",")[:,1]))
