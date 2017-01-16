import numpy as np
import sys
import matplotlib.pyplot as plt

x=np.array([0,1,2,3,4,5,6,7,8])
y=np.array([2,4,4])
zz=np.array([100,100,100])
z=np.concatenate([x,zz,y])
#print(z)


print(y*zz)

for xx in range(0, 9):
    print ("We're on time %d" % (x[xx]))