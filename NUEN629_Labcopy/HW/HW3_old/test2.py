import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
import sys
import os
import os
 
from scipy.interpolate import griddata
 
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2





grid_x, grid_y = np.mgrid[0:1:160j, 0:1:160j]
points = np.zeros((160,2))
pointsx=np.random.rand(160,1)
pointsy=np.random.rand(160,1)
points[:,0]=pointsx.T
points[:,1]=pointsy.T


values = func(points[:,0], points[:,1])

#print (pointsx,pointsy)
print (values.shape)
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
#print(grid_z0[1,1])