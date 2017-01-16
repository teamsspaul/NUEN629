import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
import sys
import os
import os
from scipy import interpolate
import matplotlib.pyplot as plt
 
g=0
k=0
z=np.zeros(1)
z[0]=0.5
J=160
I=160
 
phi_out_g_k=np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")
x_out=np.genfromtxt('x.txt',delimiter=",")
y_out=np.genfromtxt('y.txt',delimiter=",")
f=interpolate.interp2d(x_out,y_out,phi_out_g_k,kind='cubic',bounds_error=False,fill_value=phi_out_g_k[-1,:])

phi_new=f(x_out,y_out)
plt.pcolor(np.transpose(np.log10(phi_new[:,:])),rasterized=True); plt.colorbar();
plt.clim(-6,np.log10(phi_new[:,:]).max())
plt.savefig("PLOTTTT.pdf")
#points = np.zeros((len(x_out),2))
#
#points[:,0]=x_out.T
#points[:,1]=y_out.T
#
#grid_x, grid_y = np.mgrid[0:231:10j, 0:231:20j]
#from scipy.interpolate import griddata
#print (phi_out_g_k.shape)
#

#phi_interp_g_k = griddata(points,phi_out_g_k,(grid_x,grid_y),method='linear')#, fill_value=phi_out_g_k[-1,1])

#grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
#phi0[:,:,k,g]=phi_interp_g_k

#phi=np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")

#print(len(np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")[:,1]))
if(len(np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")[:,1])<=J and len(np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")[1,:])<=I):
	print ("Hello")
