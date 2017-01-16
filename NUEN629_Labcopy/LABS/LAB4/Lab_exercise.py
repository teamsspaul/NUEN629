



import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import inv



#cross-sections in barns
position = {'28':0,'29':1,'20':2,'39':3,'30':4,'49':5,'40':6,'41':7,'51':8}
previous_cap = {'28':-1,'29':0,'20':1,'39':-1,'30':3,'49':-1,'40':5,'41':6,'51':-1}
previous_beta = {'28':-1,'29':-1,'20':-1,'39':1,'30':2,'49':3,'40':4,'41':-1,'51':7}
sig_gamma = 1.0E-24*np.array([2.7,22.0,0,60,274,290,326,532])
sig_a = 1.0E-24*np.array([12.0,22,0,60,0,274+698,290+53,326+938,535])
lam = np.array([0,42.4737,1.17982,0.294956,138.629,0,0,0.000131877,0])

A = np.zeros((9,9))

phi = 1.0e14 * 60 * 60 * 24 #10^14 1/cm^2/s in 1/cm^2/day
for i in position:
    row = position[i]
    A[row,row] = -lam[row] - phi*sig_a[row]
    if previous_cap[i]>=0:
            A[row,previous_cap[i]] = phi*sig_gamma[previous_cap[i]]
    if previous_beta[i]>=0:
            A[row,previous_beta[i]] = lam[previous_beta[i]]
plt.spy(A)
b = np.zeros(9)
b[0] = 1.0



Npoints = (24,) #(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32)

for N in Npoints:
    pos = 0
    theta = np.pi*np.arange(1,N,2)/N
    z = N*(.1309 - 0.1194*theta**2 + .2500j*theta)
    w = N*(- 2*0.1194*theta + .2500j)
    c = 1.0j/N*np.exp(z)*w
    #plt.plot(np.real(z),np.imag(z),'o-')
    #plt.show()
    u = np.zeros(9)
    for k in range(int(N/2)):
        n,code = splinalg.gmres(z[k]*sparse.identity(9)  - A*365,b, tol=1e-12, maxiter=2000)
        if (code):
            print(code)
        u = u- c[k]*n
    u = 2*np.real(u)
    print(u)
	
	

v=np.dot(expm(A*365),b)
print(v)
Id=np.zeros(9)
w,code=splinalg.gmres(sparse.identity(9)-A*365,b, tol=1e-16,maxiter=5000)
print(w)

#compute exp
Npoints = (32,)
errors = np.zeros(len(Npoints))
pos = 0
for N in Npoints:
    theta = np.pi*np.arange(1,N,2)/N
    z = N*(.1309 - 0.1194*theta**2 + .2500j*theta)
    w = N*(- 2*0.1194*theta + .2500j)
    c = 1.0j/N*np.exp(z)*w
    plt.plot(np.real(z),np.imag(z),'o-')
    u = 0
    for k in range(int(N/2)):
        u -=  c[k]/((z[k] + 1))
    errors[pos] = np.abs(2*np.real(u)-0.3678794411714423215955238)
    pos += 1
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.axis([-40,10,0,30])
print("Errors =",errors)

