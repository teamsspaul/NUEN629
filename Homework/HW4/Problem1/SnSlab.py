
# coding: utf-8

# We begin by defining a function that will perform a transport sweep in 1-D slabs.

# In[78]:

import numpy as np
def sweep1D(I,hx,q,sigma_t,mu,boundary):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
    Outputs:
        psi:             value of angular flux in each zone
    """
    assert(np.abs(mu) > 1.e-10)
    psi = np.zeros(I)
    ihx = 1./hx
    if (mu > 0):
        psi_left = boundary
        for i in range(I):
            psi_right = (q[i]*0.5 + mu*psi_left*ihx)/(sigma_t[i] + mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_left = psi_right
    else:
        psi_right = boundary
        for i in reversed(range(I)):
            psi_left = (q[i]*0.5 - mu*psi_right*ihx)/(sigma_t[i] - mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_right = psi_left
    return psi


# We create the source iteration algorithm to that will call the sweep next. Luckily, NumPy has the Gauss-Legendre quadrature points built in.

# In[348]:

def source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones
        hx:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi:             value of scalar flux in each zone
    """
    phi = np.zeros(I)
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    iteration = 1
    while not(converged):
        phi = np.zeros(I)
        #sweep over each direction
        for n in range(N):
            tmp_psi = sweep1D(I,hx,q + phi_old*sigma_s,sigma_t,MU[n],BCs[n])
            phi += tmp_psi*W[n]
        #check convergence
        change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
        converged = (change < tolerance) or (iteration > maxits)
        if (LOUD>0) or (converged and LOUD<0):
            print("Iteration",iteration,": Relative Change =",change)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
        iteration += 1
        phi_old = phi.copy()
    x = np.linspace(hx/2,I*hx-hx/2,I)
    return x, phi


# In[368]:

#simple test problem
I = 30
hx = 1./I
q = np.zeros(I)
sigma_t = np.ones(I)
sigma_s = 0*sigma_t
N = 2
BCs = np.zeros(N)
BCs[(N/2.):N] = 1.0

x,phi_sol = source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=True )


# In[369]:

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.plot(x,phi_sol,'o',label="Numerical Solution")
X = np.linspace(0,1,100)
plt.plot(X,np.exp(-X*np.sqrt(3.0)),label="Exact Solution")
plt.legend()
plt.show()


# In[370]:

#Now a more complicated test
I = 200
hx = 100./I
q = np.ones(I)
sigma_t = 1.5*np.ones(I)
sigma_s = 1.0*np.ones(I)
N = 2
BCs = np.zeros(N)

x,phi_sol = source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-12,maxits = 100, LOUD=-1 )
plt.plot(x,phi_sol,'o',label="Numerical Solution")
plt.legend()
plt.show()


# In[308]:

#Now a more complicated test, with a higher scattering ratio
I = 100
hx = 100./I
q = np.ones(I)
sigma_t = (1.55)*np.ones(I)
sigma_s = 1.549*np.ones(I)
N = 2
BCs = np.zeros(N) + 0.5/(sigma_t[0]-sigma_s[0])

x,phi_sol = source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-12,maxits = 20000, LOUD=-1 )
plt.plot(x,phi_sol,'o',label="Numerical Solution")
plt.plot(x,x*0+1/(sigma_t[0]-sigma_s[0]),'*-',label="Inf medium Solution")
plt.legend(loc=0)
plt.show()


# Reed's Problem

# In[309]:

#in this case all three are constant
def Sigma_t(r):
    value = 0 + ((1.0*(r>=14) + 1.0*(r<=4)) +
                 5.0 *((np.abs(r-11.5)<0.5) or (np.abs(r-6.5)<0.5)) +
                 50.0 * (np.abs(r-9)<=2) )
    return value;
def Sigma_a(r):
    value = 0 + (0.1*(r>=14) + 0.1*(r<=4) +
                 5.0 *((np.abs(r-11.5)<0.5) or (np.abs(r-6.5)<0.5)) +
                 50.0 * (np.abs(r-9)<=2) )
    return value;
def Q(r):
    value = 0 + 1.0*((r<16) * (r>14))+ 1.0*((r>2) * (r<4)) + 50.0*(np.abs(r-9)<=2)
    return value;


# In[310]:

I = 200
L = 18.
hx = L/I
xpos = hx/2;
q = np.zeros(I)
sigma_t = np.zeros(I)
sigma_s = np.zeros(I)
for i in range(I):
    sigma_t[i] = Sigma_t(xpos)
    sigma_s[i] = sigma_t[i]-Sigma_a(xpos)
    q[i] = Q(xpos)
    xpos += hx
N = 18
BCs = np.zeros(N)

x,phi_sol = source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-12,maxits = 1000, LOUD=-1 )
plt.plot(x,phi_sol,'-',label="Numerical Solution")
plt.show()


### Multigroup problems

# In[362]:

def multigroup_ss(I,hx,G,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Solve multigroup SS problem
    Inputs:
        I:               number of zones
        hx:              size of each zone
        G:               number of groups
        q:               source array
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    phi = np.zeros((I,G))
    phi_old = phi.copy()
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    iteration = 1
    while not(converged):
        phi = np.zeros((I,G))
        #solve each group
        if (LOUD > 0):
            print("Group Iteration",iteration)
            print("====================")
        for g in range(G):
            #compute scattering source
            Q = q[:,g].copy()
            for gprime in range(G):
                Q += phi[:,gprime]*sigma_s[:,gprime,g]
            if (LOUD > 0):
                print("Group",g)
            x,phi[:,g] = source_iteration(I,hx,Q,sigma_t[:,g],sigma_s[:,g,g],N,BCs[:,g],
                                        tolerance = tolerance*0.1,maxits = 1000, LOUD=LOUD)
        #check convergence
        change = np.linalg.norm(np.reshape(phi-phi_old,(I*G,1)))/np.linalg.norm(np.reshape(phi,(I*G,1)))
        converged = (change < tolerance) or (iteration > maxits)
        if (iteration > maxits):
            print("Warning: Group Iterations did not converge")
        if (LOUD>0) or (converged and LOUD<0):
            print("====================")
            print("Outer (group) Iteration",iteration,": Relative Change =",change)
            print("====================")
        iteration += 1
        phi_old = phi.copy()
    return x, phi


# In[363]:

#simple test problem
I = 30
hx = 1./I
G = 2
q = np.zeros((I,G))
sigma_t = np.ones((I,G))
sigma_s = np.zeros((I,G,G))
N = 2
BCs = np.zeros((N,G))
BCs[(N/2):N,:] = 1.0

x,phi_sol = multigroup_ss(I,hx,G,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=True )
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
X = np.linspace(0,1,100)
plt.plot(X,np.exp(-X*np.sqrt(3.0)),label="Exact Solution")
plt.legend()
plt.show()


# In[365]:

#Now a more complicated test
I = 2000
hx = 20./I
G = 2
q = np.ones((I,G))
q[:,1] = 0
sigma_t = np.ones((I,G))
sigma_s = np.zeros((I,G,G))
sigma_s[:,0,1] = 1
sigma_s[:,0,0] = 0.5
sigma_s[:,1,1] = 0.25
sigma_t[:,0] = 5.5
sigma_t[:,1] = 5.25

N = 2
BCs = np.zeros((N,G))
BCs[(N/2):N,:] = 0.0

x,phi_sol = multigroup_ss(I,hx,G,q,sigma_t,sigma_s,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=True )
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
plt.legend()
plt.show()


# K-eigenvalue solve

# In[356]:

def multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs, tolerance = 1.0e-8,maxits = 100, LOUD=False ):
    """Solve k eigenvalue problem
    Inputs:
        I:               number of zones
        hx:              size of each zone
        G:               number of groups
        sigma_t:         array of total cross-sections format [i,g]
        sigma_s:         array of scattering cross-sections format [i,gprime,g]
        nusigma_f:       array of nu times fission cross-sections format [i,g]
        chi:             energy distribution of fission neutrons
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        x:               value of center of each zone
        phi(I,G):             value of scalar flux in each zone
    """
    phi = np.random.rand(I,G)
    phi_old = phi.copy()
    k = 1.0
    converged = False
    MU, W = np.polynomial.legendre.leggauss(N)
    iteration = 1
    while not(converged):
        #compute fission source
        Q = sigma_t*0.0
        for g in range(G):
            for gprime in range(G):
                Q[:,g] += chi[:,g] * phi_old[:,gprime] * nusigma_f[:,gprime]
        x,phi = multigroup_ss(I,hx,G,Q,sigma_t,sigma_s,N,BCs, tolerance = tolerance*0.1,maxits = 100, LOUD=LOUD )
        knew = np.linalg.norm(np.reshape(nusigma_f*phi,I*G))/np.linalg.norm(np.reshape(nusigma_f*phi_old,I*G))
        #check convergence
        solnorm = np.linalg.norm(np.reshape(phi_old,I*G))
        converged = ((np.abs(knew-k) < tolerance) or (iteration > maxits))
        if (LOUD>0) or (converged):
            print("*************************====================")
            print("Power Iteration",iteration,": k =",knew,"Relative Change =",np.abs(knew-k))
            print("*************************====================")
        iteration += 1
        k = knew
        phi_old = phi/k
    if (iteration > maxits):
        print("Warning: Power Iterations did not converge")
    return x, k, phi_old


# In[359]:

#Should have k=1 if inf medium (both groups are the same)
#Now a more complicated test
I = 100
hx = 10./I
G = 2
q = np.ones((I,G))

sigma_t = np.ones((I,G))
nusigma_f = np.ones((I,G))
chi = 0.5*np.ones((I,G))
sigma_s = np.zeros((I,G,G))
sigma_s[:,0,1] = 0.0
sigma_s[:,0,0] = 1.0
sigma_s[:,1,1] = 1.0
sigma_t[:,0] = 5.5
sigma_t[:,1] = 5.5
nusigma_f[:,0] = 4.5
nusigma_f[:,1] = 4.5

N = 2
BCs = np.zeros((N,G))
BCs[(N/2):N,:] = 0.0

x,k,phi_sol = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,
                           tolerance = 1.0e-6,maxits = 100, LOUD=0 )
print("k =",k)
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
plt.legend()
plt.show()


# In[371]:

#Should have k=2 if inf medium
I = 100
hx = 200./I
G = 2
q = np.ones((I,G))

sigma_t = np.ones((I,G))
nusigma_f = np.ones((I,G))
chi = np.ones((I,G))
chi[:,1] = 0 #only fast fission neutrons born
sigma_s = np.zeros((I,G,G))
sigma_s[:,0,1] = 4.5
sigma_s[:,0,0] = 1.0
sigma_s[:,1,1] = 1.0
sigma_t[:,0] = 5.5
sigma_t[:,1] = 5.5
nusigma_f[:,0] = 4.5
nusigma_f[:,1] = 4.5

N = 2
BCs = np.zeros((N,G))
BCs[(N/2):N,:] = 0.0

x,k,phi_sol = multigroup_k(I,hx,G,sigma_t,sigma_s,nusigma_f,chi,N,BCs,
                           tolerance = 1.0e-6,maxits = 1000, LOUD=0 )
print("k =",k)
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
plt.legend()
plt.show()


### Product Quadrature

# In[499]:

def prod_quad(N):
    """Compute ordinates and weights for product quadrature
    Inputs:
        N:               Order of Legendre or Chebyshev quad
    Outputs:
        w:               weights
        eta,xi,mu:       direction cosines (x,y,z)
    """
    #get legendre quad
    MUL, WL = np.polynomial.legendre.leggauss(N)
    #get chebyshev y's
    Y, WC = np.polynomial.chebyshev.chebgauss(N)
    #get all pairs
    place = 0
    eta = np.zeros(N*N*2)
    xi = np.zeros(N*N*2)
    mu = np.zeros(N*N*2)
    w = np.zeros(N*N*2)

    for i in range(N):
        for j in range(N):
            mul = MUL[i]
            y = Y[j]
            mu[place] = mul
            mu[place+1] = mul
            gamma = np.arccos(y)
            gamma2 = -gamma
            sinTheta = np.sqrt(1-mul*mul)
            eta[place] =   sinTheta*np.cos(gamma)
            eta[place+1] = sinTheta*np.cos(gamma2)
            xi[place] =   sinTheta*np.sin(gamma)
            xi[place+1] = sinTheta*np.sin(gamma2)
            w[place] = WL[i]*WC[j]
            w[place+1] = WL[i]*WC[j]
            place += 2
    return w, eta,xi,mu



# In[500]:

w,eta,xi,mu = prod_quad(18)
#check integrals
print("This should be 0:",sum(w*mu**7))
print("This should be 0:",sum(w*eta**7))
print("This should be 0:",sum(w*xi**7))
print("This should be 0.11968:",sum(w*xi**2*eta**2*mu**2))
print("This should be 0.00379076:",sum(w*xi**12*eta**2*mu**2))


# In[501]:

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(eta,xi,mu)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()


# In[502]:

#just positive mus
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.scatter(eta[mu>0],xi[mu>0],mu[mu>0])
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()


# In[ ]:
