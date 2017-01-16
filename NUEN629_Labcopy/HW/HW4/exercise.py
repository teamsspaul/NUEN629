import numpy as np
import matplotlib.pyplot as plt
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
    assert(np.abs(mu) > 1e-10)
    psi = np.zeros(I)
    ihx = 1/hx
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

#Now a more complicated test
I = 2000
hx = 20/I
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
%matplotlib inline
plt.plot(x,phi_sol[:,0],'o',label="Group 1")
plt.plot(x,phi_sol[:,1],'o',label="Group 2")
plt.legend()
plt.savefig("savefig.pdf")


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