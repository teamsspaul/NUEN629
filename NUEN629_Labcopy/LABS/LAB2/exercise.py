def coordLookup_l(i, j, k, I, J): #two helper functions to look up stuff
    """get the position in a 1-D vector
    for the (i,j,k) index
    """
    return i + j*I + k*J*I

def coordLookup_ijk(l, I, J):
    """get the position in a (i,j,k)  coordinates
    for the index l in a 1-D vector
    """
    k = (l // (I*J)) + 1 #double slash is integer division in python
    j = (l - k*J*I) // I + 1
    i = l - (j*I + k*J*I)-1
    return i,j,k
	
# We will also need to define arrays that contain the diffusion coefficient, $D$, the macroscopic absorption cross-section, $\Sigma_\mathrm{a}$, and the source, $Q$.  We will define these with $ijk$ coordinates.
# 
# The there are six boundary conditions that we need to define for the problem: the two $x$-faces that we call the left and right faces, the two $y$-faces that we call front and back, and the two $z$-faces that we call top and bottom. For each of these we need to define 3 values, $\mathcal{A}$, $\mathcal{B}$, and $\mathcal{C}$.
# 
# The next code snippet will build the linear system and solve it using SciPy.  We use sparse matrices for this problem. 

# In[269]:

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg


def diffusion_steady_fixed_source(Dims,Lengths,BCs,D,Sigma,Q, tolerance=1.0e-12, LOUD=False):
    """Solve a steady state, single group diffusion problem with a fixed source
    Inputs:
        Dims:            number of zones (I,J,K)
        Lengths:         size in each dimension (Lx,Ly,Lz)
        BCs:             A, B, and C for each boundary, there are 8 of these
        D,Sigma,Q:       Each is an array of size (I,J,K) containing the quantity
    Outputs:
        x,y,z:           Vectors containing the cell centers in each dimension
        phi:             A vector containing the solution
    """
    I = Dims[0]
    J = Dims[1]
    K = Dims[2]
    L = I*J*K
    Nx = Lengths[0]
    Ny = Lengths[1]
    Nz = Lengths[2]
    
    hx,hy,hz = np.array(Lengths)/np.array(Dims)
    ihx2,ihy2,ihz2 = (1.0/hx**2,1.0/hy**2,1.0/hz**2)
    
    #allocate the A matrix, and b vector
    A = sparse.lil_matrix((L,L))
    b = np.zeros(L)
    
    temp_term = 0
    for k in range(K):
        for j in range(J):
            for i in range(I):
                temp_term = Sigma[i,j,k]
                row = coordLookup_l(i,j,k,I,J)
                b[row] = Q[i,j,k]
                #do x-term left
                if (i>0):
                    Dhat = 2* D[i,j,k]*D[i-1,j,k] / (D[i,j,k] + D[i-1,j,k])
                    temp_term += Dhat*ihx2
                    A[row, coordLookup_l(i-1,j,k,I,J)]  = -Dhat*ihx2
                else:
                    bA,bB,bC = BCs[0,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (i<I-1):
                            temp_term += -1.5*D[i,j,k]*bA/bB/hx
                            b[row] += -D[i,j,k]/bB*bC/hx
                            A[row,  coordLookup_l(i+1,j,k,I,J)]  += 0.5*D[i,j,k]*bA/bB/hx
                        else:
                            temp_term += -0.5*D[i,j,k]*bA/bB/hx
                            b[row] += -D[i,j,k]/bB*bC/hx
                    else:
                        temp_term += D[i,j,k]*ihx2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihx2*2.0
                #do x-term right
                if (i < I-1):
                    Dhat = 2* D[i,j,k]*D[i+1,j,k] / (D[i,j,k] + D[i+1,j,k])
                    temp_term += Dhat*ihx2
                    A[row, coordLookup_l(i+1,j,k,I,J)]  += -Dhat*ihx2
                else:
                    bA,bB,bC = BCs[1,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (i>0):
                            temp_term += 1.5*D[i,j,k]*bA/bB/hx
                            b[row] += D[i,j,k]/bB*bC/hx
                            A[row,  coordLookup_l(i-1,j,k,I,J)]  += -0.5*D[i,j,k]*bA/bB/hx
                        else:
                            temp_term += -0.5*D[i,j,k]*bA/bB/hx
                            b[row] += -D[i,j,k]/bB*bC/hx
                  
                    else:
                        temp_term += D[i,j,k]*ihx2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihx2*2.0
                #do y-term
                if (j>0):
                    Dhat = 2* D[i,j,k]*D[i,j-1,k] / (D[i,j,k] + D[i,j-1,k])
                    temp_term += Dhat*ihy2
                    A[row, coordLookup_l(i,j-1,k,I,J)]  += -Dhat*ihy2
                else:
                    bA,bB,bC = BCs[2,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (j<J-1):
                            temp_term += -1.5*D[i,j,k]*bA/bB/hy
                            b[row] += -D[i,j,k]/bB*bC/hy
                            A[row,  coordLookup_l(i,j+1,k,I,J)]  += 0.5*D[i,j,k]*bA/bB/hy
                        else:
                            temp_term += -0.5*D[i,j,k]*bA/bB/hy
                            b[row] += -D[i,j,k]/bB*bC/hy
                    else:
                        temp_term += D[i,j,k]*ihy2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihy2*2.0
                if (j < J-1):
                    Dhat = 2* D[i,j,k]*D[i,j+1,k] / (D[i,j,k] + D[i,j+1,k])
                    temp_term += Dhat*ihy2
                    A[row, coordLookup_l(i,j+1,k,I,J)]  += -Dhat*ihy2
                else:
                    bA,bB,bC = BCs[3,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (j>0):
                            temp_term += 1.5*D[i,j,k]*bA/bB/hy
                            b[row] += D[i,j,k]/bB*bC/hy
                            A[row,  coordLookup_l(i,j-1,k,I,J)]  += -0.5*D[i,j,k]*bA/bB/hy
                        else:
                            temp_term += 0.5*D[i,j,k]*bA/bB/hy
                            b[row] += D[i,j,k]/bB*bC/hy
                  
                    else:
                        temp_term += D[i,j,k]*ihy2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihy2*2.0
                #do z-term
                if (k>0):
                    Dhat = 2* D[i,j,k]*D[i,j,k-1] / (D[i,j,k] + D[i,j,k-1])
                    temp_term += Dhat*ihz2
                    A[row, coordLookup_l(i,j,k-1,I,J)]  += -Dhat*ihz2
                else:
                    bA,bB,bC = BCs[4,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (k<K-1):
                            temp_term += -1.5*D[i,j,k]*bA/bB/hz
                            b[row] += -D[i,j,k]/bB*bC/hz
                            A[row,  coordLookup_l(i,j,k+1,I,J)]  += 0.5*D[i,j,k]*bA/bB/hz
                        else:
                            temp_term += -0.5*D[i,j,k]*bA/bB/hz
                            b[row] += -D[i,j,k]/bB*bC/hz
                    else: 
                        temp_term += D[i,j,k]*ihz2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihz2*2.0
                if (k < K-1):
                    Dhat = 2* D[i,j,k]*D[i,j,k+1] / (D[i,j,k] + D[i,j,k+1])
                    temp_term += Dhat*ihz2
                    A[row, coordLookup_l(i,j,k+1,I,J)]  += -Dhat*ihz2
                else:
                    bA,bB,bC = BCs[5,:]
                    if (np.abs(bB) > 1.0e-8):
                        if (k>0):
                            temp_term += 1.5*D[i,j,k]*bA/bB/hz
                            b[row] += D[i,j,k]/bB*bC/hz
                            A[row,  coordLookup_l(i,j,k-1,I,J)]  += -0.5*D[i,j,k]*bA/bB/hz
                        else:
                            temp_term += 0.5*D[i,j,k]*bA/bB/hz
                            b[row] += D[i,j,k]/bB*bC/hz
                  
                    else:
                        temp_term += D[i,j,k]*ihz2*2.0
                        b[row] += D[i,j,k]*bC/bA*ihz2*2.0
                A[row,row] += temp_term
    phi,code = splinalg.cg(A,b, tol=tolerance)
    if (LOUD):
        print("The CG solve exited with code",code)
    phi_block = np.zeros((I,J,K))
    for k in range(K):
        for j in range(J):
            for i in range(I):
                phi_block[i,j,k] = phi[coordLookup_l(i,j,k,I,J)]
    x = np.linspace(hx*.5,Nx-hx*.5,I)
    y = np.linspace(hy*.5,Ny-hy*.5,J)
    z = np.linspace(hz*.5,Nz-hz*.5,K)
    if (I*J*K <= 10):
        print(A.toarray())
    return x,y,z,phi_block
	
	
	
	

# To test this code we will solve a simple infinite medium problem.  We will have reflective boundary conditions everywhere, $\mathcal{A} = \mathcal{C} = 0$, and $\mathcal{B} = 1$. We will set the diffusion coefficient to 3 and $\Sigma_\mathrm{a} = 2$ and $Q=1$.  The solution to this problem is $\phi = 0.5$ everywhere.  

# In[270]:

I = 3
J = 2
K = 1
Sigma = np.ones((I,J,K))*2
D = Sigma*3/2
Q = Sigma*0.5
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0

x,y,z,phi_infinite_medium = diffusion_steady_fixed_source((I,J,K),(3,2,1),BCs,D,Sigma,Q)
print("Solution is")
print(phi_infinite_medium)


# That solution looks good.  Now let's try a different problem. It will have vacuum Marshak conditions, and $Q = D = \Sigma_\mathrm{a} = 1$. The solution to this problem is
# $$ \phi(x) = \frac{e^{1-x}+e^x+1-3 e}{1-3 e}.$$

# In[271]:

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#solve in x direction
print("Solving Problem in X direction")
I = 50
J = 1
K = 1
Nx = 1
Sigma = np.ones((I,J,K))
D = Sigma.copy()
Q = Sigma.copy()
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0
BCs[0,:] = [0.25,-D[0,0,0]/2,0]
BCs[1,:] = [0.25,D[I-1,0,0]/2,0]

x,y,z,phi_x = diffusion_steady_fixed_source((I,J,K),(Nx,Nx*1,Nx),BCs,D,Sigma,Q)
plt.plot(x,phi_x[:,0,0],label='x')
solution = (np.exp(1-x) + np.exp(x) + 1 - 3*np.exp(1))/(1-3*np.exp(1))
#solve in y direction
print("Solving Problem in Y direction")
I = 1
J = 50
K = 1
Nx = 1
Sigma = np.ones((I,J,K))
D = Sigma.copy()
Q = Sigma.copy()
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0
BCs[2,:] = [0.25,-D[0,0,0]/2,0]
BCs[3,:] = [0.25,D[0,J-1,0]/2,0]
x,y,z,phi_y = diffusion_steady_fixed_source((I,J,K),(Nx,Nx*1,Nx),BCs,D,Sigma,Q)
plt.plot(y,phi_y[0,:,0],label='y')

#solve in z direction
print("Solving Problem in Z direction")
I = 1
J = 1
K = 50
Nx = 1
Sigma = np.ones((I,J,K))
D = Sigma.copy()
Q = Sigma.copy()
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0
BCs[4,:] = [0.25,-D[0,0,0]/2,0]
BCs[5,:] = [0.25,D[0,J-1,0]/2,0]
x,y,z,phi_z = diffusion_steady_fixed_source((I,J,K),(Nx,Nx*1,Nx),BCs,D,Sigma,Q)
plt.plot(z,phi_z[0,0,:],label='z')

plt.plot(z,solution,'o-',label='Analytic')
plt.xlabel("x,y, or z")
plt.ylabel("$\phi$")
plt.legend(loc=4)
plt.savefig("XYorZ.pdf")
plt.show()
plt.close()


def lattice(Lengths,Dims):
    I = Dims[0]
    J = Dims[1]
    K = Dims[2]
    L = I*J*K
    Nx = Lengths[0]
    Ny = Lengths[1]
    Nz = Lengths[2]
    hx,hy,hz = np.array(Lengths)/np.array(Dims)
    
    Sigma = np.ones((I,J,K))*1
    Q = np.zeros((I,J,K))
    for k in range(K):
        for j in range(J):
            for i in range(I):
                x = (i+0.5)*hx
                y = (j+0.5)*hy
                z = (k+0.5)*hz
                
                if (x>=3.0) and (x<=4.0): 
                    if (y>=3.0) and (y<=4.0):
                        Q[i,j,k] = 1.0
                    if (y>=1.0) and (y<=2.0):
                        Sigma[i,j,k] = 10.0
                if ( ((x>=1.0) and (x<=2.0)) or ((x>=5.0) and (x<=6.0))): 
                    if ( ((y>=1.0) and (y<=2.0)) or
                        ((y>=3.0) and (y<=4.0)) or
                        ((y>=5.0) and (y<=6.0))):
                        Sigma[i,j,k] = 10.0
                if ( ((x>=2.0) and (x<=3.0)) or ((x>=4.0) and (x<=5.0))): 
                    if ( ((y>=2.0) and (y<=3.0)) or
                        ((y>=4.0) and (y<=5.0))):
                        Sigma[i,j,k] = 10.0
    
    D = 1.0/(3.0*Sigma)
    return D,Q,Sigma
    
    
I = 150
J = 150
K = 1
Nx = 7
D,Q,Sigma = lattice((Nx,Nx,1),(I,J,K))
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0
BCs[(0,2,4),:] = [0.25,-D[0,0,0]/2,0]
BCs[(1,3,5),:] = [0.25,D[I-1,0,0]/2,0]

x,y,z,phi = diffusion_steady_fixed_source((I,J,K),(Nx,Nx*1,Nx),BCs,D,Sigma,Q)
plt.pcolor(x,y,np.transpose(np.log10(np.abs(phi[:,:,0]))))
plt.colorbar()
plt.clim([-8,0])
plt.axes().set_aspect('equal',adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title("$\log_{10} (\phi)$")
plt.savefig("Lattice.pdf")
plt.show()
plt.close()

# In[273]:

np.min(phi)


# Now we will make this solve a fixed source, time-dependent problem

# In[274]:

def time_dependent_diffusion(phi0, v, dt, T, Dims,Lengths,BCs,D,Sigma,Q, tolerance=1.0e-12):
    """Solve a time dependent diffusion problem
    Inputs:
    phi0:       initial condition
    v:          particle speed
    dt:         time step size
    T:          final time
    The remaining inputs are the same as for diffusion_steady_fixed_source
    Outputs:
        x,y,z:           Vectors containing the cell centers in each dimension
        phi:             A vector containing the solution at time T
    """
    numsteps = int(T/dt)
    for step in range(numsteps):
        Qhat = Q + phi0/dt/v
        Sighat = Sigma + 1.0/dt/v
        print("Solving for time",(step+1)*dt)
        x,y,z,phi0 = diffusion_steady_fixed_source(Dims,Lengths,BCs,D,   #Changed Sighat, and Qhat
                                                   Sighat,Qhat, tolerance)
    return x,y,z,phi0
    
    


# In[275]:

I = 35
J = 35
K = 1
Nx = 7
D,Q,Sigma = lattice((Nx,Nx,1),(I,J,K))
BCs = np.ones((6,3))
BCs[:,0] = 0
BCs[:,2] = 0
BCs[(0,2,4),:] = [0.25,-D[0,0,0]/2,0]
BCs[(1,3,5),:] = [0.25,D[I-1,0,0]/2,0]

phi = np.zeros((I,J,K))
x,y,z,phi = time_dependent_diffusion(phi,1,0.1,1,(I,J,K),(Nx,Nx*1,Nx),BCs,D,Sigma,Q)
plt.pcolor(x,y,np.transpose(np.log10(np.abs(phi[:,:,0]))))
plt.colorbar()
plt.clim([-8,0])
plt.axes().set_aspect('equal',adjustable='box')
plt.xlabel('x')
plt.ylabel('y')
plt.title("$\log_{10} (\phi)$")
plt.savefig("Lattice_T2.pdf")
plt.show()
plt.close()

# In[276]:

def lattice2G(Lengths,Dims):
    I = Dims[0]
    J = Dims[1]
    K = Dims[2]
    L = I*J*K
    Nx = Lengths[0]
    Ny = Lengths[1]
    Nz = Lengths[2]
    hx,hy,hz = np.array(Lengths)/np.array(Dims)
    
    Sigmaag = np.ones((I,J,K,2))*1
    Sigmasgg = np.zeros((I,J,K,2,2))
    nuSigmafg = np.zeros((I,J,K,2))
    nug = np.ones((I,J,K,2))*2.3
    chig = np.zeros((I,J,K,2))
    D = np.zeros((I,J,K,2))
    Q = np.zeros((I,J,K,2))
    
    Sigmasgg[:,:,:,0,0] = 0.05
    Sigmasgg[:,:,:,0,1] = 0.1
    Sigmasgg[:,:,:,1,1] = 0.25
    
    
    
    
    for k in range(K):
        for j in range(J):
            for i in range(I):
                x = (i+0.5)*hx
                y = (j+0.5)*hy
                z = (k+0.5)*hz
                
            
                if (x>=3.0) and (x<=4.0): 
                    if (y>=3.0) and (y<=4.0):
                        Q[i,j,k,0] = 1.0 
                    if (y>=1.0) and (y<=2.0):
                        Sigmaag[i,j,k,(0,1)] = [10.0,15.0]
                        nuSigmafg[i,j,k,(0,1)] = [5.0,7.5]
                        chig[i,j,k,(0,1)] = [1.0,0.0]
                elif ( ((x>=1.0) and (x<=2.0)) or ((x>=5.0) and (x<=6.0))): 
                    if ( ((y>=1.0) and (y<=2.0)) or
                        ((y>=3.0) and (y<=4.0)) or
                        ((y>=5.0) and (y<=6.0))):
                        Sigmaag[i,j,k,(0,1)] = [10.0,15.0]
                        nuSigmafg[i,j,k,(0,1)] = [5.0,7.5]
                        chig[i,j,k,(0,1)] = [1.0,0.0]
                elif  ( ((x>=2.0) and (x<=3.0)) or ((x>=4.0) and (x<=5.0))): 
                    if ( ((y>=2.0) and (y<=3.0)) or
                        ((y>=4.0) and (y<=5.0))):
                        Sigmaag[i,j,k,(0,1)] = [10.0,15.0]
                        nuSigmafg[i,j,k,(0,1)] = [5.0,7.5]
                        chig[i,j,k,(0,1)] = [1.0,0.0]
    
    D[:,:,:,0] = 1.0/(3.0*(Sigmaag[:,:,:,0]))
    D[:,:,:,1] = 1.0/(3.0*(Sigmaag[:,:,:,1]))
    return   Sigmaag,Sigmasgg,nuSigmafg,nug,chig,D,Q


# In[277]:


def steady_multigroup_diffusion(G,Dims,Lengths,BCGs,
                                Sigmatg,Sigmasgg,nuSigmafg,
                                nug,chig,D,Q,
                                lintol=1.0e-8,grouptol=1.0e-6,maxits = 12,
                                LOUD=False):
    I = Dims[0]
    J = Dims[1]
    K = Dims[2]
    iteration = 1
    converged = False
    phig = np.zeros((I,J,K,G))
    while not(converged):
        phiold = phig.copy()
        for g in range(G):
            #compute Qhat and Sigmar
            Qhat = Q[:,:,:,g].copy()
            Sigmar = Sigmatg[:,:,:,g] - Sigmasgg[:,:,:,g,g] - chig[:,:,:,g]*nuSigmafg[:,:,:,g]
            for gprime in range(0,G):
                if (g != gprime):
                    Qhat += (chig[:,:,:,g]*nuSigmafg[:,:,:,gprime] + Sigmasgg[:,:,:,gprime,g])*phig[:,:,:,gprime]
            x,y,z,phi0 = diffusion_steady_fixed_source(Dims,Lengths,BCGs[:,:,g],D[:,:,:,g],
                                                       Sigmar,Qhat, lintol)
            phig[:,:,:,g] = phi0.copy()
        change = np.linalg.norm(np.reshape(phig - phiold,I*J*K*G)/(I*J*K*G))
        if LOUD:
            print("Iteration",iteration,"Change =",change)
        iteration += 1
        converged = (change < grouptol) or iteration > maxits
    return x,y,z,phig







