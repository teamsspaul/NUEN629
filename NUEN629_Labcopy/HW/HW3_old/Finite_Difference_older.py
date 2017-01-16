import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import time
import sys
import os
from scipy import interpolate

start_time = time.time()

#Functions
def coordLookup_l(i, j, k, I, J):
    """get the position in a 1-D vector for the (i,j,k) index"""
    return i + j*I + k*J*I
def coordLookup_ijk(l, I, J):
    """get the position in a (i,j,k) coordinates for the index l in a 1-D vector"""
    k = (l // (I*J)) + 1
    j = (l - k*J*I) // I + 1
    i = l - (j*I + k*J*I)-1
    return i,j,k
def diffusion_steady_fixed_source(I,J,K,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCs,D,Sigma,Q,L,tolerance=1.0e-12,LOUD=False):
    """Solve a steady state, single group diffusion problem with a fixed source
    Inputs:
        Dims:            number of zones (I,J,K)
        Lengths:         size in each dimension (Nx,Ny,Nz)
        BCs:             A, B, and C for each boundary, there are 8 of these
        D,Sigma,Q:       Each is an array of size (I,J,K) containing the quantity
    Outputs:
        x,y,z:           Vectors containing the cell centers in each dimension
        phi:             A vector containing the solution
    """
    
    
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
    #phi,code = splinalg.cg(A,b, tol=tolerance)
    phi = sparse.linalg.spsolve(A,b)
    if (LOUD):
        print("The CG solve exited with code",code)
    phi_block = np.zeros((I,J,K))
    for k in range(K):
        for j in range(J):
            for i in range(I):
                phi_block[i,j,k] = phi[coordLookup_l(i,j,k,I,J)]
    if (I*J*K <= 10):
        print(A.toarray())
    return phi_block
def lattice2G():
    Geometry = np.genfromtxt('Geometry.txt', delimiter=",") #Geometry.txt generated from Excel file
    Power = np.genfromtxt('Power.txt', delimiter=",") #Geometry.txt generated from Excel file
    I=int(Geometry[0,3])
    J=int(Geometry[0,4])
    K=int(Geometry[0,5])
    L=int(Geometry[0,6])
    Nx=Geometry[0,7]
    Ny=Geometry[0,8]
    Nz=Geometry[0,9]
    G=2
    hx,hy,hz = np.array([Geometry[0,0],Geometry[0,1],Geometry[0,2]])
    Sigmaag = np.zeros((I,J,K,G))*1
    Sigmasgg = np.zeros((I,J,K,G,G)) 
    nuSigmafg = np.zeros((I,J,K,G))
    nug = np.ones((I,J,K,G))*2.4
    chig = np.zeros((I,J,K,G))
    D = np.zeros((I,J,K,G))
    Q = np.ones((I,J,K,G))
    #Q[:,:,:,1]=0
    Sigmarg=np.zeros((I,J,K,G))
    
    x = np.linspace(hx*.5,Nx-hx*.5,I)
    y = np.linspace(hy*.5,Ny-hy*.5,J)
    z = np.linspace(hz*.5,Nz-hz*.5,K)
    
    BCGs = np.ones((6,3,G))
    BCGs[:,0,:] = 0
    BCGs[:,2,:] = 0
    BCGs[1,0,:]=0.25
    BCGs[1,1,:]=0.50
    BCGs[3,0,:]=0.25
    BCGs[3,1,:]=0.5
    
    ihx2,ihy2,ihz2 = (1.0/hx**2,1.0/hy**2,1.0/hz**2)
    X_Sections = np.genfromtxt('Xsections2.csv', delimiter=",") #Get our Cross Sections
    for k in range(K):
        for j in range(J):
            for i in range(I):
                assy=int(Geometry[J-j,i])
                Sigmaag[i,j,k,(0,1)]=[X_Sections[assy,3],X_Sections[assy,8]]
                nuSigmafg[i,j,k,(0,1)]=[X_Sections[assy,4],X_Sections[assy,9]]
                chig[i,j,k,(0,1)] = [1.0,0.0]
                D[i,j,k,(0,1)]=[X_Sections[assy,2],X_Sections[assy,7]]
                Sigmasgg[i,j,k,0,(0,1)]=[0,X_Sections[assy,5]]
                Sigmasgg[i,j,k,1,(0,1)]=[X_Sections[assy,10],0]
                #Can define Q here if you want
                Sigmarg[i,j,k,(0,1)]=[X_Sections[assy,3]+X_Sections[assy,5],X_Sections[assy,8]+X_Sections[assy,10]]
    return   Power,hx,hy,hz,ihx2,ihy2,ihz2,x,y,z,Sigmarg,Sigmasgg,nuSigmafg,nug,chig,D,Q,L,I,J,K,Nx,Ny,Nz,G,BCGs
def inner_iteration(G,I,J,K,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCGs,Sigmarg,Sigmasgg,D,Q,L,lintol=1.0e-8,grouptol=1.0e-6,maxits = 20,LOUD=False):
    iteration = 1
    converged = False
    phig = np.zeros((I,J,K,G))
    while not(converged):
        phiold = phig.copy()
        for g in range(G):
            #compute Qhat
            Qhat = Q[:,:,:,g].copy()
            for gprime in range(0,G):
                if (g != gprime):
                    Qhat +=   Sigmasgg[:,:,:,gprime,g]*phig[:,:,:,gprime]
            phi0 = diffusion_steady_fixed_source(I,J,K,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCGs[:,:,g],D[:,:,:,g],Sigmarg[:,:,:,g],Qhat[:,:,:],L,lintol)
            #if(g==1): print(phi0)
            phig[:,:,:,g] = phi0.copy()
        change = np.linalg.norm(np.reshape(phig - phiold,I*J*K*G)/(I*J*K*G))
        if LOUD:
            print("Iteration",iteration,"Change =",change)
        iteration += 1
        converged = (change < grouptol) or iteration > maxits
    return phig
def kproblem_mg_diffusion(x,y,I,J,K,G,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCGs,Sigmarg,Sigmasgg,nuSigmafg,chig,D,L,lintol=1.0e-8,grouptol=1.0e-5,tol=1.0e-6,maxits = 3, k = 1, LOUD=True):
    phi0=np.random.rand(I,J,K,G)
    for g in range(G): #We are going to assume axial heights are the same between runs and number of groups
        for k in range(K):
            if(os.path.isfile('phi_out_'+str(g)+'_'+str(z[k])+'.txt')):
                phi_out_g_k=np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")
                x_out=np.genfromtxt('x.txt',delimiter=",")
                y_out=np.genfromtxt('y.txt',delimiter=",")
                phi_interp_g_k = interpolate.interp2d(x_out,y_out,phi_out_g_k,kind='linear',bounds_error=False,fill_value=phi_out_g_k[-1,1])
                phi0[:,:,k,g]=phi_interp_g_k(x,y)
                phi_out_g_k=None
                x_out=None
                y_out=None
                phi_interp_g_k=None
    phi0 = phi0 / np.linalg.norm(np.reshape(phi0,I*J*K*G))
    phiold = phi0.copy()
    converged = False
    iteration = 1
    if(os.path.isfile('k_effective_out.txt')):
        k_old_file=np.genfromtxt('k_effective_out.txt',delimiter=",")
        k=k_old_file[len(k_old_file[:])-1]
        os.remove('k_effective_out.txt')
    while not(converged):
        Qhat = chig*0
        for g in range(G):
            for gprime in range(G):
                Qhat[:,:,:,g] += chig[:,:,:,g]*nuSigmafg[:,:,:,gprime]*phi0[:,:,:,gprime]
        phi0 = inner_iteration(G,I,J,K,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCGs,Sigmarg,Sigmasgg,D,Qhat,L)
        knew = np.linalg.norm(np.reshape(phi0,I*J*K*G))/np.linalg.norm(np.reshape(phiold,I*J*K*G))
        solnorm = np.linalg.norm(np.reshape(phiold,I*J*K*G))
        if (LOUD):
            print("================================")
            print("Iteration",iteration,": k =",knew)
        converged = ((np.abs(knew-k) < tol) or (iteration > maxits))
        with open("k_effective_out.txt","a") as myfile:
            myfile.write(str(knew)+"\n")
        k = knew
        phi0 /= k
        phiold = phi0.copy()
        #if (LOUD):
        #    print("================================")
        #    print("Iteration",iteration,": k =",k)
        iteration += 1
    k = knew
    phi0 /= k
    return k,iteration-1,phi0
def Power_find(I,J,K,Power):
	check=0
	for i in range(I):
		if(Power[len(Power[:,1])-1,0]!=Power[len(Power[:,1])-1,i] and check==0):
			check=1
			iswitch=i
	check=0
	for j in range(J):
		if(Power[len(Power[:,1])-1,0]!=Power[J-j,0] and check==0):
			check=1
			jswitch=j

	#print (iswitch)
	#print (jswitch)

	Powermat=np.zeros(int(Power[1:,:].max()))
	for assy in range(int(Power[1:,:].max())):
		for k in range(K):
			for j in range(J):
				for i in range(I):
					if(assy+1==int(Power[J-j,i])):
						for g in range(G):
							if(assy==0):
								Powermat[assy] +=4*hx*hy*hz*200*phig[i,j,k,g]*nuSigmafg[i,j,k,g] #Assuming only thermal fissions but could loop over groups
							elif(i<iswitch):
								Powermat[assy] +=2*hx*hy*hz*200*phig[i,j,k,g]*nuSigmafg[i,j,k,g] #Assuming only thermal fissions but could loop over groups
							elif(j<jswitch):
								Powermat[assy] +=2*hx*hy*hz*200*phig[i,j,k,g]*nuSigmafg[i,j,k,g] #Assuming only thermal fissions but could loop over groups
							else:
								Powermat[assy] +=hx*hy*hz*200*phig[i,j,k,g]*nuSigmafg[i,j,k,g] #Assuming only thermal fissions but could loop over groups
	Powermap=np.zeros((I,J,K))
	for assy in range(int(Power[1:,:].max())):
		for k in range(K):
			for j in range(J):
				for i in range(I):
					if(assy+1==int(Power[J-j,i])): Powermap[i,j,k]=Powermat[assy] #Assuming only thermal fissions but could loop over groups
	return Powermap
def plotting_printing(phig,Powermap,G,I,J,K,x,y,z):
	for g in range(G): #We are going to assume axial heights are the same between runs
		for k in range(K):
			if(os.path.isfile('phi_out_'+str(g)+'_'+str(z[k])+'.txt')):
				if(len(np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")[:,1])<=J and len(np.genfromtxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',delimiter=",")[1,:])<=I):
					os.remove('phi_out_'+str(g)+'_'+str(z[k])+'.txt')
					os.remove('x.txt')
					os.remove('y.txt')
					np.savetxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',phig[:,:,k,g],delimiter=',')
					np.savetxt('x.txt',x,delimiter=',')
					np.savetxt('y.txt',y,delimiter=',')
			else:
				np.savetxt('phi_out_'+str(g)+'_'+str(z[k])+'.txt',phig[:,:,k,g],delimiter=',')
				np.savetxt('x.txt',x,delimiter=',')
				np.savetxt('y.txt',y,delimiter=',')
	
	
	
	if (sys.argv[1]=="t"):
		import matplotlib.pyplot as plt
		plt.pcolor(np.transpose(np.log10(phig[:,:,0,1])),rasterized=True); plt.colorbar();
		plt.clim(-6,np.log10(phig[:,:,0,1]).max())
		plt.savefig("Thermal.pdf")
		plt.clf()
		plt.pcolor(np.transpose(np.log10(phig[:,:,0,0])),rasterized=True); plt.colorbar();
		plt.clim(-6,np.log10(phig[:,:,0,1]).max())
		plt.savefig("Fast.pdf")
		plt.clf()

	if (sys.argv[2]=="t"):
		import matplotlib.pyplot as plt
		plt.pcolor(np.transpose((Powermap[:,:,0])),rasterized=True); plt.colorbar();
		plt.clim(Powermap[:,:,0].min(),Powermap[:,:,0].max())
		plt.savefig("Powermap.pdf")
		plt.clf()


#Define the Problem
Power,hx,hy,hz,ihx2,ihy2,ihz2,x,y,z,Sigmarg,Sigmasgg,nuSigmafg,nug,chig,D,Q,L,I,J,K,Nx,Ny,Nz,G,BCGs = lattice2G()
#k Convergence 
k,iterations,phig = kproblem_mg_diffusion(x,y,I,J,K,G,Nx,Ny,Nz,hx,hy,hz,ihx2,ihy2,ihz2,BCGs,Sigmarg,Sigmasgg,nuSigmafg,chig,D,L,lintol=1.0e-13,grouptol=1.0e-5,tol=1.0e-6,maxits = 100, k = 1, LOUD=True)
#Determine Assembly Power Profile 
Powermap=Power_find(I,J,K,Power)
#End the Run Time
print(str(time.time() - start_time)+"\n")
with open("time.txt","a") as myfile:
    myfile.write(str(time.time() - start_time)+"\n")
#Make the plots
plotting_printing(phig,Powermap,G,I,J,K,x,y,z)


