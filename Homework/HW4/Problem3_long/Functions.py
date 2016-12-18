#!/usr/bin/env python3

"""
FractionAM converts atom fractions to mass fractions
and mass fractions to atom fractions. Input is a 
single string with MCNP style fractions.
"""

__author__     =  "Paul Mendoza"
__copyright__  =  "Copyright 2016, Planet Earth"
__credits__    = ["Sunil Chirayath",
                  "Charles Folden",
                  "Jeremy Conlin"]
__license__    =  "GPL"
__version__    =  "1.0.1"
__maintainer__ =  "Paul Mendoza"
__email__      =  "paul.m.mendoza@gmail.com"
__status__     =  "Production"

################################################################
##################### Import packages ##########################
################################################################

import numpy as np
from scipy.stats import beta
from scipy import interpolate
from scipy import integrate
from scipy.integrate import trapz

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


################################################################
######################### Functions ############################
################################################################


def BuildA(v,D,w,dt,dx,NX):
    #i-1
    a=(-v/dx-D/pow(dx,2))
    #i
    b=(1/dt)+(v/dx)+2*D/pow(dx,2)+w
    #i+1
    c=(-D/pow(dx,2))
    
    A = np.zeros([NX,NX])
  
    for i in range(0,NX):
        if (i == 0):
            A[i,i] = b
            A[i,1] = c
        elif(i==(NX-1)):
            A[i,i-1] = a
            A[i,i] = b
        else:  
            A[i,i-1] = a
            A[i,i] = b
            A[i,i+1] = c

    #Boundary conditions
    A[0,NX-1] = A[0,0]
    A[NX-1,0] = A[NX-1,NX-1]
    return(A)

def FindIndex(list,item):
    idx = (np.abs(arr - v)).argmin()


def Integrate(u,xmin,xmax,x,t):

    x_low=np.abs(x-xmin).argmin()
    x_high=np.abs(x-xmax).argmin()

    
    r=u[x_low:x_high+1,:]

    IntegratedOverTime=np.zeros([x_high+1-x_low])

    
    for i in range(0,x_high+1-x_low):
        IntegratedOverTime[i]=np.trapz(r[i,:],t)

    IntTimeSpace=np.trapz(IntegratedOverTime,x[x_low:x_high+1])
    
    return(IntTimeSpace)

def SolveQoI(NX,Nt,x,t,A,dt,w):
    #Build Matrix with solutions
    u=np.zeros([NX,Nt])
 
    for i in range(0,NX):
        if x[i]<=2.5:
            u[i,0]=1
 
    #Solve the system
    for i in range(1,len(t)):
        u[:,i] = np.linalg.solve(A,u[:,i-1]*(1/dt))

    
    #Integrate the system
    Integral=Integrate(u,5,6,x,t)
    RXNRate=Integral*w
    return(RXNRate)

    
def Plot(x,y,Label,fig,ax,ColorStyle,Varied,Scaled):
    

    ax.plot(x,y,ColorStyle,linewidth=2.0,label=Label,
            markersize=8)

    if Scaled:
        yLabel=r'$\text{Scaled Sensitivity Coeff}|_i=\left.\mu_i\frac{\delta\text{QoI}}{\delta\theta_i}\right|_{\bar{\theta}}$'
    else:
        yLabel=r'$\text{Sensitivity Index}|_i=\left.\sigma_i\frac{\delta\text{QoI}}{\delta\theta_i}\right|_{\bar{\theta}}$'
        
    ax.set_xlabel(r'$\Delta$'+Varied,fontsize=18)
    ax.set_ylabel(yLabel,fontsize=16)
    #ax.yaxis.labelpad=55
    
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    ax.grid(alpha=0.8,color='black',linestyle='dotted')
    ax.grid(alpha=0.8,color='black',linestyle='dotted')
    
    return(fig,ax)
