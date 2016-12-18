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
from math import factorial as fact 
import copy
from statistics import mean
from statistics import variance
from scipy.special import expn #Exponential Integral
from scipy.special import la_roots #Laguerre Roots
from scipy.special import assoc_laguerre as PSolve #Laguerre Solve
from scipy.special import gamma  as gammaf #Gamma Function
from scipy import interpolate
from scipy import integrate
from scipy.integrate import trapz
from scipy.stats import gamma as gammad #Gamma distribution

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]





################################################################
######################### Functions ############################
################################################################

#Needed for Van Sampling
def vdc(n, base=2):
    """
    Generates van der Corput sequence function
    """
    vdc, denom = 0,1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

#Van Sampling
def Rvdc(N,base=2):
    """
    Will generate a vector N long of the van
    der Corput sequence with a chosen base
    """
    RN=[]
    for i in range(0,N):
        RN.append(vdc(i+1,base))
    return(RN)

def HIST(Xlabel,Samples,Nbins,N):
    """
    Plot histogram
    """
    #Plot histogram
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(Xlabel,fontsize=16)
    #For normalization of histogram
    weights=np.ones_like(Samples)/(len(Samples))
    ax.set_ylabel('Normalized counts with tot N = '+str(N),fontsize=18)
    #Get data from histogram and plot
    n,bins,patches=ax.hist(Samples,Nbins,
            weights=weights,color='green',alpha=0.7,edgecolor='black')
    return(n,bins,ax,fig)

def HISTDataToPDF(n,bins,ax,fig):
    
    Probability=n #Probability of being in a particular bin
                  #Make sure to have used the weights when getting n
    XValues=[]
    PDF=[]   #Probability density function (different than Prob)
             #Divide probability by bin width
    for i in range(0,len(n)):
        XValues.append((bins[i]+bins[i+1])/2)
        PDF.append(Probability[i]/(bins[i+1]-bins[i]))
        
    #Fit the data to a function
    
    #First set a zero value (this is assuming the PDF is non negative
    #and is increasing towards zero)
    #m=(PDF[1]-PDF[0])/(XValues[1]-XValues[0]) #slope at start
    #b=PDF[0]-m*XValues[0]  #y intercept
    #print(b,PDF[0],PDF[1])
    
    #Append the zero value, assuming first two bins slope are constant
    #XValues.insert(0,0)
    #PDF.insert(0,b)

    #Create a PDF function
    I=interpolate.interp1d(XValues,PDF,
                           fill_value=max(Probability),bounds_error=False)
    
    #Evaluate the function and see how well it
    #fits the data by plotting it
    x=np.linspace(min(XValues),max(XValues),10000)
    y=I(x)
    ax.plot(x,y,'r',linewidth=2.0)
    
    #does the pdf integrate to 1?
    print("Your PDF integrates to: "+str(integrate.trapz(I(x),x)))
    return(ax,fig)

def ExpIntEval(x):
    """
    Evaluate Exponential Integral at n=2
    """
    try:
        if len(x)==1:
            E2=expn(2,x[0])
            phi=float(E2)
            return(phi)
        else:
            phi=[]
            for i in range(0,len(x)):
                E2=expn(2,x[i])
                phi.append(float(E2))
            return(phi)
    except TypeError:
        E2=expn(2,x)
        phi=float(E2)
        return(phi)

def PolyChaos(cn,alpha,x):
    """
    The name...chaos, I like it
    """
    Sum=0
    for n in range(0,len(cn)):
        Sum=Sum+cn[n]*PSolve(x,n,k=alpha)
    return(Sum)

def Fnear(x):
    #return(np.cos(x))
    return(ExpIntEval(x))

def Determine_cn(n,alpha,beta,x):
    """
    This function will evaluate cn constants for 
    estimation of a function, 'Fnear'.
    """
    diff=5;tol=0.000001;nprime=1;IntOld=1000
    Cprefix=fact(n)/(gammaf(n+alpha+1))
    while diff>tol:
        Vals=la_roots(nprime,alpha) #Roots in [0], weights[1]
        # f(z) as defined in quad rule
        fRoots=Fnear((Vals[0]*x)/beta)*PSolve(Vals[0],n,k=alpha)
        # Evaluate integral for the quad rule
        Int=sum(Vals[1]*fRoots)
        # See how different the integral is with one less n
        diff=abs(Int-IntOld)
        IntOld=copy.copy(Int)

        nprime=nprime+1
        if nprime==1000:
            print("Did not converge on quadrature")
            quit()

    #Return the cn value (prefix * Integral)
    return(Cprefix*Int)

def Print(Method,List):
    if len(List)>2:
        print1=str(mean(List))
        print2=str(variance(List))
    else:
        print1=str(List[0])
        print2=str(List[1])
    print(Method+" calculation mean is "+print1)
    print(Method+" calculation variance is "+print2)
    print("")


def Plot(x,y,Variance,filename,scale):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x,y,'k--',linewidth=2.0)
    
    ax.set_xlabel('x',fontsize=18)
    ax.set_ylabel('Mean Flux',fontsize=18)
    #ax.yaxis.labelpad=55
    ax.set_yscale(scale)
    
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    ax.grid(alpha=0.8,color='black',linestyle='dotted')
    ax.grid(alpha=0.8,color='black',linestyle='dotted')


    plt.savefig(filename)
    #return(fig,ax)
