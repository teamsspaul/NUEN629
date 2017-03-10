#!/usr/bin/env python3

################################################################
##################### Import packages ##########################
################################################################

import sys
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import scipy.linalg as scil
import scipy.special as sps
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import random as rn
import matplotlib.mlab as mlab
import copy
import os
import pandas as pd
import os.path

#############################################################
######################### Variables #########################
#############################################################

# Basic information
FigureSize = (11, 6)              # Dimensions of the figure
TypeOfFamily='monospace'          # This sets the type of font for text
font = {'family' : TypeOfFamily}  # This sets the type of font for text
LegendFontSize = 12
Lfont = {'family' : TypeOfFamily}  # This sets up legend font
Lfont['size']=LegendFontSize

Title = ''
TitleFontSize = 22
TitleFontWeight = "bold"  # "bold" or "normal"

#Xlabel='E (eV)'   # X label
XFontSize=18          # X label font size
XFontWeight="normal"  # "bold" or "normal"
XScale="linear"       # 'linear' or 'log'

YFontSize=18                    # Y label font size
YFontWeight="normal"            # "bold" or "normal"
YScale="linear"                 # 'linear' or 'log'

Check=0


Colors=["aqua","gray","red","blue","black",
                "green","magenta","indigo","lime","peru","steelblue",
                "darkorange","salmon","yellow","lime","black"]

# If you want to highlight a specific item
# set its alpha value =1 and all others to 0.4
# You can also change the MarkSize (or just use the highlight option below)
Alpha_Value=[1  ,1  ,1  ,1  ,1  ,1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
MarkSize=   [8  ,8  ,8  ,8  ,8  ,8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8]

Linewidth=[1  ,1  ,1  ,1  ,1  ,1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

# Can change all these to "." or "" for nothing "x" isn't that good
MarkerType=["8","s","p","D","*","H","h","d","^",">"]

# LineStyles=["solid","dashed","dash_dot","dotted","."]
LineStyles=["solid"]

SquishGraph = 0.75
BBOXX = 1.24
BBOXY = 0.5       # Set legend on right side of graph

NumberOfLegendColumns=1

#Xlabel='Time [days]'
Xlabel='Time [years]'
Ylabel="Mass $\\left[\\frac{g}{\\text{tHM}}\\right]$"

XlabelBurn='Burnup $\\left[\\frac{\\text{MWd}}{\\text{tHM}}\\right]$'
YlabelBurn='$^{239}$Pu Mass Fraction'

Na=6.0221409E23


################################################################
################### Functions for building #####################
################### Lists                  #####################
################################################################

def Returnfloat(string):
    """
    string has format   238.023249814(23) 
            or format   [15.99903-15.99977]
            or format   235.04+/-0.0000019

    Returns just the number, no uncertainties
    """
    if "(" in string:
        Number=str(string.split('(')[0])
        LastErrorNumber=str(string.split("(")[1].replace(")",""))
        NumberOfZeros=len(Number.split(".")[1])-len(LastErrorNumber)
        Error="0."
        for i in range(0,NumberOfZeros):
            Error=Error+"0"
        Error=Error+LastErrorNumber
    elif "[" in string:
        FirstNum=float(string.split('-')[0].replace("[",''))
        SecondNum=float(string.split('-')[1].replace(']',''))
        Number=str((FirstNum+SecondNum)/2)
        Error=str(float(Number)-FirstNum)
    elif "+/-" in string:
        Number=string.split("+/-")[0]
        Error=string.split("+/-")[1]
        
    return(float(Number))

def Isotopes(ToAdd):
  """
  This function will create a dictionary 'Nuclides'
  with nuclides found in tape9.inp, excluding activation isotopes
  """
  Nuclides={}
  Nuclide_Names=()

  with open('Data/tape9.inp') as f:
    TAPE9Content=f.readlines()

  count=0
  for line in TAPE9Content:
    hold=line.split()

    #No activation products or the -1 between libraries
    if not '1' in hold[0] and not '601' in hold[0] and "-" not in hold[0]:

      #No repeats                 #No decimals              #No text
      if hold[1] not in Nuclides and "." not in hold[1] and hold[1].isdigit():
        #Filter out lower mass isotopes
        if len(hold[1])==6:
          Nuclides[hold[1]]=count
          Nuclide_Names=Nuclide_Names+(hold[1],)
          count=count+1

  for isotope in ToAdd:
      Nuclides[isotope]=count
      Nuclide_Names=Nuclide_Names+(isotope,)
      count=count+1
      
  return(Nuclides,Nuclide_Names)

def GatherDecay(Nuclide_Names):
  """
  This function will create an array 'Decay_Consts'
  that has all the half-life information for our system.
  """

  Decay_Consts = np.zeros(len(Nuclide_Names))

  
  with open('Data/tape9.inp') as f:
    TAPE9Content=f.readlines()

  for i in range(0,len(Nuclide_Names)):
      Nuclide=Nuclide_Names[i]
      for line in TAPE9Content:
          hold=line.split()
          
          #Look for half life information, and decay type information
          #No activation products or the -1 between libraries
          if hold[0]=='2' or hold[0]=='3': 
              if hold[1] == Nuclide:
                  Thalf=float(hold[3])
                  if hold[2]=='1':  #seconds
                      const=np.log(2)/Thalf
                  elif hold[2]=='2':  #minutes
                      const=np.log(2)/(Thalf*60)
                  elif hold[2]=='3': #hours
                      const=np.log(2)/(Thalf*60*60)
                  elif hold[2]=='4': #days
                      const=np.log(2)/(Thalf*60*60*24)
                  elif hold[2]=='5': #years
                      const=np.log(2)/(Thalf*60*60*24*365.25)
                  elif hold[2]=='6': #Stable
                      const=-1
                  elif hold[2]=='7':
                      const=np.log(2)/(Thalf*60*60*24*365.25*10**3)
                  elif hold[2]=='8':
                      const=np.log(2)/(Thalf*60*60*24*365.25*10**6)
                  elif hold[2]=='9':
                      const=np.log(2)/(Thalf*60*60*24*365.25*10**9)
                  else:
                      print("could not find a proper halflife")
                      print(line)
                      quit()
                  Decay_Consts[i]=const
        
  return(Decay_Consts)



def FindAtomicMass(df,proton,Isotope):
    """
    This function will take in a dataset 'df' look through the
    'df.Protons' column and find the column that matches with 
    'proton'. If the row that contains 'proton' also contains
    'Isotope' in the 'df.Isotope' column, then the value stored
    in 'df.Relative_Atomic_Mass' is reported for that row.
    Because the proton numbering scheme can have a format
    '10' for hydrogen and '10' for neon (following MCNP ZAID 
    naming conventions) if we don't find a value with the whole
    string of 'proton' then the program looks through the first
    element of string and tries to match that 'proton[0]'
    If no matches are found, and error is thrown out.

    df = dataset with columns 'Protons' 'Isotopes' and 
    'Relative_Atomic_Mass'. Dataset created with pandas

    proton = string with proton number (follow MCNP zaid format)

    Isotope = string with isotope number (just put the atomic mass
    do not follow MCNP format - different for few cases)
    """
    #print(df)
    for i in range(0,len(df.Protons)):
        dfPro=str(df.Protons[i])
        if proton==dfPro:
            dfIso=str(df.Isotope[i])
            if Isotope==dfIso:
                Mass=df.Relative_Atomic_Mass[i]
                break
    try:
        Mass
    except NameError:
        for i in range(0,len(df.Protons)):
            dfPro=str(df.Protons[i])
            if proton[0]==dfPro:
                dfIso=str(df.Isotope[i])
                if Isotope==dfIso:
                    Mass=df.Relative_Atomic_Mass[i]
                    break
    try:
        Mass
    except NameError:
        print("Could not find atomic mass for proton = "\
              +proton+" and for Isotope = "+Isotope)
        Mass='10000.09(23)'
    Mass=Returnfloat(Mass)
    return(Mass)

def GatherMasses(df,Nuclides):
  """
  Make numpy array of masses which correspond to isotopes in Nuclides
  Nuclides is a dictionary. Where each key is a zaid number
  """
  
  Atom_Mass=np.zeros(len(Nuclides))
  for key,value in Nuclides.items():
    if not len(key)==6:
      print("Did not filter out the lower mass isotopes, quitting")
      quit()
    proton=key[0:2]
    if key[2]=="0":
        Isotope=key[3:5]
    elif key[4]=="0" and float(proton)*3<100:
        Isotope=key[2:4]
    elif key[2]!="0":
        Isotope=key[2:5]

    Mass=FindAtomicMass(df,proton,Isotope)
    if Mass>9000:
        print("Mass is over 9,000!!!")
        print(key)
        quit()
    Atom_Mass[value]=Mass
  return(Atom_Mass)

################################################################
####### Functions for solving the system Ax=b (kind of) ########
################################################################

def MatExp(A,n0,t,maxits,tolerance=1e-12,LOUD=False):

  converged = False
  m=0
  sum_old=n0.copy()*0
  
  while not(converged):

    if m==0:
        APowerm=np.identity(A.shape[0])
        Factorial=1
    else:
        APowerm=np.dot(APowerm,A)
        Factorial=Factorial*m
    Sum=sum_old+(1/Factorial)*np.dot((APowerm)*(t**m),n0)
    
    #Avoid dividing by zero
    if sum(Sum)==0: m+=1;sum_old=Sum.copy();continue
    
    change = np.linalg.norm(Sum-sum_old)/np.linalg.norm(Sum)    
    converged = (change < tolerance) or (m > maxits)
    
    if (LOUD>0) or (converged and LOUD<0):
      print("Iteration",m,": Relative Change =",change)
    if (m > maxits):
      print("Warning: Source Iteration did not converge : "+\
            " m : "+str(m)+", Diff : %.2e" % change)
    #Prepare for next iteration
    m += 1
    sum_old = Sum.copy()

  return(Sum)

def BackEuler(A,no,dt):
    I=np.identity(A.shape[0])
    return(np.dot(np.linalg.inv(I-A*dt),no))

def DeterminePolesNResidues(n):
    """
    This program takes the algorithm from the reference
    and converts to a python script...I know its janky
    but it works
    """
    def Append(List1,List2):
        for item in List2:
            for item2 in item:
                List1=np.append(List1,item2)
        return(List1)
    def absG(List):
        List2=[]
        for item in List:
            if abs(item)>1:
                List2.append(item)
        return(List2)
    #function [zk,ck] = cf(n);
    K = 75;                             # no of Cheb coeffs
    nf = 1024;                          # no of pts for FFT
    #Roots correct?
    roots=np.arange(0,nf,1)/nf
    #w = np.exp(2i*pi*(0:nf-1)/nf);     # roots of unity

    w=np.exp(2j*np.pi*roots)
    t = np.real(w);                     # Cheb pts (twice over)
    scl = 9;                            # scale factor for stability
    #F = np.exp(scl*(t-1)./(t+1+1e-16)); # exp(x) transpl. to [-1,1]
    F = np.exp(scl*(t-1)/(t+1+1e-16)); # exp(x) transpl. to [-1,1]
    c = np.real(np.fft.fft(F))/nf;      # Cheb coeffs of F
    index=reversed(np.arange(1,K+2,1))
    partofc=[]
    for i in index:
        partofc.append(c[i-1])
    #f = np.polyval(c(K+1:-1:1),w);      # analytic part f of F
    f = np.polyval(partofc,w);      # analytic part f of F

    #[U,S,V] = svd(hankel(c(2:K+1)));    # SVD of Hankel matrix
    hankie=scil.hankel(c[1:K+1])
    U,S,V=np.linalg.svd(hankie,full_matrices=False)

    #s = S(n+1,n+1);                     # singular value
    s=S[n]
    #u = U(K:-1:1,n+1); v = V(:,n+1);  # singular vector
    u=[]
    index=reversed(np.arange(0,K,1))
    for i in index:
        u.append(U[i,n])
    #v=np.array(V[:,n].copy())
    v=np.array(V[n,:].copy())
    #zz = zeros(1,nf-K);                 # zeros for padding
    zz=np.zeros([1,nf-K])
    #b = fft([u zz])./fft([v zz]);       # finite Blaschke product
    b=np.fft.fft(Append(u,zz))/np.fft.fft(Append(v,zz))
    #rt = f-s*w.^K.*b;                   # extended function r-tilde
    rt=f-s*(w**K)*b;
    #rtc = real(fft(rt))/nf;             # its Laurent coeffs
    rtc=np.real(np.fft.fft(rt))/nf;
    #zr = roots(v); qk = zr(abs(zr)>1);  # poles
    zr=np.roots(v);qk=np.array(absG(zr));
    #qc = poly(qk);                      # coeffs of denominator
    qc=np.poly(qk);
    #pt = rt.*polyval(qc,w);             # numerator
    pt=rt*np.polyval(qc,w);
    #ptc = real(fft(pt)/nf);             # coeffs of numerator
    ptc=np.real(np.fft.fft(pt)/nf);
    #ptc = ptc(n+1:-1:1); ck = 0*qk;
    index=reversed(np.arange(0,n+1,1))
    ptc2=[]
    for i in index:  #Can I just reversed ptc?
        ptc2.append(ptc[i])
    ptc=ptc2.copy()
    ck=0*qk
    #N+1?
    #for k =1:n              # calculate residues
    #    q = qk(k); q2 = poly(qk(qk~=q));
    #    ck(k) = polyval(ptc,q)/polyval(q2,q);
    for k in range(0,n):
        if len(qk)==k:
            print("we are short a qk")
            continue
        q=qk[k];
        q2=[];
        for item in qk:
            if not q==item:
                q2.append(item)
        q2=np.poly(q2);
        ck[k]=np.polyval(ptc,q)/np.polyval(q2,q)
    #zk = scl*(qk-1).^2./(qk+1).^2;      # poles in z-plane
    zk=scl*((qk-1)**2)/((qk+1)**2)
    #ck = 4*ck.*zk./(qk.^2-1);           # residues in z-plane
    ck=4*ck*zk/(qk**2-1)
    #Cut down ck and zk to half the original points
    ck2=[];zk2=[]
    for i in range(0,len(ck)):
        if i % 2 == 0:
            ck2=np.append(ck2,ck[i])
            zk2=np.append(zk2,zk[i])
    
    return(ck2,zk2)

def RationalPrep(N,Phi):
    """Calculate constants for a rational approximation
    Inputs:
    N:               Number of Quadrature points
    Phi:             'Parabola',
                     'Cotangent', or
                     'Hyperbola' (shape of Phi)

    Outputs:
    ck:              First set of constants for approximation
    zk:             Second set of constants for approximation
    """
    theta=np.pi*np.arange(1,N,2)/N
    if Phi=='Parabola':
        zk=N*(0.1309-0.1194*theta**2+0.2500j*theta)
        w=N*(-2*0.1194*theta+0.2500j)
    elif Phi=='Cotangent':
        cot=1/np.tan(0.6407*theta)
        ncsc=-0.6407/(np.sin(0.6407*theta)**2)
        zk=N*(0.5017*theta*cot-0.6122+0.2645j*theta)
        w=N*(0.2645j+0.5017*cot+0.5017*theta*ncsc)
    elif Phi=='Hyperbola':
        zk=2.246*N*(1-np.sin(1.1721-0.3443j*theta))
        w=2.246*N*(0.3443j*np.cos(1.1721-0.3443j*theta))
    elif Phi=='Best':
        ck,zk=DeterminePolesNResidues(N)
        return(ck,zk)
    else:
        print("Did not pick proper rational approximation dude")
        print("Quiting now")
        quit()
             
    ck=1.0j/N*np.exp(zk)*w
    return(ck,zk)

def RationalApprox(A,n0,t,N,ck,zk,tol=1e-12,maxits=2000):
    """
    Calculate the rational approximation solution for n(t)
    Inputs:
    A:          Matrix with system to be solved
    n0:         initial conditions of the system 
    t:          time at which solution is determined
    N:          Number of quadrature points (should be less than 20)
    ck:         constants for quadrature solution
    zk:         constants for quadrature solution
    tol:        Tolerence for convergence for GMRES
    maxits:     Maximium iterations for GMRES
    Outputs:
    nt:         Solution at time t
    """
    nt=np.zeros(len(n0))
    for k in range(int(N/2)):
        if len(n0)>1:
            #phi,code=spla.gmres(zk[k]*sparse.identity(len(n0))-A*t,n0,
            #                    tol=tol,maxiter=maxits)
            phi=np.dot(np.linalg.inv(zk[k]*np.identity(len(n0))-A*t),
                       n0)
            #if (code):
            #    print(code)
        else:
            phi=(zk[k]-A*t)**(-1)*n0
        nt=nt-2*np.real(ck[k]*phi)
    return(nt)

################################################################
################### Plotting Functions #########################
################################################################

def reduceList(List,N):
    List2=[List[0]]
    Div=int(len(List)/N)
    for i in range(1,len(List)-1):
        if i % Div == 0:
            List2.append(List[i])
    List2.append(List[-1])
    return(List2)

def PlotPoints(Ntot,Nplot):
    t=1
def loop_values(list1,index):
    """                                                                                                                                              
    This function will loop through values in list even if
    outside range (in the positive sense not negative)
    """
    while True:
        try:
            list1[index]
            break
        except IndexError:
            index=index-len(list1)
    return(list1[index])

def Legend(ax):
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc='best',
              fontsize=LegendFontSize,prop=font)
    return(ax)
def Legend2(ax):
    handles,labels=ax.get_legend_handles_labels()
    ax.legend(handles,labels,loc='upper left',
              fontsize=LegendFontSize,prop=font)
    return(ax)

# def Legend(ax):
#         handles,labels=ax.get_legend_handles_labels()
#         box=ax.get_position()
#         ax.set_position([box.x0, box.y0, box.width*SquishGraph,
#                          box.height])
#         ax.legend(handles,labels,loc='center',
#                   bbox_to_anchor=(BBOXX,BBOXY),
#                   fontsize=LegendFontSize,prop=font,
#                   ncol=NumberOfLegendColumns)
#         return(ax)

def InList(item2,List):
  TF=False
  for item1 in List:
    if item1 == item2:
      TF=True
  if not TF:
    print("Invalid selection for plotting")
    print("Shuting down")
    quit()

def plot(df,Plotting,Name,NumOfPoints):
    #Plot In grams
    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)

    List=list(df.columns.values)
    x=df[List[0]].values[2:]

    Check=0
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:])/Na)*df[Item].values[0]
      if len(x)>NumOfPoints:
        x=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
      ax.plot(x,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=Item)
      Check=Check+1


    #Log or linear scale?
    ax.set_xscale(XScale)
    ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    ax.set_ylabel(Ylabel,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)
    plt.savefig("Plots/"+Name+'_grams.pdf')
    
    #Plot in Bq #################################

    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)

    List=list(df.columns.values)
    x=df[List[0]].values[2:]

    Check=0;Sum=np.zeros(len(x))
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:]))*df[Item].values[1]
      Sum=Sum+y
      if len(x)>NumOfPoints:
        xP=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
      else:
        xP=x.copy()
      ax.plot(xP,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=Item)
      Check=Check+1
    if len(x)>NumOfPoints:
      Sum=reduceList(Sum,NumOfPoints)
    ax.plot(xP,Sum,
            linestyle=loop_values(LineStyles,Check),
            marker=loop_values(MarkerType,Check),
            color=loop_values(Colors,Check),
            markersize=loop_values(MarkSize,Check),
            alpha=loop_values(Alpha_Value,Check),
            label="Sum")
    
    #Log or linear scale?
    ax.set_xscale(XScale)
    if sum(Sum)==0:
      ax.set_yscale('linear')
    else:
      ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    YlabelBq="Activity $\\left[\\frac{Bq}{\\text{tHM}}\\right]$"
    ax.set_ylabel(YlabelBq,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)
    plt.savefig("Plots/"+Name+'_Bq.pdf') 

def PUNAME(Item):
    if Item=="942390":
        NAME="$^{239}$Pu"
    elif Item=="942380":
        NAME="$^{238}$Pu"
    elif Item=="942400":
        NAME="$^{240}$Pu"
    elif Item=="942410":
        NAME="$^{241}$Pu"
    elif Item=="942420":
        NAME="$^{242}$Pu"
    else:
        print("No Name :(")
        quit()

    return(NAME)

def plotburnPuComp(df,Plotting,Name,NumOfPoints):
    #Plot In grams
    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)
    ax2=ax.twinx()
    List=list(df.columns.values)
    x=df[List[0]].values[2:]

    #print(Plotting)
    #quit()

          
    Cs137=((df['551370'].values[2:])/Na)*df['551370'].values[0]
    Burnup=Cs137*25.4627010527

    
    Check=0;Sum=np.zeros(len(x));Pu239=np.zeros(len(x));
    for Item in Plotting:
      NAME=PUNAME(Item)
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:])/Na)*df[Item].values[0]
      Sum=Sum+y
      if Item=="942390":
          Pu239=Pu239+y
      if len(x)>NumOfPoints:
          Burnup=reduceList(Burnup,NumOfPoints)
          x=reduceList(x,NumOfPoints)
          y=reduceList(y,NumOfPoints)
      ax.plot(Burnup,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=NAME)
      Check=Check+1

    #Log or linear scale?
    ax.set_xscale(XScale)
    ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    #ax.set_xlabel(Xlabel,
    #              fontsize=XFontSize,fontweight=XFontWeight,
    #              fontdict=font)
    ax.set_ylabel(Ylabel,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)


    Pu239Percent=Pu239/Sum
    if len(Burnup)>NumOfPoints:
        Burnup=reduceList(Burnup,NumOfPoints)
        Pu239Percent=reduceList(Pu239Percent,NumOfPoints)
    ax2.plot(Burnup,Pu239Percent,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label='$^{239}$Pu Mass Fraction')
    Check=Check+1


    #Log or linear scale?
    #ax.set_xscale(XScale)
    ax2.set_yscale(YScale)
    #Set Title
    #fig.suptitle(Title,fontsize=TitleFontSize,
    #             fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(XlabelBurn,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    ax2.set_ylabel(YlabelBurn,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend2(ax2)
    plt.savefig("Plots/"+Name+'_grams.pdf')
    
    # #Plot in Bq #################################

    # fig=plt.figure(figsize=FigureSize)  
    # ax=fig.add_subplot(111)

    # List=list(df.columns.values)
    # x=df[List[0]].values[2:]

    # for Item in Plotting:
    #   InList(Item,List) #Check if we have the isotope
    #   y=((df[Item].values[2:]))*df[Item].values[1]
    #   Sum=Sum+y
    #   if len(x)>NumOfPoints:
    #     xP=reduceList(x,NumOfPoints)
    #     y=reduceList(y,NumOfPoints)
    #   else:
    #     xP=x.copy()
    #   ax.plot(xP,y,
    #           linestyle=loop_values(LineStyles,Check),
    #           marker=loop_values(MarkerType,Check),
    #           color=loop_values(Colors,Check),
    #           markersize=loop_values(MarkSize,Check),
    #           alpha=loop_values(Alpha_Value,Check),
    #           label=Item)
    #   Check=Check+1
    # if len(x)>NumOfPoints:
    #   Sum=reduceList(Sum,NumOfPoints)
    # ax.plot(xP,Sum,
    #         linestyle=loop_values(LineStyles,Check),
    #         marker=loop_values(MarkerType,Check),
    #         color=loop_values(Colors,Check),
    #         markersize=loop_values(MarkSize,Check),
    #         alpha=loop_values(Alpha_Value,Check),
    #         label="Sum")
    
    # #Log or linear scale?
    # ax.set_xscale(XScale)
    # if sum(Sum)==0:
    #   ax.set_yscale('linear')
    # else:
    #   ax.set_yscale(YScale)
    # #Set Title
    # fig.suptitle(Title,fontsize=TitleFontSize,
    #              fontweight=TitleFontWeight,fontdict=font,ha='center')
    # #Set X and y labels
    # ax.set_xlabel(Xlabel,
    #               fontsize=XFontSize,fontweight=XFontWeight,
    #               fontdict=font)
    # YlabelBq="Activity $\\left[\\frac{Bq}{\\text{tHM}}\\right]$"
    # ax.set_ylabel(YlabelBq,
    #               fontsize=YFontSize,
    #               fontweight=YFontWeight,
    #               fontdict=font)

    # Legend(ax)
    # plt.savefig("Plots/"+Name+'_Bq.pdf') 

    
def plots2(df,df2,Plotting,Name,NumOfPoints,Method1,Method2):
    #Plot in grams
    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)

    List=list(df.columns.values)
    x=df[List[0]].values[2:]
    x2=df2[List[0]].values[2:]
    
    Check=0
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:])/Na)*df[Item].values[0]
      y2=((df2[Item].values[2:])/Na)*df2[Item].values[0]
      if len(x)>NumOfPoints:
        x=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
      if len(x2)>NumOfPoints:
        x2=reduceList(x2,NumOfPoints)
        y2=reduceList(y2,NumOfPoints)
      ax.plot(x,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method1)
      Check=Check+1
      ax.plot(x2,y2,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method2)
      Check=Check+1


    #Log or linear scale?
    ax.set_xscale(XScale)
    ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    ax.set_ylabel(Ylabel,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)
    plt.savefig("Plots/"+Name+'_Grams.pdf')


    #Plot in Bq
    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)

    List=list(df.columns.values)
    x=df[List[0]].values[2:]
    x2=df2[List[0]].values[2:]
    
    Check=0;Sum=np.zeros(len(x));Sum2=np.zeros(len(x2))
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:]))*df[Item].values[1]
      y2=((df2[Item].values[2:]))*df2[Item].values[1]
      Sum=Sum+y
      Sum2=Sum2+y2
      if len(x)>NumOfPoints:
        xP=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
      else:
        xP=x.copy()
      if len(x2)>NumOfPoints:
        xP2=redcueList(x2,NumOfPoints)
        y2=reduceList(y2,NumOfPoints)
      else:
        xP2=x2.copy()
      ax.plot(xP,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method1)
      Check=Check+1
      ax.plot(xP2,y2,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method2)
      Check=Check+1
    if len(x)>NumOfPoints:
      Sum=reduceList(Sum,NumOfPoints)
    if len(x2)>NumOfPoints:
      Sum2=reduceList(Sum2,NumOfPoints)
    ax.plot(xP,Sum,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label="Sum "+Method1)
    Check=Check+1
    ax.plot(xP2,Sum2,
            linestyle=loop_values(LineStyles,Check),
            marker=loop_values(MarkerType,Check),
            color=loop_values(Colors,Check),
            markersize=loop_values(MarkSize,Check),
            alpha=loop_values(Alpha_Value,Check),
            label="Sum "+Method2)

    #Log or linear scale?
    ax.set_xscale(XScale)
    if sum(Sum)==0:
      ax.set_yscale('linear')
    else:
      ax.set_yscale(YScale)
      
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    YlabelBq="Activity $\\left[\\frac{Bq}{\\text{tHM}}\\right]$"
    ax.set_ylabel(YlabelBq,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)
    plt.savefig("Plots/"+Name+'_Bq.pdf')

def ListToStr(List):
  Str=''
  for i in range(0,len(List)):
    if not i==len(List)-1:
      Str=Str+str(List[i])+","
    else:
      Str=Str+str(List[i])+"\n"
  return(Str)
    
def PrepFile(Name,n0,nuclide_names,atom_mass,decay_consts):
  File=open(Name,'w')
  File.write("Mass then Time (d),"+','.join(nuclide_names)+'\n')
  File.write("Masses,"+ListToStr(atom_mass)) #New line already included
  File.write("DecayConts,"+ListToStr(decay_consts))
  File.write("0,"+ListToStr(n0))
  return(File)

def Print(Method,nuclide,Results,Time,nuclides,atom_mass,nuclide_names):
  Index=nuclides[nuclide]
  MassConversion=atom_mass[Index]/Na
  string="Isotope "+nuclide_names[Index]+", Mass (g) = "
  Mass=Results[Index]*MassConversion
  Mass="%.4e" % Mass
  print(Method+" :",string,Mass,"Time=%.2f" % Time)


#####################################################################
################## for calculatin phi #############################
#####################################################################

def FindFissionXSection(Fissile_Isotopes):

    FissionXSections=np.zeros(len(Fissile_Isotopes))

    with open('Data/tape9.inp') as f:  #Save all X-section data to variable
        TAPE9Content=f.readlines()
        
    for i in range(0,len(Fissile_Isotopes)): #Loop through fissile isos
        parent=Fissile_Isotopes[i]
        
        for line in TAPE9Content: #Loop through x-section data
            hold=line.split()

            if '602' == hold[0] and hold[1]==parent:  #Find x-section
                FissionXSections[i]=hold[5]
                break
    return(FissionXSections)

def CalMevPerFiss(Fissile_Isotopes):
    """
    Given a list of fissile isotopes
    return a list of MeV/fission numbers
    calculated from an equation
    """
    MeVperFission=np.zeros(len(Fissile_Isotopes))
    for i in range(0,len(Fissile_Isotopes)):
        isotope=Fissile_Isotopes[i]
        proton=isotope[0:2]
        if isotope[2]=="0":
            Anum=isotope[3:5]
        elif isotope[4]=="0" and float(proton)*3<100:
            Anum=isotope[2:4]
        elif isotope[2]!="0":
            Anum=isotope[2:5]
        else:
            print("Missed logic in finding A number")
            quit()
        if int(Anum)<int(proton):
            print("Something is wrong, more protons than neutrons")
            print("Proton: ",proton,"A: ",Anum,"ZAID :",isotope)
            quit()
        MeVperFission[i]=1.29927*(10**-3)*(float(proton)**2)*(float(Anum)**0.5)+33.12
    return(MeVperFission)


def Calculatephi(FissionXSections,MeVperFission,n0,Power,Fissile_Isotopes,Nuclides):
    Sum=0
    for i in range(0,len(FissionXSections)):
        sigma=FissionXSections[i]*10**-24
        E=MeVperFission[i]
        N=n0[Nuclides[Fissile_Isotopes[i]]]
        Sum=Sum+sigma*E*N
    phi=(6.2414959617521E18*Power)/Sum
    return(phi)

def Makeb(Nuclides):

    b = np.zeros(len(Nuclides))
    b[Nuclides['922340']] = 6.94741E23
    b[Nuclides['922350']] = 7.6864E25
    b[Nuclides['922380']] = 2.4532E27
    return(b)

    

#####################################################################
################## For Building A and b #############################
#####################################################################

class DecayClass:
    def __init__(self):
        self.FBX    =  0. # 'The fraction of negatron beta decay transitions that results in
                          # in the daughter nuclide being in a relatively long-lived state'
                          # I think this should read the fraction of all decay events which are...
        self.IDFBX  = ''  # ZAID for daughter for FBX
        self.FPEC   =  0. # Fraction of all decay events which are positron or EC
        self.IDFPEC = ''  # ZAID for daughter for FPEC
        self.FPECX  =  0. # Fraction of all EC or positron decays which result in excited state
        self.IDFPECX= ''  # ZAID for daughter for FPECX
        self.FA     =  0. # Fraction of all decay events which are alpha
        self.IDFA   = ''  # ZAID for daughter for alpha
        self.FIT    =  0. # 'fraction of all the decay events of an excited nuclear state
                          # which result in the production of the ground state of the same nuclide'
        self.IDFIT  = ''  # ZAID for daughter of FIT
        self.FSF    =  0. # Fraction that decay events that are spontaneous fission
        # No daughter listed, will loop through all elements for yields
        self.FN     =  0. # Fraction of all decay events that are beta + neutron decays
        self.IDFN   = ''  # ZAID for daughter for FN
        #Note: Negatron beta decay = 1 - FBX - FPEC - FA - FIT - FSF - FN
        self.FB     =  0. # Fraction of all decay events which are beta
        self.IDFB   = ''


def FindPotentialMatch(List,protons,A,Fraction,Excited,LOUD=False):
    if(Fraction<0 or Fraction>1):
        print("Fraction of decays is too low or high : ",Fraction)
        print("Inquire further")
        quit()
    if Fraction>0:
        for item in List:
            if protons in item[0:2] and A in item[:-1] and item[-1]==Excited:
                Toreturn=item
    else:
        Toreturn=''
    try: #To make sure its defined
        Toreturn
    except NameError:
        if LOUD:
            print("Could not find daughter in list of isotopes when expecting one")
            print("Looking for Protons : ",protons," Total Nucleons : ",A," Fraction",Fraction)
            #print("Close items are")
            #for item in List:
            #    if protons in item[:-1] and A in item[:-1]:
            #        print(item)
        if (Fraction<2):
            if LOUD:
                print("Assuming it doesn't matter, will let slide")
            Toreturn=''
        else:
            quit()
    
    return(Toreturn)


def DecayInfo(Nuclide_Names,parent,Lambda,proton,A):
    """
    This function will store and return decay information from
    the tape9.inp file
    """

    Info=DecayClass()
    
    with open('Data/tape9.inp') as f:
        TAPE9Content=f.readlines()

    Found=False
    for line in TAPE9Content:
        hold=line.split()

        #Looking at second line in each library (put this if statement above below)
        if Found:
            #Spontaneous fission
            Info.FSF=float(hold[1])
            #Beta plus neutron
            Info.FN=float(hold[2])
            Info.IDFN=FindPotentialMatch(Nuclide_Names,str(int(proton)+1),str(int(A)-1),Info.FN,'0')
            break
        
        #Looking for fission product and actinide decay information
        #The libraries we are looking through for this information are '2' and '3'
        if ('2' == hold[0] or '3' == hold[0]) and hold[1]==parent:
            Found=True

            #Beta minus to excited
            Info.FBX=float(hold[4])
            Info.IDFBX=FindPotentialMatch(Nuclide_Names,str(int(proton)+1),A,Info.FBX,'1')
            #positron or EC
            Info.FPEC=float(hold[5])  #Total positron or EC
            Info.FPECX=float(hold[6]) #percent of above to excited
            if Info.FPECX < 1:
                Info.IDFPEC=FindPotentialMatch(Nuclide_Names,str(int(proton)-1),A,Info.FPEC,'0')
            Info.IDFPECX=FindPotentialMatch(Nuclide_Names,str(int(proton)-1),A,Info.FPECX,'1')
            Info.FA=float(hold[7])    #Fraction of events that are alpha
            Info.IDFA=FindPotentialMatch(Nuclide_Names,str(int(proton)-2),str(int(A)-4),Info.FA,'0')
            #Excited state to ground state
            Info.FIT=float(hold[8])
            Info.IDFIT=FindPotentialMatch(Nuclide_Names,proton,A,Info.FIT,'0')
            
    #Calculate the beta
    Info.FB=1-Info.FBX-Info.FPEC-Info.FA-Info.FIT-Info.FSF-Info.FN

    if(Info.FB<0 or Info.FB>1):
        #print("Fraction of beta decays is too low or high : ",Info.FB)
        if abs(Info.FB)<7e-4 or abs(Info.FB-1)<7e-4:
            #print("But I'll let it slide and set to zero")
            Info.FB=0
        else:
            print("I can't let this slide, not small enough")
            print("Beta excited",Info.FBX)
            print("EC",Info.FPEC)
            print("Alpha decay ground",Info.FA)
            print("Excited to ground",Info.FIT)
            print("Spontaneous fission",Info.FSF)
            print("Beta Plus neutron",Info.FN)
            quit()

    if Lambda>0:
        Info.IDFB=FindPotentialMatch(Nuclide_Names,str(int(proton)+1),A,Info.FB,'0')

    #Make sure all decay fractions add to one (for some reason the EC to excited is
    # probability given a EC
    FPEC=(1-Info.FPECX)*Info.FPEC
    FPECX=Info.FPECX*Info.FPEC
    Info.FPEC=FPEC
    Info.FPECX=FPECX
    
    return(Info)


def FindPotentialMatchX(List,protons,A,XSection,Excited,Lambda,LOUD=False):
    if(XSection<0):
        print("Fraction of decays is too low",XSection)
        print("Inquire further")
        quit()
    if XSection>0:
        for item in List:
            if protons in item[0:2] and A in item[:-1] and item[-1]==Excited:
                Toreturn=item
    else:
        Toreturn=''
    try: #To make sure its defined
        Toreturn
    except NameError:
        if LOUD:
            print("Could not find daughter in list of isotopes when expecting one")
            print("Looking for Protons : ",protons," Total Nucleons : ",A," XSection ",XSection)
            print("Lambda = ",Lambda)
            #print("Close items are")
            #for item in List:
            #    if protons in item[:-1] and A in item[:-1]:
            #        print(item)
        if (XSection<100000 or Lambda>0.1):
            if LOUD:
                print("Assuming it doesn't matter, will let slide")
            Toreturn=''
            XSection=0
        else:
            quit()
    
    return(Toreturn,XSection)

class XSectionClass:
    def __init__(self):
        self.SNG   = 0     # the effective, one grounp (n,y) x-section leadin to ground state
        self.IDSNG = ''    # ZAID ID for the above
        self.SN2N    =  0. # the effective, one grounp (n,2n) x-section leading to ground state
        self.IDSN2N = ''   # ZAID ID for the above
        self.SN3N  =   0   # effective to ground
        self.IDSN3N  = ''  # ZAID for daughter for above
        self.SNA   =  0.   # effective to ground n,alpha
        self.IDSNA = ''    # ZAID for daughter for above
        self.SNF  =  0.    
        self.SNP     =  0. # 
        self.IDSNP   = ''  # ZAID for daughter for above
        self.SNGX    =  0. # 
        self.IDSNGX  = ''  # ZAID for daughter of above
        self.SN2NX     =  0. # 
        self.IDSN2NX   = ''  # ZAID for daughter for above


def XSectionInfo(Nuclide_Names,parent,L,proton,A):
    """
    This function will store and return decay information from
    the tape9.inp file
    """

    Info=XSectionClass()
    
    with open('Data/tape9.inp') as f:
        TAPE9Content=f.readlines()

    Found=False
    for line in TAPE9Content:
        hold=line.split()
        
        #Looking for fission product and actinide cross section information
        #The libraries we are looking through for this information are '602' and '603'
        #set a cross section to zero if daughter is not found (second return of FindPotentialMatchX)
        if ('602' == hold[0] or '603' == hold[0]) and hold[1]==parent:

            #(n,gamma)
            Info.SNG=float(hold[2])
            Info.IDSNG,Info.SNG=FindPotentialMatchX(Nuclide_Names,proton,str(int(A)+1),Info.SNG,'0',L)
            #n,2n
            Info.SN2N=float(hold[3])
            Info.IDSN2N,Info.SN2N=FindPotentialMatchX(Nuclide_Names,
                                                      proton,str(int(A)-1),Info.SN2N,'0',L)
            if '602' == hold[0]: #Actinides
                #n,3n
                Info.SN3N=float(hold[4])
                Info.IDSN3N,Info.SN3N=FindPotentialMatchX(Nuclide_Names,
                                                          proton,str(int(A)-2),Info.SN3N,'0',L)
                #n,f
                Info.SNF=float(hold[5])
            if '603' == hold[0]: #Fission products
                #n,alpha
                Info.SNA=float(hold[4])
                Info.IDSNA,Info.SNA=FindPotentialMatchX(Nuclide_Names,str(int(proton)-2),str(int(A)-3),
                                               Info.SNA,'0',L)
                #n,p
                Info.SNP=float(hold[5])
                Info.IDSNP,Info.SNP=FindPotentialMatchX(Nuclide_Names,
                                                        str(int(proton)-1),A,Info.SNP,'0',L)
            #(n,gamma)excited
            Info.SNGX=float(hold[6])
            Info.IDSNGX,Info.SNGX=FindPotentialMatchX(Nuclide_Names,proton,
                                                      str(int(A)+1),Info.SNGX,'1',L)
            #(n,2n)excited
            Info.SN2NX=float(hold[7])
            Info.IDSN2NX,Info.SN2NX=FindPotentialMatchX(Nuclide_Names,proton,
                                                        str(int(A)-1),Info.SN2NX,'1',L)
            
    return(Info)

def YieldInfo(yieldiso,holdIndex,LOUD=False):
    with open('Data/tape9.inp') as f:
        TAPE9Content=f.readlines()

    Found=False;Yield=False
    for line in TAPE9Content:
        hold=line.split()
        
        if Found:
            if Yield:
                returnyield=float(hold[holdIndex])
            else:
                returnyield=0
            break
        if '603' == hold[0] and hold[1]==yieldiso:
            Found=True
            if float(hold[8])>0:
                Yield=True
            else:
                Yield=False
                
    if not Found:
        if LOUD:
            print("Did not find a yield for ",yieldiso)
        returnyield=0
    return(returnyield)

def AddFission(A,Nuclides,isotope,c1,c2,row,LOUD=False):
    if isotope=="922320": #Th232
        #print("Th232")
        holdIndex=1
        element="Th232"
    elif isotope=="922330": #U233
        #print("U233")
        holdIndex=2
        element="U233"
    elif isotope=="922350": #U235
        #print("U235")
        holdIndex=3
        element="U235"
    elif isotope=="922380": #U238
        #print("U238")
        holdIndex=4
        element="U238"
    elif isotope=="942390": #Pu239
        #print("Pu239")
        holdIndex=5
        element="Pu239"
    elif isotope=="942410": #Pu241
        #print("Pu241")
        holdIndex=6
        element="Pu241"
    elif isotope=="962450": #Cm245
        #print("Cm245")
        holdIndex=7
        element="Cm245"
    elif isotope=="982490": #Cf249
        #print("Cf249")
        holdIndex=8
        element="Cf249"
    else:
        if LOUD:
            print("Did not find yields for ",isotope," because not provided")
        holdIndex=100

    if holdIndex<40:
        YieldSum=0 #Check what the yields sum up to
        for yieldiso in Nuclides:
            actualrow=Nuclides[yieldiso]
            Yield=YieldInfo(yieldiso,holdIndex,LOUD=False)
            A[actualrow,row]=A[actualrow,row]+c1*c2*Yield/100
            YieldSum=YieldSum+Yield
        if LOUD:
            print("Yield Sum for ",element," = ",YieldSum) 
    return(A)

def MakeAb(phi,Nuclides,Nuclide_Names,Decay_Conts):

    # Create Activation and Decay Matrix and initial
    # nuclide quantity vector
    A = np.zeros((len(Nuclides),len(Nuclides)))

    #10^14 1/cm^2/s in 1/cm^2 /year #Bars included
    phi = phi * 60 * 60 * 24 * 365.25 *(10**(-24))


    for isotope in Nuclides:
        row = Nuclides[isotope]
        proton=isotope[0:2]
        if isotope[2]=="0":
            Anum=isotope[3:5]
        elif isotope[4]=="0" and float(proton)*3<100:
            Anum=isotope[2:4]
        elif isotope[2]!="0":
            Anum=isotope[2:5]
        else:
            print("Missed logic in finding A number")
            quit()
        if int(Anum)<int(proton):
            print("Something is wrong, more protons than neutrons")
            print("Proton: ",proton,"A: ",Anum,"ZAID :",isotope)
            quit()

        #print(row,isotope,Decay_Conts[row])
        Lambda=Decay_Conts[row]
        #Store all decay and x section information for row (I think I mean column)
        row_decay =  DecayInfo(Nuclide_Names,isotope,Lambda,proton,Anum)        
        row_XSection=XSectionInfo(Nuclide_Names,isotope,Lambda,proton,Anum)

        #Convert LAmbda to years^-1 (If radioactive)
        if Lambda>0:
            Lambday=Lambda*60*60*24*365.25
        else:
            Lambday=0

        XList=["SN2N","SN2NX","SN3N","SNA","SNG","SNGX","SNP","SNF"]
        DList=["FA","FB","FBX","FIT","FN","FPEC","FPECX","FSF"]
        
        #USeful to see things
        #for a in dir(row_XSection):
        #    if not a.startswith('__'):
        #        print(a)
        #        print(getattr(row_XSection,a))
        #quit()
        #for a in dir(row_decay):
        #    if not a.startswith('__'):
        #        print(a)
        #        print(getattr(row_decay,a))
        #quit()
        
        #Sum up all the sigma abs
        sigma_sum=0
        for xsec in XList:
            sigma_sum=sigma_sum+getattr(row_XSection,xsec)
        
        # Diagonal Assignment
        A[row,row] = -Lambday - phi*sigma_sum

        ##does a single isotope produce another isotope through more than one path
        ##like EC and NP
        #for xsec in XList:
        #    for decay in DList:
        #        if not decay[-1]=='F' and not "F" in xsec and len(getattr(row_decay,"ID"+decay))>1:
        #            if getattr(row_XSection,"ID"+xsec) == getattr(row_decay,"ID"+decay):
        #                print("Isotope ",getattr(row_XSection,"ID"+xsec),"Produced from ",
        #                      isotope," with RNS ",decay,xsec)
        #                print("Both will be used")

        #Off diagonal assignment adding x-sec productions except from fission
        for xsec in XList:
            if not "F" in xsec: #Don't do fission yet
                Product=getattr(row_XSection,"ID"+xsec)
                if len(Product)>1:
                    actualrow=Nuclides[Product]
                    A[actualrow,row]=A[actualrow,row]+phi*getattr(row_XSection,xsec)
        #Off diagonal assignment adding all decay except from spontaneous fission
        for decay in DList:
            if not decay[-1]=='F':
                Product=getattr(row_decay,"ID"+decay)
                if len(Product)>1:
                    actualrow=Nuclides[Product]
                    A[actualrow,row]=A[actualrow,row]+Lambday*getattr(row_decay,decay)

        #Now for xfission (these next two if statements take the longest)
        if row_XSection.SNF>0:
            A=AddFission(A,Nuclides,isotope,row_XSection.SNF,phi,row,LOUD=False)
        #Now for spontaneous fission
        if row_decay.FSF>0:
            A=AddFission(A,Nuclides,isotope,row_decay.FSF,Lambday,row,LOUD=False)
   
    b = np.zeros(len(Nuclides))
    b[Nuclides['922340']] = 6.94741E23
    b[Nuclides['922350']] = 7.6864E25
    b[Nuclides['922380']] = 2.4532E27

    return(A,b)

  
