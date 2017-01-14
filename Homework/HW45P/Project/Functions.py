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
YScale="log"                 # 'linear' or 'log'

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

Xlabel='Time [days]'
Ylabel="Mass $\\left[\\frac{g}{kg \\text{ FLiBe}}\\right]$"


nuclides = {  'H1':0,    'H2':1,    'H3':2,  'He3':3,  'He4':4,
              'He6':5,   'Li6':6,   'Li7':7,  'Li8':8,  'Be8':9,
              'Be9':10, 'Be10':11, 'Be11':12, 'B10':13, 'B11':14,
              'B12':15,  'C12':16,  'C13':17, 'C14':18, 'C15':19,
              'N13':20,  'N14':21,  'N15':22, 'N16':23, 'N17':24,
              'O16':25,  'O17':26,  'O18':27, 'O19':28, 'F18':29,
              'F19':30, 'F20':31, 'Ne20':32}



atom_mass = np.array([1.007825032,2.014101778,3.0160492779,  #2
                      3.016029320,4.002603254,6.151228874,   #5
                      6.015122887,7.0160034366,8.022486246,  #8
                      8.005305102,9.012183065,10.013534695,  #11
                      11.02166108,10.01293695,11.00930536,   #14
                      12.0269221, 12, 13.003354835,          #17
                      14.003241988, 15.01059926,13.00573861, #20
                      14.003074004, 15.000108898, 16.0061019,#23
                      17.008449, 15.994914619, 16.999131756, #26
                      17.999159612, 19.0035780,17.99915961286,#29
                      18.998403162, 19.999981252, 19.992440176])

nuclide_names = ('H1', 'H2', 'H3', 'He3', 'He4', 'He6', 'Li6',
                 'Li7', 'Li8', 'Be8', 'Be9', 'Be10',
                 'Be11', 'B10', 'B11', 'B12', 'C12', 'C13',
                 'C14', 'C15', 'N13', 'N14', 'N15', 'N16',
                 'N17', 'O16', 'O17', 'O18', 'O19', 'F18',
                 'F19', 'F20', 'Ne20')

decay_consts = np.array([0., 0., np.log(2)/3.887896E8, #H1 H2 H3
                         0., 0., np.log(2)/0.807, #He3 He4 He6
                         0.,0., np.log(2)/0.840,  #Li6 #Li7 #Li8
                         np.log(2)/6E-17,0.,    #Be8 #Be9
                         np.log(2)/4.73364E13,np.log(2)/13.8, # Be10,11
                         0., 0., np.log(2)/0.0202, #B10 B11 B12
                         0., 0.,np.log(2)/1.803517E11, #C12 C13 C14
                         np.log(2)/2.45,np.log(2)/598.2, #C15 N13
                         0., 0., np.log(2)/7.13, # N14 N15 N16
                         np.log(2)/4.174, 0., 0., 0., #N17 O16 O17 O18
                         np.log(2)/26.9, np.log(2)/6586.2, #O19 F18
                         0., np.log(2)/11.1, 0.]) #F19 F20 Ne20
Na=6.0221409E23


################################################################
################### Function for Vars ##########################
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

def Isotopes():
  """
  This function will create a dictionary 'Nuclides'
  with nuclides found in tape9.inp, excluding activation isotopes
  """
  Nuclides={}
  Nuclide_Names=()

  with open('tape9.inp') as f:
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
        
  return(Nuclides,Nuclide_Names)

def GatherDecay(Nuclide_Names):
  """
  This function will create an array 'Decay_Consts'
  that has all the half-life information for our system.
  """

  Decay_Consts = np.zeros(len(Nuclide_Names))

  
  with open('tape9.inp') as f:
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
###################### Functions ###############################
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

def MakeAb(hi_flux_frac = 0.5,phi = 1.0e14):
    
    """Interaction functions
    @ In, nuclides:  dictionary with isotope keywords and 
                     corresponding indices
    @ In, parent:    parent nuclides undergoing a decay or interaction
    @Out, value:     new value in interaction matrix, either a half 
                     life [secs] or 2.45 MeV and 14.1 MeV cross 
                     sections [barns]
    """
   
    def betanegdecay(nuclides, parent):
        if   parent == 'F20':  return nuclides['Ne20'], 11.1 # s
        elif parent == 'O19':  return nuclides['F19'],  26.9 # s
        elif parent == 'N16':  return nuclides['O16'],  7.13 # s
        elif parent == 'N17':  return nuclides['O17'],  4.174 # s
        elif parent == 'C14':  return nuclides['N14'],  1.803517E11 # s
        elif parent == 'C15':  return nuclides['N15'],  2.45 # s
        elif parent == 'B12':  return nuclides['C12'],  0.0202 # s
        elif parent == 'Be10': return nuclides['B10'],  4.73364E13 # s
        elif parent == 'Be11': return nuclides['B11'],  13.8 # s
        elif parent == 'Li8':  return nuclides['Be8'],  0.840 # s
        elif parent == 'He6':  return nuclides['Li6'],  0.807 # s
        elif parent == 'H3':   return nuclides['He3'],  3.887896E8 # s
        else: return -1, 0.0

    def betaposdecay(nuclides, parent):
        if   parent == 'F18':  return nuclides['O18'],  6586.2 # s
        elif parent == 'N13':  return nuclides['C13'],  598.2 # s
        else: return -1, 0.0

    def twoalphadecay(nuclides, parent):
        if   parent == 'Be8':  return nuclides['He4'],  7.0E-17 # s
        else: return -1, 0.0

    def n_gamma(nuclides, parent):
        if   parent == 'F19':
            return nuclides['F20'],  8.649107E-5,    3.495035E-5
        elif parent == 'O16':
            return nuclides['O17'],  1.0E-4,         1.0E-4
        elif parent == 'O17':
            return nuclides['O18'],  2.2675E-4,      2.087114E-4
        elif parent == 'N14':
            return nuclides['N15'],  2.397479E-5,    1.679535E-5
        elif parent == 'N15':
            return nuclides['N16'],  8.121795E-6,    8.56E-6
        elif parent == 'Be9':
            return nuclides['Be10'],1.943574E-6,    1.660517E-6
        elif parent == 'Li6':
            return nuclides['Li7'],  1.106851E-5,    1.017047E-5
        elif parent == 'Li7':
            return nuclides['Li8'],  4.677237E-6,    4.105546E-6
        elif parent == 'He3':
            return nuclides['He4'],  9.28775E-5,     3.4695E-5
        elif parent == 'H2':
            return nuclides['H3'],   8.413251E-6,    9.471512E-6
        else:
            return -1, 0.0, 0.0


    def n_2n(nuclides, parent):
        if   parent == 'F19':
            return nuclides['F18'],  0.0,            0.04162
        elif parent == 'O17':
            return nuclides['O16'],  0.0,            0.066113
        elif parent == 'N14':
            return nuclides['N13'],  0.0,            0.006496
        elif parent == 'N15':
            return nuclides['N14'],  0.0,            0.112284
        elif parent == 'B11':
            return nuclides['B10'],  0.0,            0.018805
        elif parent == 'Be9':
            return nuclides['Be8'],  0.0205,         0.484483
        elif parent == 'Li7':
            return nuclides['Li6'],  0.0,            0.031603
        elif parent == 'H3':
            return nuclides['H2'],   0.0,            0.0497
        elif parent == 'H2':
            return nuclides['H1'],   0.0,            0.166767
        else:
            return -1, 0.0, 0.0

    def n_alpha(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['N16'],nuclides['He4']],2.1667E-5,0.028393
        elif parent == 'O16':
            return [nuclides['C13'], nuclides['He4']], 0.0, 0.144515
        elif parent == 'O17':
            return [nuclides['C14'], nuclides['He4']],0.117316,0.260809
        elif parent == 'N14':
            return [nuclides['B11'], nuclides['He4']],0.104365,0.080516
        elif parent == 'N15':
            return [nuclides['B12'], nuclides['He4']], 0.0,0.069240
        elif parent == 'B10':
            return [nuclides['Li7'], nuclides['He4']],0.281082,0.044480
        elif parent == 'B11':
            return [nuclides['Li8'], nuclides['He4']], 0.0,0.031853
        else:
            return [-1,-1], 0.0, 0.0

    def n_2alpha(nuclides, parent):
        if   parent == 'N14':
            return [nuclides['Li7'], nuclides['He4']],  0.0,0.031771
        elif parent == 'B10':
            return [nuclides['H3'],  nuclides['He4']],0.038439,0.095487
        else:
            return [-1,-1], 0.0, 0.0

    def n_nalpha(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['N15'], nuclides['He4']], 0.0,0.3818
        elif parent == 'O17':
            return [nuclides['C13'], nuclides['He4']], 0.0,0.043420
        elif parent == 'N15':
            return [nuclides['B11'], nuclides['He4']], 0.0,0.012646
        elif parent == 'B11':
            return [nuclides['Li7'], nuclides['He4']], 0.0,0.286932
        elif parent == 'Be9':
            return [nuclides['He6'], nuclides['He4']], 0.0825,0.0104
        else:
            return [-1,-1], 0.0, 0.0

    def n_2nalpha(nuclides, parent):
        if   parent == 'Li6':
            return [nuclides['H1'], nuclides['He4']],  0.0,0.0783
        elif parent == 'Li7':
            return [nuclides['H2'], nuclides['He4']],  0.0,0.020195
        else:
            return [-1,-1], 0.0, 0.0

    def n_3nalpha(nuclides, parent):
        if   parent == 'Li7':
            return [nuclides['H1'], nuclides['He4']],  0.0,6.556330E-5
        else:
            return [-1,-1], 0.0, 0.0

    def n_p(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['O19'],  nuclides['H1']], 0.0,0.018438
        elif parent == 'O16':
            return [nuclides['N16'],  nuclides['H1']], 0.0,0.042723
        elif parent == 'O17':
            return [nuclides['N17'],  nuclides['H1']], 0.0, 0.041838
        elif parent == 'N14':
            return [nuclides['C14'], nuclides['H1']],0.014102,0.043891
        elif parent == 'N15':
            return [nuclides['C15'],  nuclides['H1']], 0.0,0.019601
        elif parent == 'B10':
            return [nuclides['Be10'], nuclides['H1']],0.018860,0.034093
        elif parent == 'B11':
            return [nuclides['Be11'], nuclides['H1']], 0.0,0.005564
        elif parent == 'Li6':
            return [nuclides['He6'],  nuclides['H1']], 0.0,0.00604
        elif parent == 'He3':
            return [nuclides['H3'],nuclides['H1']],0.714941, 0.121
        else:
            return [-1,-1], 0.0, 0.0

    def n_np(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['O18'],nuclides['H1']], 0.0, 0.061973
        elif parent == 'N15':
            return [nuclides['C14'],  nuclides['H1']], 0.0, 0.044827
        elif parent == 'B11':
            return [nuclides['Be10'], nuclides['H1']], 0.0, 0.001016
        else:
            return [-1,-1], 0.0, 0.0

    def n_d(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['O18'], nuclides['H2']],  0.0, 0.022215
        elif parent == 'O16':
            return [nuclides['N15'], nuclides['H2']],  0.0,0.017623
        elif parent == 'O17':
            return [nuclides['N16'], nuclides['H2']],  0.0,0.020579
        elif parent == 'N14':
            return [nuclides['C13'], nuclides['H2']],  0.0, 0.042027
        elif parent == 'N15':
            return [nuclides['C14'], nuclides['H2']],  0.0,0.014926
        elif parent == 'B10':
            return [nuclides['Be9'], nuclides['H2']],  0.0, 0.031270
        elif parent == 'Li7':
            return [nuclides['He6'], nuclides['H2']],  0.0, 0.010199
        elif parent == 'He3':
            return [nuclides['H2'],  nuclides['H2']],  0.0,0.07609
        else:
            return [-1,-1], 0.0, 0.0

    def n_t(nuclides, parent):
        if   parent == 'F19':
            return [nuclides['O17'], nuclides['H3']],  0.0,0.01303
        elif parent == 'N14':
            return [nuclides['C12'], nuclides['H3']],  0.0,0.028573
        elif parent == 'N15':
            return [nuclides['C13'], nuclides['H3']],  0.0,0.020163
        elif parent == 'B11':
            return [nuclides['Be9'], nuclides['H3']],  0.0,0.015172
        elif parent == 'Be9':
            return [nuclides['Li7'], nuclides['H3']],  0.0,0.020878
        elif parent == 'Li6':
            return [nuclides['He4'], nuclides['H3']],  0.206155,0.0258
        else:
            return [-1,-1], 0.0, 0.0

    
    # Create Activation and Decay Matrix and initial
    # nuclide quantity vector
    A = np.zeros((len(nuclides),len(nuclides)))

    lo_flux_frac = (1.0-hi_flux_frac)

    phi = phi * 60 * 60 * 24 #10^14 1/cm^2/s in 1/cm^2 /day
    phi_hi = hi_flux_frac*phi*1.0e-24
    phi_lo = lo_flux_frac*phi*1.0e-24


    for isotope in nuclides:
        row = nuclides[isotope]
        row_betanegdecay =  betanegdecay(nuclides, isotope)
        row_betaposdecay =  betaposdecay(nuclides, isotope)
        row_2alphadecay =   twoalphadecay(nuclides, isotope)
        row_n_gamma =       n_gamma(nuclides, isotope)
        row_n_2n =          n_2n(nuclides, isotope)
        row_n_alpha =       n_alpha(nuclides, isotope)
        row_n_2alpha =      n_2alpha(nuclides, isotope)
        row_n_nalpha =      n_nalpha(nuclides, isotope)
        row_n_2nalpha =     n_2nalpha(nuclides, isotope)
        row_n_3nalpha =     n_3nalpha(nuclides, isotope)
        row_n_p =           n_p(nuclides, isotope)
        row_n_np =          n_np(nuclides, isotope)
        row_n_d =           n_d(nuclides, isotope)
        row_n_t =           n_t(nuclides, isotope)
        row_lo_act_sum = row_n_gamma[1] + row_n_2n[1] +\
                         row_n_alpha[1] + row_n_2alpha[1] +\
                         row_n_nalpha[1] + row_n_2nalpha[1] + \
                         row_n_3nalpha[1] + row_n_p[1] +\
                         row_n_np[1] + row_n_d[1] +\
                         row_n_t[1]
        row_hi_act_sum = row_n_gamma[2] + row_n_2n[2] +\
                         row_n_alpha[2] + row_n_2alpha[2] +\
                         row_n_nalpha[2] + row_n_2nalpha[2] + \
                         row_n_3nalpha[2] + row_n_p[2] +\
                         row_n_np[2] + row_n_d[2] +row_n_t[2]
        # try:
        #     if row_n_alpha[0] >= 0:
        #         print(row_n_alpha)
        #         donotuse=100
        #     continue
        # except TypeError:
        #     print(row_n_alpha)
        #     print(row_n_alpha[0][0])
        #     quit()
        
        if row_betanegdecay[0] >= 0:
            # [days^-1]
            row_lambda = np.log(2)*60*60*24/row_betanegdecay[1] 
        elif row_betaposdecay[0] >= 0:
            # [days^-1]
            row_lambda = np.log(2)*60*60*24/row_betaposdecay[1] 
        elif row_2alphadecay[0] >= 0:
            # [days^-1]
            row_lambda = np.log(2)*60*60*24/row_2alphadecay[1] 
        else:
            row_lambda = 0.0
    
        # Diagonal Assignment
        A[row,row] = -row_lambda - phi_lo*row_lo_act_sum -\
                     phi_hi*row_hi_act_sum
        # Off Diagonal Assignment
        if row_betanegdecay[0] >= 0:
            A[row_betanegdecay[0],row] = np.log(2)*60*60*24/\
                                         row_betanegdecay[1]
        if row_betaposdecay[0] >= 0:
            A[row_betaposdecay[0],row] = np.log(2)*60*60*24/\
                                         row_betaposdecay[1]
        if row_2alphadecay[0] >= 0:
            A[row_2alphadecay[0],row] = np.log(2)*60*60*24/\
                                        row_2alphadecay[1]
        if row_n_gamma[0] >= 0:
            A[row_n_gamma[0],row] = phi_lo*row_n_gamma[1] +\
                                    phi_hi*row_n_gamma[2]
        if row_n_2n[0] >= 0:
            A[row_n_2n[0],row] = phi_lo*row_n_2n[1] +\
                                 phi_hi*row_n_2n[2]
        if row_n_alpha[0][0] >= 0:
            for i in row_n_alpha[0]:
                A[i,row] = phi_lo*row_n_alpha[1] +\
                           phi_hi*row_n_alpha[2]
        if row_n_2alpha[0][0] >= 0:
            for i in row_n_2alpha[0]:
                A[i,row] = phi_lo*row_n_2alpha[1] +\
                           phi_hi*row_n_2alpha[2]
        if row_n_nalpha[0][0] >= 0:
            for i in row_n_nalpha[0]:
                A[i,row] = phi_lo*row_n_nalpha[1] +\
                           phi_hi*row_n_nalpha[2]
        if row_n_2nalpha[0][0] >= 0:
            for i in row_n_2nalpha[0]:
                A[i,row] = phi_lo*row_n_2nalpha[1] +\
                           phi_hi*row_n_2nalpha[2]
        if row_n_3nalpha[0][0] >= 0:
            for i in row_n_3nalpha[0]:
                A[i,row] = phi_lo*row_n_3nalpha[1] +\
                           phi_hi*row_n_3nalpha[2]
        if row_n_p[0][0] >= 0:
            for i in row_n_p[0]:
                A[i,row] = phi_lo*row_n_p[1] + phi_hi*row_n_p[2]
        if row_n_np[0][0] >= 0:
            for i in row_n_np[0]:
                A[i,row] = phi_lo*row_n_np[1] + phi_hi*row_n_np[2]
        if row_n_d[0][0] >= 0:
            for i in row_n_d[0]:
                A[i,row] = phi_lo*row_n_d[1] + phi_hi*row_n_d[2]
        if row_n_t[0][0] >= 0:
            for i in row_n_t[0]:
                A[i,row] = phi_lo*row_n_t[1] + phi_hi*row_n_t[2]

    
    b = np.zeros(len(nuclides))

    # N_0 expressed as kg nuclide per kg FLiBe
    #b[nuclides['F19']] = 0.7685
    #b[nuclides['Be9']] = 0.0911
    #b[nuclides['Li6']] = 0.01065636
    #b[nuclides['Li7']] = 0.12974364
    AtomsofFLiBe=6.0899894727155e24
    b[nuclides['F19']] = AtomsofFLiBe*4
    b[nuclides['Be9']] = AtomsofFLiBe*1
    b[nuclides['Li6']] = AtomsofFLiBe*2*0.0759
    b[nuclides['Li7']] = AtomsofFLiBe*2*0.9241

    return(A,b)

################################################################
################### Plotting Function ##########################
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
    x=df[List[0]].values[2:-1]

    Check=0
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:-1])/Na)*df[Item].values[0]
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
    x=df[List[0]].values[2:-1]

    Check=0;Sum=np.zeros(len(x))
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:-1]))*df[Item].values[1]
      Sum=Sum+y
      if len(x)>NumOfPoints:
        xP=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
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
    YlabelBq="Activity $\\left[\\frac{Bq}{kg \\text{ FLiBe}}\\right]$"
    ax.set_ylabel(YlabelBq,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)

    Legend(ax)
    plt.savefig("Plots/"+Name+'_Bq.pdf') 
    
def plots2(df,df2,Plotting,Name,NumOfPoints,Method1,Method2):
    #Plot in grams
    fig=plt.figure(figsize=FigureSize)  
    ax=fig.add_subplot(111)

    List=list(df.columns.values)
    x=df[List[0]].values[2:-1]
    
    Check=0
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:-1])/Na)*df[Item].values[0]
      y2=((df2[Item].values[2:-1])/Na)*df2[Item].values[0]
      if len(x)>NumOfPoints:
        x=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
        y2=reduceList(y2,NumOfPoints)
      ax.plot(x,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method1)
      Check=Check+1
      ax.plot(x,y2,
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
    x=df[List[0]].values[2:-1]
    
    Check=0;Sum=np.zeros(len(x));Sum2=np.zeros(len(x))
    for Item in Plotting:
      InList(Item,List) #Check if we have the isotope
      y=((df[Item].values[2:-1]))*df[Item].values[1]
      y2=((df2[Item].values[2:-1]))*df2[Item].values[1]
      Sum=Sum+y
      Sum2=Sum2+y2
      if len(x)>NumOfPoints:
        xP=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
        y2=reduceList(y2,NumOfPoints)
      ax.plot(xP,y,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method1)
      Check=Check+1
      ax.plot(xP,y2,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check),
              alpha=loop_values(Alpha_Value,Check),
              label=Item+" "+Method2)
      Check=Check+1
    if len(x)>NumOfPoints:
      Sum=reduceList(Sum,NumOfPoints)
      Sum2=reduceList(Sum2,NumOfPoints)
    ax.plot(xP,Sum,
              linestyle=loop_values(LineStyles,Check),
              marker=loop_values(MarkerType,Check),
              color=loop_values(Colors,Check),
              markersize=loop_values(MarkSize,Check)*1.5,
              alpha=loop_values(Alpha_Value,Check),
              label="Sum "+Method1)
    Check=Check+1
    ax.plot(xP,Sum2,
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
    YlabelBq="Activity $\\left[\\frac{Bq}{kg \\text{ FLiBe}}\\right]$"
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
    
def PrepFile(Name,n0):
  File=open(Name,'w')
  File.write("Mass then Time (d),"+','.join(nuclide_names)+'\n')
  File.write("Masses,"+ListToStr(atom_mass)) #New line already included
  File.write("DecayConts,"+ListToStr(decay_consts))
  File.write("0,"+ListToStr(n0))
  return(File)

def Print(Method,nuclide,Results,Time):
  Index=nuclides[nuclide]
  MassConversion=atom_mass[Index]/Na
  string="Isotope "+nuclide_names[Index]+", Mass (g) = "
  Mass=Results[Index]*MassConversion
  Mass="%.4e" % Mass
  print(Method+" :",string,Mass,"Time=%.2f" % Time)

def Years(Method,nuclide,Results):
  Index=nuclides[nuclide]
  LambdaY=decay_consts[Index]*60*60*24*365.25
  Lambdas=decay_consts[Index]
  string="Isotope "+nuclide_names[Index]+", Years to 444 Bq = "
  Years=(-1/LambdaY)*np.log(444/(Results[Index]*Lambdas))
  print(Method+" :",string,"%.3e" % Years)

#####################################################################
#################### Under Construction #############################
#####################################################################

def BetaNegDecay(nuclides, parent):
    if   parent == 'F20':  return nuclides['Ne20'], 11.1 # s
    elif parent == 'O19':  return nuclides['F19'],  26.9 # s
    elif parent == 'N16':  return nuclides['O16'],  7.13 # s
    elif parent == 'N17':  return nuclides['O17'],  4.174 # s
    elif parent == 'C14':  return nuclides['N14'],  1.803517E11 # s
    elif parent == 'C15':  return nuclides['N15'],  2.45 # s
    elif parent == 'B12':  return nuclides['C12'],  0.0202 # s
    elif parent == 'Be10': return nuclides['B10'],  4.73364E13 # s
    elif parent == 'Be11': return nuclides['B11'],  13.8 # s
    elif parent == 'Li8':  return nuclides['Be8'],  0.840 # s
    elif parent == 'He6':  return nuclides['Li6'],  0.807 # s
    elif parent == 'H3':   return nuclides['He3'],  3.887896E8 # s
    else: return -1, 0.0

def BetaPosDecay(nuclides, parent):
    if   parent == 'F18':  return nuclides['O18'],  6586.2 # s
    elif parent == 'N13':  return nuclides['C13'],  598.2 # s
    else: return -1, 0.0

def n_gamma(nuclides, parent):
    if   parent == 'F19':
        return nuclides['F20'],  8.649107E-5,    3.495035E-5
    elif parent == 'O16':
        return nuclides['O17'],  1.0E-4,         1.0E-4
    elif parent == 'O17':
        return nuclides['O18'],  2.2675E-4,      2.087114E-4
    elif parent == 'N14':
        return nuclides['N15'],  2.397479E-5,    1.679535E-5
    elif parent == 'N15':
        return nuclides['N16'],  8.121795E-6,    8.56E-6
    elif parent == 'Be9':
        return nuclides['Be10'],1.943574E-6,    1.660517E-6
    elif parent == 'Li6':
        return nuclides['Li7'],  1.106851E-5,    1.017047E-5
    elif parent == 'Li7':
        return nuclides['Li8'],  4.677237E-6,    4.105546E-6
    elif parent == 'He3':
        return nuclides['He4'],  9.28775E-5,     3.4695E-5
    elif parent == 'H2':
        return nuclides['H3'],   8.413251E-6,    9.471512E-6
    else:
        return -1, 0.0, 0.0


def n_2n(nuclides, parent):
    if   parent == 'F19':
        return nuclides['F18'],  0.0,            0.04162
    elif parent == 'O17':
        return nuclides['O16'],  0.0,            0.066113
    elif parent == 'N14':
        return nuclides['N13'],  0.0,            0.006496
    elif parent == 'N15':
        return nuclides['N14'],  0.0,            0.112284
    elif parent == 'B11':
        return nuclides['B10'],  0.0,            0.018805
    elif parent == 'Be9':
        return nuclides['Be8'],  0.0205,         0.484483
    elif parent == 'Li7':
        return nuclides['Li6'],  0.0,            0.031603
    elif parent == 'H3':
        return nuclides['H2'],   0.0,            0.0497
    elif parent == 'H2':
        return nuclides['H1'],   0.0,            0.166767
    else:
        return -1, 0.0, 0.0

def n_alpha(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['N16'],nuclides['He4']],2.1667E-5,0.028393
    elif parent == 'O16':
        return [nuclides['C13'], nuclides['He4']], 0.0, 0.144515
    elif parent == 'O17':
        return [nuclides['C14'], nuclides['He4']],0.117316,0.260809
    elif parent == 'N14':
        return [nuclides['B11'], nuclides['He4']],0.104365,0.080516
    elif parent == 'N15':
            return [nuclides['B12'], nuclides['He4']], 0.0,0.069240
    elif parent == 'B10':
        return [nuclides['Li7'], nuclides['He4']],0.281082,0.044480
    elif parent == 'B11':
        return [nuclides['Li8'], nuclides['He4']], 0.0,0.031853
    else:
        return [-1,-1], 0.0, 0.0

def n_2alpha(nuclides, parent):
    if   parent == 'N14':
        return [nuclides['Li7'], nuclides['He4']],  0.0,0.031771
    elif parent == 'B10':
        return [nuclides['H3'],  nuclides['He4']],0.038439,0.095487
    else:
        return [-1,-1], 0.0, 0.0

def n_nalpha(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['N15'], nuclides['He4']], 0.0,0.3818
    elif parent == 'O17':
        return [nuclides['C13'], nuclides['He4']], 0.0,0.043420
    elif parent == 'N15':
        return [nuclides['B11'], nuclides['He4']], 0.0,0.012646
    elif parent == 'B11':
        return [nuclides['Li7'], nuclides['He4']], 0.0,0.286932
    elif parent == 'Be9':
        return [nuclides['He6'], nuclides['He4']], 0.0825,0.0104
    else:
        return [-1,-1], 0.0, 0.0

def n_2nalpha(nuclides, parent):
    if   parent == 'Li6':
        return [nuclides['H1'], nuclides['He4']],  0.0,0.0783
    elif parent == 'Li7':
        return [nuclides['H2'], nuclides['He4']],  0.0,0.020195
    else:
        return [-1,-1], 0.0, 0.0

def n_3nalpha(nuclides, parent):
    if   parent == 'Li7':
        return [nuclides['H1'], nuclides['He4']],  0.0,6.556330E-5
    else:
        return [-1,-1], 0.0, 0.0

def n_p(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['O19'],  nuclides['H1']], 0.0,0.018438
    elif parent == 'O16':
        return [nuclides['N16'],  nuclides['H1']], 0.0,0.042723
    elif parent == 'O17':
        return [nuclides['N17'],  nuclides['H1']], 0.0, 0.041838
    elif parent == 'N14':
        return [nuclides['C14'], nuclides['H1']],0.014102,0.043891
    elif parent == 'N15':
        return [nuclides['C15'],  nuclides['H1']], 0.0,0.019601
    elif parent == 'B10':
        return [nuclides['Be10'], nuclides['H1']],0.018860,0.034093
    elif parent == 'B11':
        return [nuclides['Be11'], nuclides['H1']], 0.0,0.005564
    elif parent == 'Li6':
        return [nuclides['He6'],  nuclides['H1']], 0.0,0.00604
    elif parent == 'He3':
        return [nuclides['H3'],nuclides['H1']],0.714941, 0.121
    else:
        return [-1,-1], 0.0, 0.0

def n_np(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['O18'],nuclides['H1']], 0.0, 0.061973
    elif parent == 'N15':
        return [nuclides['C14'],  nuclides['H1']], 0.0, 0.044827
    elif parent == 'B11':
        return [nuclides['Be10'], nuclides['H1']], 0.0, 0.001016
    else:
        return [-1,-1], 0.0, 0.0
    
def n_d(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['O18'], nuclides['H2']],  0.0, 0.022215
    elif parent == 'O16':
        return [nuclides['N15'], nuclides['H2']],  0.0,0.017623
    elif parent == 'O17':
        return [nuclides['N16'], nuclides['H2']],  0.0,0.020579
    elif parent == 'N14':
        return [nuclides['C13'], nuclides['H2']],  0.0, 0.042027
    elif parent == 'N15':
        return [nuclides['C14'], nuclides['H2']],  0.0,0.014926
    elif parent == 'B10':
        return [nuclides['Be9'], nuclides['H2']],  0.0, 0.031270
    elif parent == 'Li7':
        return [nuclides['He6'], nuclides['H2']],  0.0, 0.010199
    elif parent == 'He3':
        return [nuclides['H2'],  nuclides['H2']],  0.0,0.07609
    else:
        return [-1,-1], 0.0, 0.0
    
def n_t(nuclides, parent):
    if   parent == 'F19':
        return [nuclides['O17'], nuclides['H3']],  0.0,0.01303
    elif parent == 'N14':
        return [nuclides['C12'], nuclides['H3']],  0.0,0.028573
    elif parent == 'N15':
        return [nuclides['C13'], nuclides['H3']],  0.0,0.020163
    elif parent == 'B11':
        return [nuclides['Be9'], nuclides['H3']],  0.0,0.015172
    elif parent == 'Be9':
        return [nuclides['Li7'], nuclides['H3']],  0.0,0.020878
    elif parent == 'Li6':
        return [nuclides['He4'], nuclides['H3']],  0.206155,0.0258
    else:
        return [-1,-1], 0.0, 0.0




class DecayClass:
    def __init__(self):
        self.lambdas = -1  # Half life in seconds
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
    

def FindPotentialMatch(List,protons,A,Fraction,Excited):
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
        print("Could not find daughter in list of isotopes when expecting one")
        print("Looking for Protons : ",protons," Totat Nucleons : ",A," Fraction",Fraction)
        print("Close items are")
        for item in List:
            if protons in item[:-1] and A in item[:-1]:
                print(item)
        if (Fraction<4e-4):
            print("Small fraction, will let slide")
            Toreturn=''
        else:
            quit()
    
    return(Toreturn)

def DecayInfo(Nuclides,Nuclide_Names,parent,Decay_Conts,proton,neutrons,A):
    """
    This function will store and return decay information from
    the tape9.inp file
    """

    Info=DecayClass()
    Info.lambdas=Decay_Conts[Nuclides[parent]]
    
    with open('tape9.inp') as f:
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

    Info.FB=1-Info.FBX-Info.FPEC-Info.FA-Info.FIT-Info.FSF-Info.FN

    if(Info.FB<0 or Info.FB>1):
        print("Fraction of beta decays is too low or high : ",Info.FB)
        if abs(Info.FB)<1e-8:
            print("But I'll let it slide and set to zero")
            Info.FB=0
        else:
            print("I can't let this slide, not small enough")
            quit()
    
    return(Info)

    
def MakeAb2(phi,Nuclides,Nuclide_Names,Decay_Conts):

    # Create Activation and Decay Matrix and initial
    # nuclide quantity vector
    A = np.zeros((len(Nuclides),len(Nuclides)))

    #10^14 1/cm^2/s in 1/cm^2 /year
    phi = phi * 60 * 60 * 24 * 365.25 


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
        neutrons=str(int(Anum)-int(proton))

        print(row,isotope)
        row_decay =  DecayInfo(Nuclides,Nuclide_Names,isotope,Decay_Conts,proton,neutrons,Anum)
        # row_n_gamma =       n_gamma(nuclides, isotope)
        # row_n_2n =          n_2n(nuclides, isotope)
        # row_n_alpha =       n_alpha(nuclides, isotope)
        # row_n_2alpha =      n_2alpha(nuclides, isotope)
        # row_n_nalpha =      n_nalpha(nuclides, isotope)
        # row_n_2nalpha =     n_2nalpha(nuclides, isotope)
        # row_n_3nalpha =     n_3nalpha(nuclides, isotope)
        # row_n_p =           n_p(nuclides, isotope)
        # row_n_np =          n_np(nuclides, isotope)
        # row_n_d =           n_d(nuclides, isotope)
        # row_n_t =           n_t(nuclides, isotope)
        # row_lo_act_sum = row_n_gamma[1] + row_n_2n[1] +\
        #                  row_n_alpha[1] + row_n_2alpha[1] +\
        #                  row_n_nalpha[1] + row_n_2nalpha[1] + \
        #                  row_n_3nalpha[1] + row_n_p[1] +\
        #                  row_n_np[1] + row_n_d[1] +\
        #                  row_n_t[1]
        # row_hi_act_sum = row_n_gamma[2] + row_n_2n[2] +\
        #                  row_n_alpha[2] + row_n_2alpha[2] +\
        #                  row_n_nalpha[2] + row_n_2nalpha[2] + \
        #                  row_n_3nalpha[2] + row_n_p[2] +\
        #                  row_n_np[2] + row_n_d[2] +row_n_t[2]
        # try:
        #     if row_n_alpha[0] >= 0:
        #         print(row_n_alpha)
        #         donotuse=100
        #     continue
        # except TypeError:
        #     print(row_n_alpha)
        #     print(row_n_alpha[0][0])
        #     quit()
        
        # if row_betanegdecay[0] >= 0:
        #     # [days^-1]
        #     row_lambda = np.log(2)*60*60*24/row_betanegdecay[1] 
        # elif row_betaposdecay[0] >= 0:
        #     # [days^-1]
        #     row_lambda = np.log(2)*60*60*24/row_betaposdecay[1] 
        # elif row_2alphadecay[0] >= 0:
        #     # [days^-1]
        #     row_lambda = np.log(2)*60*60*24/row_2alphadecay[1] 
        # else:
        #     row_lambda = 0.0
    
        # # Diagonal Assignment
        # A[row,row] = -row_lambda - phi_lo*row_lo_act_sum -\
        #              phi_hi*row_hi_act_sum
        # # Off Diagonal Assignment
        # if row_betanegdecay[0] >= 0:
        #     A[row_betanegdecay[0],row] = np.log(2)*60*60*24/\
        #                                  row_betanegdecay[1]
        # if row_betaposdecay[0] >= 0:
        #     A[row_betaposdecay[0],row] = np.log(2)*60*60*24/\
        #                                  row_betaposdecay[1]
        # if row_2alphadecay[0] >= 0:
        #     A[row_2alphadecay[0],row] = np.log(2)*60*60*24/\
        #                                 row_2alphadecay[1]
        # if row_n_gamma[0] >= 0:
        #     A[row_n_gamma[0],row] = phi_lo*row_n_gamma[1] +\
        #                             phi_hi*row_n_gamma[2]
        # if row_n_2n[0] >= 0:
        #     A[row_n_2n[0],row] = phi_lo*row_n_2n[1] +\
        #                          phi_hi*row_n_2n[2]
        # if row_n_alpha[0][0] >= 0:
        #     for i in row_n_alpha[0]:
        #         A[i,row] = phi_lo*row_n_alpha[1] +\
        #                    phi_hi*row_n_alpha[2]
        # if row_n_2alpha[0][0] >= 0:
        #     for i in row_n_2alpha[0]:
        #         A[i,row] = phi_lo*row_n_2alpha[1] +\
        #                    phi_hi*row_n_2alpha[2]
        # if row_n_nalpha[0][0] >= 0:
        #     for i in row_n_nalpha[0]:
        #         A[i,row] = phi_lo*row_n_nalpha[1] +\
        #                    phi_hi*row_n_nalpha[2]
        # if row_n_2nalpha[0][0] >= 0:
        #     for i in row_n_2nalpha[0]:
        #         A[i,row] = phi_lo*row_n_2nalpha[1] +\
        #                    phi_hi*row_n_2nalpha[2]
        # if row_n_3nalpha[0][0] >= 0:
        #     for i in row_n_3nalpha[0]:
        #         A[i,row] = phi_lo*row_n_3nalpha[1] +\
        #                    phi_hi*row_n_3nalpha[2]
        # if row_n_p[0][0] >= 0:
        #     for i in row_n_p[0]:
        #         A[i,row] = phi_lo*row_n_p[1] + phi_hi*row_n_p[2]
        # if row_n_np[0][0] >= 0:
        #     for i in row_n_np[0]:
        #         A[i,row] = phi_lo*row_n_np[1] + phi_hi*row_n_np[2]
        # if row_n_d[0][0] >= 0:
        #     for i in row_n_d[0]:
        #         A[i,row] = phi_lo*row_n_d[1] + phi_hi*row_n_d[2]
        # if row_n_t[0][0] >= 0:
        #     for i in row_n_t[0]:
        #         A[i,row] = phi_lo*row_n_t[1] + phi_hi*row_n_t[2]

    
    b = np.zeros(len(nuclides))

    # N_0 expressed as kg nuclide per kg FLiBe
    #b[nuclides['F19']] = 0.7685
    #b[nuclides['Be9']] = 0.0911
    #b[nuclides['Li6']] = 0.01065636
    #b[nuclides['Li7']] = 0.12974364
    AtomsofFLiBe=6.0899894727155e24
    b[nuclides['F19']] = AtomsofFLiBe*4
    b[nuclides['Be9']] = AtomsofFLiBe*1
    b[nuclides['Li6']] = AtomsofFLiBe*2*0.0759
    b[nuclides['Li7']] = AtomsofFLiBe*2*0.9241

    return(A,b)

  
