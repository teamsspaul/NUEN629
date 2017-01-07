#!/usr/bin/env python3

################################################################
##################### Import packages ##########################
################################################################

import sys
import numpy as np
import scipy.sparse.linalg as spla

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
XScaleE='log'      # Same but for error plot

YFontSize=18                    # Y label font size
YFontWeight="normal"            # "bold" or "normal"
YScale="linear"                 # 'linear' or 'log'
YScaleE='log'

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

Xlabel='z position [cm]'
Ylabel="$\phi\left[\\frac{n\cdot cm}{cm^3\cdot s}\\right]$"

XlabelE='Iterations'
YlabelE="Error = $\\frac{||\phi^{\ell+1}-\phi^\ell||}{||\phi^{\ell+1}||}$"

################################################################
######################### Functions ############################
################################################################

def Sigma_tReed(r):
    value = 0 + ((1.0*(r>=14) + 1.0*(r<=4)) +
                 5.0 *((np.abs(r-11.5)<0.5) or (np.abs(r-6.5)<0.5)) +
                 50.0 * (np.abs(r-9)<=2) )
    return value;
def Sigma_aReed(r):
    value = 0 + (0.1*(r>=14) + 0.1*(r<=4) +
                 5.0 *((np.abs(r-11.5)<0.5) or (np.abs(r-6.5)<0.5)) +
                 50.0 * (np.abs(r-9)<=2) )
    return value;
def QReed(r):
    value = 0 + 1.0*((r<16) * (r>14))+ 1.0*((r>2) * (r<4)) + 50.0*(np.abs(r-9)<=2)
    return value;

def Timevector(T,dt):
    Time=[dt]
    while Time[-1]<T:
        Time.append(Time[-1]+dt)
    return(Time)

def diamond_sweep1D(I,hx,q,sigma_t,mu,boundary):
  """Compute a transport diamond difference sweep for a given
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
  ihx = 1./hx
  if (mu > 0):
    psi_left = boundary
    for i in range(I):
      psi_right = (q[i] + (mu*ihx-0.5*sigma_t[i])*psi_left)\
                  /(0.5*sigma_t[i] + mu*ihx)
      psi[i] = 0.5*(psi_right + psi_left)
      psi_left = psi_right
  else:
    psi_right = boundary
    for i in reversed(range(I)):
      psi_left = (q[i]+ (-mu*ihx-0.5*sigma_t[i])*psi_right)\
                 /(0.5*sigma_t[i] - mu*ihx)
      psi[i] = 0.5*(psi_right + psi_left)
      psi_right = psi_left
  return psi

def step_sweep1D(I,hx,q,sigma_t,mu,boundary):
  """Compute a transport step sweep for a given
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
  ihx = 1./hx
  if (mu > 0):
    psi_left = boundary
    psi[0] = 0
    for i in range(1,I):
      psi_right = (q[i] + mu*ihx*psi_left)/(mu*ihx + sigma_t[i])
      psi[i] = 0.5*(psi_right + psi_left)
      psi_left = psi_right
  else:
    psi_right = boundary
    psi[-1] = 0
    for i in reversed(range(0,I-1)):
      psi_left = (q[i] - mu*ihx*psi_right)/(sigma_t[i] - mu*ihx)
      psi[i] = 0.5*(psi_right + psi_left)
      psi_right = psi_left
  return psi


def source_iteration(I,hx,q,sigma_t,sigma_s,N,psiprevioustime,
                     v,dt,Time,BCs,sweep_type,
                     tolerance = 1.0e-8,maxits = 100, LOUD=False ):
  """Perform source iteration for single-group steady state problem
  Inputs:
    I:               number of zones
    hx:              size of each zone
    q:               source array
    sigma_t:         array of total cross-sections
    sigma_s:         array of scattering cross-sections
    N:               number of angles
    BCs:             Boundary conditions for each angle
    sweep_type:      type of 1D sweep to perform solution
    tolerance:       the relative convergence tolerance for the iterations
    maxits:          the maximum number of iterations
    LOUD:            boolean to print out iteration stats
  Outputs:
    x:               value of center of each zone
    phi:             value of scalar flux in each zone
  """
  iterations = []
  Errors = []
  phi = np.zeros(I)
  phi_old = phi.copy()
  converged = False
  MU, W = np.polynomial.legendre.leggauss(N)
  iteration = 1
  tmp_psi=psiprevioustime.copy()
  if len(Time)==1:
      sigma_ts=sigma_t
  else:
      sigma_ts=sigma_t+1/(v*dt)

  while not(converged):
    phi = np.zeros(I)
    #sweep over each direction   
    for n in range(N):
      #qs=(q*W[n])/2+(phi_old*sigma_s)/2+psiprevioustime[n,:]/(v*dt)
      qs=(q)/2+(phi_old*sigma_s)/2+psiprevioustime[n,:]/(v*dt) 
      if sweep_type == 'dd':
        tmp_psi[n,:] = diamond_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
      elif sweep_type == 'step':
        tmp_psi[n,:] = step_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
      else:
        sys.exit("Sweep method specified not defined in SnMethods")
      phi = phi+tmp_psi[n,:]*W[n]
    #check convergence
    change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
    iterations.append(iteration)
    Errors.append(change)
    #iterations.append(iteration)
    #Errors.append(change)
    converged = (change < tolerance) or (iteration > maxits)
    if (LOUD>0) or (converged and LOUD<0):
      print("Iteration",iteration,": Relative Change =",change)
    if (iteration > maxits):
      print("Warning: Source Iteration did not converge : "+\
            sweep_type+", I : "+str(I)+", Diff : %.2e" % change)
    #Prepare for next iteration
    iteration += 1
    phi_old = phi.copy()
  if sweep_type == 'step':
      x = np.linspace(0,(I-1)*hx,I)
  elif sweep_type == 'dd':
      x = np.linspace(hx/2,I*hx-hx/2,I)
  return x, phi, iterations, Errors, tmp_psi


def gmres_solve(I,hx,q,sigma_t,sigma_s,N,psiprevioustime,
                v,dt,Time,BCs, sweep_type,
                tolerance = 1.0e-8,maxits = 100, LOUD=False,
                restart = 20 ):
  """Solve, via GMRES, a single-group steady state problem
  Inputs:
    I:               number of zones
    hx:              size of each zone
    q:               source array
    sigma_t:         array of total cross-sections
    sigma_s:         array of scattering cross-sections
    N:               number of angles
    BCs:             Boundary conditions for each angle
    sweep_type:      type of 1D sweep to perform solution
    tolerance:       the relative convergence tolerance for the iterations
    maxits:          the maximum number of iterations
    LOUD:            boolean to print out iteration stats
  Outputs:
    x:               value of center of each zone
    phi:             value of scalar flux in each zone
  """
  iterations = []
  Errors = []

  #compute RHS side
  RHS = np.zeros(I)

  MU, W = np.polynomial.legendre.leggauss(N)
  tmp_psi=psiprevioustime.copy()
  if len(Time)==1:
      sigma_ts=sigma_t
  else:
      sigma_ts=sigma_t+1/(v*dt)
      
  for n in range(N):
    qs=q/2+psiprevioustime[n,:]/(v*dt)
    if sweep_type == 'dd':
      tmp_psi[n,:] = diamond_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
    elif sweep_type == 'step':
      tmp_psi[n,:] = step_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
    #tmp_psi = sweep1D(I,hx,q,sigma_t,MU[n],BCs[n])
    RHS += tmp_psi[n,:]*W[n]

  #define linear operator for gmres
  def linop(phi):
    tmp = phi*0
    #sweep over each direction
    for n in range(N):
      if sweep_type == 'dd':
        tmp_psi[n,:] = diamond_sweep1D(I,hx,(phi*sigma_s)/2,
                                  sigma_ts,MU[n],BCs[n])
      elif sweep_type == 'step':
        tmp_psi[n,:] = step_sweep1D(I,hx,(phi*sigma_s)/2,
                                    sigma_ts,MU[n],BCs[n])
      tmp += tmp_psi[n,:]*W[n]
    return phi-tmp
  A = spla.LinearOperator((I,I), matvec = linop, dtype='d')

  
  #define a little function to call when the iteration is called
  iteration = np.zeros(1)
  def callback(rk, iteration=iteration):
    iteration += 1
    if (LOUD>0):
      print("Iteration",iteration[0],"norm of residual",np.linalg.norm(rk))
    iterations.append(iteration[0])
    Errors.append(np.linalg.norm(rk))

  #Do the GMRES Solve
  phi,info = spla.gmres(A,RHS,x0=RHS,tol=tolerance,
                        restart=int(restart),callback=callback)

  #Print important information
  if (LOUD):
    print("Finished in",iteration[0],"iterations.")
  if (info >0):
    print("Warning, convergence not achieved :"+str(sweep_type)+" "+str(hx))
  if sweep_type == 'step':
      x = np.linspace(0,(I-1)*hx,I)
  elif sweep_type == 'dd':
      x = np.linspace(hx/2,I*hx-hx/2,I)

  #Calculate Psi for time iterations
  phi2 = np.zeros(I)
  #sweep over each direction   
  for n in range(N):
      #qs=(q*W[n])/2+(phi_old*sigma_s)/2+psiprevioustime[n,:]/(v*dt)
      qs=(q)/2+(phi*sigma_s)/2+psiprevioustime[n,:]/(v*dt) 
      if sweep_type == 'dd':
          tmp_psi[n,:] = diamond_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
      elif sweep_type == 'step':
          tmp_psi[n,:] = step_sweep1D(I,hx,qs,sigma_ts,MU[n],BCs[n])
      else:
          sys.exit("Sweep method specified not defined in SnMethods")
      phi2 = phi2+tmp_psi[n,:]*W[n]

  return x, phi, iterations, Errors,tmp_psi

def solver(I,hx,q,Sig_t,Sig_s,N,psi,v,dt,Time,BCs,Scheme,tol,MAXITS,loud):
    Method=Scheme.split(':')[1]
    if "Iteration" in Scheme:
        x, phi, iterations, errors, psi =source_iteration(I,
            hx,q,Sig_t,Sig_s,N,psi,v,dt,Time,BCs,
            Method,tolerance=tol,maxits=MAXITS,LOUD=loud)
    elif "GMRES" in Scheme:
        x, phi, iterations, errors, psi =gmres_solve(I,
            hx,q,Sig_t,Sig_s,N,psi,v,dt,Time,BCs,
            Method,tolerance=tol,maxits=MAXITS,LOUD=loud,restart=MAXITS)
    else:
        print("Improper sweep selected")
        quit()
    return x, phi, iterations, errors,psi

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

def plot(x,y,ax,label,fig,Check,NumOfPoints):
    if len(x)>300:
        x=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
    #Plot X and Y
    ax.plot(x,y,
            linestyle=loop_values(LineStyles,Check),
            marker=loop_values(MarkerType,Check),
            color=loop_values(Colors,Check),
            markersize=loop_values(MarkSize,Check),
            alpha=loop_values(Alpha_Value,Check),
            label=label)
    
    #Log or linear scale?
    ax.set_xscale(XScale)
    ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,
                                                          ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    ax.set_ylabel(Ylabel,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)
    return(ax,fig)                                    


def plotE(x,y,erax,label,erfig,Check,NumOfPoints):
    if len(x)>300:
        x=reduceList(x,NumOfPoints)
        y=reduceList(y,NumOfPoints)
    #Plot X and Y
    erax.plot(x,y,
            linestyle=loop_values(LineStyles,Check),
            marker=loop_values(MarkerType,Check),
            color=loop_values(Colors,Check),
            markersize=loop_values(MarkSize,Check),
            alpha=loop_values(Alpha_Value,Check),
            label=label)
    
    #Log or linear scale?
    erax.set_xscale(XScaleE)
    erax.set_yscale(YScaleE)
    #Set Title
    erfig.suptitle(Title,fontsize=TitleFontSize,
                 fontweight=TitleFontWeight,fontdict=font,
                                                          ha='center')
    #Set X and y labels
    erax.set_xlabel(XlabelE,
                  fontsize=XFontSize,fontweight=XFontWeight,
                  fontdict=font)
    erax.set_ylabel(YlabelE,
                  fontsize=YFontSize,
                  fontweight=YFontWeight,
                  fontdict=font)
    return(erax,erfig)                                    

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


################################################################
#################### Functions Making ##########################
################################################################

def MatExp(A,n0,t,maxits,tolerance=1e-12,LOUD=False):

  converged = False
  m=0
  sum_old=n0.copy()*0
  
  while not(converged):

    #Upgrade so A is a matrix multiply (keep t)
    sum=sum_old+(1/np.math.factorial(m))*(A**m)*(t**m)*n0

    #Avoid dividing by zero
    if sum==0: m+=1;sum_old=sum.copy();continue
    
    change = np.linalg.norm(sum-sum_old)/np.linalg.norm(sum)    
    converged = (change < tolerance) or (m > maxits)
    
    if (LOUD>0) or (converged and LOUD<0):
      print("Iteration",m,": Relative Change =",change)
    if (m > maxits):
      print("Warning: Source Iteration did not converge : "+\
            ", m : "+str(m)+", Diff : %.2e" % change)
    #Prepare for next iteration
    m += 1
    sum_old = sum.copy()

  return sum

def BackEuler():
