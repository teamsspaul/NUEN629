"""
Discrete Ordinates Methods
author: Ryan McClarren
"""
import sys
import numpy as np


def dd_sweep1D(I,hx,q,sigma_t,mu,boundary):
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
      psi_right = (q[i]*0.5 + (mu*ihx-0.5*sigma_t[i])*psi_left)\
                  /(0.5*sigma_t[i] + mu*ihx)
      psi[i] = 0.5*(psi_right + psi_left)
      psi_left = psi_right
  else:
    psi_right = boundary
    for i in reversed(range(I)):
      psi_left = (q[i]*0.5+ (-mu*ihx-0.5*sigma_t[i])*psi_right)\
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
    for i in range(I):
      psi_right = (q[i]/2. + mu*ihx*psi_left)/(mu*ihx + sigma_t[i])
      psi[i] = 0.5*(psi_right + psi_left)
      psi_left = psi_right
  else:
    psi_right = boundary
    for i in reversed(range(I)):
      psi_left = (q[i]/2. - mu*ihx*psi_right)/(sigma_t[i] - mu*ihx)
      psi[i] = 0.5*(psi_right + psi_left)
      psi_right = psi_left
  return psi


def source_iteration(I,hx,q,sigma_t,sigma_s,N,BCs,sweep_type,
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
  
  while not(converged):
    phi = np.zeros(I)
    #sweep over each direction
    for n in range(N):
      if sweep_type == 'diamond_difference':
        tmp_psi = dd_sweep1D(I,hx,q + phi_old*sigma_s,sigma_t,MU[n],BCs[n])
      elif sweep_type == 'step':
        tmp_psi = step_sweep1D(I,hx,q + phi_old*sigma_s,sigma_t,MU[n],BCs[n])
      else:
        sys.exit("Sweep method specified not defined in SnMethods")
      phi += tmp_psi*W[n]
    #check convergence
    change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
    iterations.append(iteration)
    Errors.append(change)
    converged = (change < tolerance) or (iteration > maxits)
    if (LOUD>0) or (converged and LOUD<0):
      print("Iteration",iteration,": Relative Change =",change)
    if (iteration > maxits):
      print("Warning: Source Iteration did not converge")
    iteration += 1
    phi_old = phi.copy()
  x = np.linspace(hx/2,I*hx-hx/2,I)
  return x, phi, iterations, Errors


import scipy.sparse.linalg as spla
def gmres_solve(I,hx,q,sigma_t,sigma_s,N,BCs, sweep_type,
                tolerance = 1.0e-8,maxits = 100, LOUD=False, restart = 20 ):
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

  #compute left-hand side
  LHS = np.zeros(I)

  MU, W = np.polynomial.legendre.leggauss(N)
  for n in range(N):
    if sweep_type == 'diamond_difference':
      tmp_psi = dd_sweep1D(I,hx,q,sigma_t,MU[n],BCs[n])
    elif sweep_type == 'step':
      tmp_psi = step_sweep1D(I,hx,q,sigma_t,MU[n],BCs[n])
    #tmp_psi = sweep1D(I,hx,q,sigma_t,MU[n],BCs[n])
    LHS += tmp_psi*W[n]

  #define linear operator for gmres
  def linop(phi):
    tmp = phi*0
    #sweep over each direction
    for n in range(N):
      if sweep_type == 'diamond_difference':
        tmp_psi = dd_sweep1D(I,hx,phi*sigma_s,sigma_t,MU[n],BCs[n])
      elif sweep_type == 'step':
        tmp_psi = step_sweep1D(I,hx,phi*sigma_s,sigma_t,MU[n],BCs[n])
      #tmp_psi = sweep1D(I,hx,phi*sigma_s,sigma_t,MU[n],BCs[n])
      tmp += tmp_psi*W[n]
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

  #now call GMRES
  phi,info = spla.gmres(A,LHS,x0=LHS,restart=restart,
                        tol=tolerance,callback=callback)
  if (LOUD):
    print("Finished in",iteration[0],"iterations.")
  if (info >0):
    print("Warning, convergence not achieved")
  x = np.linspace(hx*.5,I*hx-hx*.5,I)
  return x, phi, iterations, Errors
