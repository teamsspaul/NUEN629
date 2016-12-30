"""
Homework 4 Part 1
Step and Diamond Difference Solutions to 1D Transport Problem

methods author: Ryan G. McClarren
solution author: James B. Tompkins
created on 11/19/2015

We attempt to solve a problem with uniform source of Q=0.001, Sigma_t=Sigma_s=100
for slab of width 10 using both step and diamond difference discretizations
using 10, 50, and 100 zones (hx = 1,0.02,0.01) and a chosen angular quadrature.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

import SnMethods as SnM

Q = 0.01
Sigma_t = 100.
Sigma_s = Sigma_t

L = 10.
Is = (10, 50, 100)
N = 2
BCs = np.zeros(N)

# Initialize storage lists
x_list = []
phi_SI_dd_list = []
phi_SI_step_list = []
phi_GMRES_dd_list = []
phi_GMRES_step_list = []
iter_SI_dd_list = []
error_SI_dd_list = []
iter_SI_step_list = []
error_SI_step_list = []
iter_GMRES_dd_list = []
error_GMRES_dd_list = []
iter_GMRES_step_list = []
error_GMRES_step_list = []

for I in Is:
  hx = L/I
  q = np.ones(I)*Q
  Sig_t_discr = np.ones(I)*Sigma_t
  Sig_s_discr = np.ones(I)*Sigma_s

  # Source iteration Diamond Difference Solution
  x, phi_dd, iterations, errors = SnM.source_iteration(I,hx,q,Sig_t_discr,
                                   Sig_s_discr,N,BCs,"diamond_difference",
                                   tolerance=1.0e-8,maxits=10000,LOUD=True)
  x_list.append(x)
  phi_SI_dd_list.append(phi_dd)
  iter_SI_dd_list.append(iterations)
  error_SI_dd_list.append(errors)

  # Source iteration Step Solution
  x, phi_step, iterations, errors = SnM.source_iteration(I,hx,q,Sig_t_discr,
                                     Sig_s_discr,N,BCs,"step",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True)
  phi_SI_step_list.append(phi_step)
  iter_SI_step_list.append(iterations)
  error_SI_step_list.append(errors)

  # GMRES Diamond Difference Solution
  x, phi_dd, iterations, errors = SnM.gmres_solve(I,hx,q,Sig_t_discr,
                                     Sig_s_discr,N,BCs,"diamond_difference",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True,restart=I/2)
  phi_GMRES_dd_list.append(phi_dd)
  iter_GMRES_dd_list.append(iterations)
  error_GMRES_dd_list.append(errors)

  # GMRES Step Solution
  x, phi_step, iterations, errors = SnM.gmres_solve(I,hx,q,Sig_t_discr,
                                     Sig_s_discr,N,BCs,"step",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True,restart=I/2)
  phi_GMRES_step_list.append(phi_step)
  iter_GMRES_step_list.append(iterations)
  error_GMRES_step_list.append(errors)

print(iter_GMRES_step_list[0])
print(error_GMRES_step_list[0])

# Plot Source Iteration Solutions
plt.plot(x_list[0],phi_SI_step_list[0],label="10 zone Step Solution")
plt.plot(x_list[1],phi_SI_step_list[1],label="50 zone Step Solution")
plt.plot(x_list[2],phi_SI_step_list[2],label="100 zone Step Solution")
plt.plot(x_list[0],phi_SI_dd_list[0],label="10 zone DD Solution")
plt.plot(x_list[1],phi_SI_dd_list[1],label="50 zone DD Solution")
plt.plot(x_list[2],phi_SI_dd_list[2],label="100 zone DD Solution")
plt.yscale('log')
plt.legend()
plt.show()

# Plot Source Iteration and GMRES Step Solutions
plt.plot(x_list[0],phi_SI_step_list[0],label="10 zone SI Step Solution")
plt.plot(x_list[1],phi_SI_step_list[1],label="50 zone SI Step Solution")
plt.plot(x_list[2],phi_SI_step_list[2],label="100 zone SI Step Solution")
plt.plot(x_list[0],phi_GMRES_step_list[0],label="10 zone GMRES Step Solution")
plt.plot(x_list[1],phi_GMRES_step_list[1],label="50 zone GMRES Step Solution")
plt.plot(x_list[2],phi_GMRES_step_list[2],label="100 zone GMRES Step Solution")
plt.yscale('log')
plt.legend()
plt.show()

# Plot Source Iteration and GMRES Diamond Difference Solutions
plt.plot(x_list[0],phi_SI_dd_list[0],label="10 zone SI DD Solution")
plt.plot(x_list[1],phi_SI_dd_list[1],label="50 zone SI DD Solution")
plt.plot(x_list[2],phi_SI_dd_list[2],label="100 zone SI DD Solution")
plt.plot(x_list[0],phi_GMRES_dd_list[0],label="10 zone GMRES DD Solution")
plt.plot(x_list[1],phi_GMRES_dd_list[1],label="50 zone GMRES DD Solution")
plt.plot(x_list[2],phi_GMRES_dd_list[2],label="100 zone GMRES DD Solution")
plt.yscale('log')
plt.legend()
plt.show()

"""
Homework 4 Part 3
Error at Each Iteration for 1D Transport Problem Solutions

Plot error after each iteration using a 0 initial guess for the
step discretization with source iteration and GMRES.
"""

# Plot convergence for Step Methods
plt.plot(iter_SI_step_list[0],error_SI_step_list[0],label="10 zone SI Step Solution")
plt.plot(iter_SI_step_list[1],error_SI_step_list[1],label="50 zone SI Step Solution")
plt.plot(iter_SI_step_list[2],error_SI_step_list[2],label="100 zone SI Step Solution")
plt.plot(iter_GMRES_step_list[0],error_GMRES_step_list[0],label="10 zone GMRES Step Solution")
plt.plot(iter_GMRES_step_list[1],error_GMRES_step_list[1],label="50 zone GMRES Step Solution")
plt.plot(iter_GMRES_step_list[2],error_GMRES_step_list[2],label="100 zone GMRES Step Solution")
plt.yscale('log')
plt.legend()
plt.show()

# Plot convergence for Diamond Difference Methods
plt.plot(iter_SI_dd_list[0],error_SI_dd_list[0],label="10 zone SI DD Solution")
plt.plot(iter_SI_dd_list[1],error_SI_dd_list[1],label="50 zone SI DD Solution")
plt.plot(iter_SI_dd_list[2],error_SI_dd_list[2],label="100 zone SI DD Solution")
plt.plot(iter_GMRES_dd_list[0],error_GMRES_dd_list[0],label="10 zone GMRES DD Solution")
plt.plot(iter_GMRES_dd_list[1],error_GMRES_dd_list[1],label="50 zone GMRES DD Solution")
plt.plot(iter_GMRES_dd_list[2],error_GMRES_dd_list[2],label="100 zone GMRES DD Solution")
plt.yscale('log')
plt.legend()
plt.show()

# Plot convergence for GMRES Methods
plt.plot(iter_GMRES_step_list[0],error_GMRES_step_list[0],label="10 zone GMRES Step Solution")
plt.plot(iter_GMRES_step_list[1],error_GMRES_step_list[1],label="50 zone GMRES Step Solution")
plt.plot(iter_GMRES_step_list[2],error_GMRES_step_list[2],label="100 zone GMRES Step Solution")
plt.plot(iter_GMRES_dd_list[0],error_GMRES_dd_list[0],label="10 zone GMRES DD Solution")
plt.plot(iter_GMRES_dd_list[1],error_GMRES_dd_list[1],label="50 zone GMRES DD Solution")
plt.plot(iter_GMRES_dd_list[2],error_GMRES_dd_list[2],label="100 zone GMRES DD Solution")
plt.yscale('log')
plt.legend()
plt.show()
