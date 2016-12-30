"""
Homework 4 Part 4
Transport Solution to Reed's Problem

Solve Reed's problem. Present convergence plots for the solution in space and
angle to a "refined" solution in space and angle.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

import SnMethods as SnM

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

Is = (20, 50, 100, 200, 500)
L = 18.
Ns = (2, 4, 8, 12, 18)

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

for N in Ns:
  for I in Is:
    hx = L/I
    print(hx)
    xpos = hx/2.
    q = np.zeros(I)
    sigma_t = np.zeros(I)
    sigma_s = np.zeros(I)
    for i in range(I):
      sigma_t[i] = Sigma_t(xpos)
      sigma_s[i] = sigma_t[i]-Sigma_a(xpos)
      q[i] = Q(xpos)
      xpos += hx
    BCs = np.zeros(N)

    # Source iteration Diamond Difference Solution
    x, phi_dd, iterations, errors = SnM.source_iteration(I,hx,q,sigma_t,
                                   sigma_s,N,BCs,"diamond_difference",
                                   tolerance=1.0e-8,maxits=10000,LOUD=True)
    x_list.append(x)
    phi_SI_dd_list.append(phi_dd)
    iter_SI_dd_list.append(iterations)
    error_SI_dd_list.append(errors)

    # Source iteration Step Solution
    x, phi_step, iterations, errors = SnM.source_iteration(I,hx,q,sigma_t,
                                     sigma_s,N,BCs,"step",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True)
    phi_SI_step_list.append(phi_step)
    iter_SI_step_list.append(iterations)
    error_SI_step_list.append(errors)

    # GMRES Diamond Difference Solution
    x, phi_dd, iterations, errors = SnM.gmres_solve(I,hx,q,sigma_t,
                                     sigma_s,N,BCs,"diamond_difference",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True,restart=I/2)
    phi_GMRES_dd_list.append(phi_dd)
    iter_GMRES_dd_list.append(iterations)
    error_GMRES_dd_list.append(errors)

    # GMRES Step Solution
    x, phi_step, iterations, errors = SnM.gmres_solve(I,hx,q,sigma_t,
                                     sigma_s,N,BCs,"step",
                                     tolerance=1.0e-8,maxits=10000,LOUD=True,restart=I/2)
    phi_GMRES_step_list.append(phi_step)
    iter_GMRES_step_list.append(iterations)
    error_GMRES_step_list.append(errors)

# Plot spatial resolution solutions to SI DD Method
plt.plot(x_list[5],phi_SI_dd_list[5],label="20 zone 4 angle SI DD")
plt.plot(x_list[6],phi_SI_dd_list[6],label="50 zone 4 angle SI DD")
plt.plot(x_list[7],phi_SI_dd_list[7],label="100 zone 4 angle SI DD")
plt.plot(x_list[8],phi_SI_dd_list[8],label="200 zone 4 angle SI DD")
plt.plot(x_list[9],phi_SI_dd_list[9],label="500 zone 4 angle SI DD")
plt.legend()
plt.show()

# Plot angular resolution solutions to SI DD Method
plt.plot(x_list[1],phi_SI_dd_list[1],label="50 zone 2 angle SI DD")
plt.plot(x_list[6],phi_SI_dd_list[6],label="50 zone 4 angle SI DD")
plt.plot(x_list[11],phi_SI_dd_list[11],label="50 zone 8 angle SI DD")
plt.plot(x_list[16],phi_SI_dd_list[16],label="50 zone 12 angle SI DD")
plt.plot(x_list[21],phi_SI_dd_list[21],label="50 zone 18 angle SI DD")
plt.legend()
plt.show()

# Plot convergence for SI DD Method Solutions
"""
plt.plot(iter_SI_dd_list[0],error_SI_dd_list[0],label="20 zone 2 angle SI DD")
plt.plot(iter_SI_dd_list[1],error_SI_dd_list[1],label="50 zone 2 angle SI DD")
plt.plot(iter_SI_dd_list[2],error_SI_dd_list[2],label="100 zone 2 angle SI DD")
plt.plot(iter_SI_dd_list[3],error_SI_dd_list[3],label="200 zone 2 angle SI DD")
plt.plot(iter_SI_dd_list[4],error_SI_dd_list[4],label="500 zone 2 angle SI DD")
plt.plot(iter_SI_dd_list[5],error_SI_dd_list[5],label="20 zone 4 angle SI DD")
plt.plot(iter_SI_dd_list[6],error_SI_dd_list[6],label="50 zone 4 angle SI DD")
plt.plot(iter_SI_dd_list[7],error_SI_dd_list[7],label="100 zone 4 angle SI DD")
plt.plot(iter_SI_dd_list[8],error_SI_dd_list[8],label="200 zone 4 angle SI DD")
plt.plot(iter_SI_dd_list[9],error_SI_dd_list[9],label="500 zone 4 angle SI DD")
plt.plot(iter_SI_dd_list[10],error_SI_dd_list[10],label="20 zone 8 angle SI DD")
plt.plot(iter_SI_dd_list[11],error_SI_dd_list[11],label="50 zone 8 angle SI DD")
plt.plot(iter_SI_dd_list[12],error_SI_dd_list[12],label="100 zone 8 angle SI DD")
plt.plot(iter_SI_dd_list[13],error_SI_dd_list[13],label="200 zone 8 angle SI DD")
plt.plot(iter_SI_dd_list[14],error_SI_dd_list[14],label="500 zone 8 angle SI DD")
plt.plot(iter_SI_dd_list[15],error_SI_dd_list[15],label="20 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[16],error_SI_dd_list[16],label="50 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[17],error_SI_dd_list[17],label="100 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[18],error_SI_dd_list[18],label="200 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[19],error_SI_dd_list[19],label="500 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[20],error_SI_dd_list[20],label="20 zone 12 angle SI DD")
plt.plot(iter_SI_dd_list[21],error_SI_dd_list[21],label="50 zone 18 angle SI DD")
plt.plot(iter_SI_dd_list[22],error_SI_dd_list[22],label="100 zone 18 angle SI DD")
plt.plot(iter_SI_dd_list[23],error_SI_dd_list[23],label="200 zone 18 angle SI DD")
plt.plot(iter_SI_dd_list[24],error_SI_dd_list[24],label="500 zone 18 angle SI DD")
plt.yscale('log')
plt.legend()
plt.show()
"""

# Plot spatial resolution solutions to SI Step Method
plt.plot(x_list[5],phi_SI_step_list[5],label="20 zone 4 angle SI Step")
plt.plot(x_list[6],phi_SI_step_list[6],label="50 zone 4 angle SI Step")
plt.plot(x_list[7],phi_SI_step_list[7],label="100 zone 4 angle SI Step")
plt.plot(x_list[8],phi_SI_step_list[8],label="200 zone 4 angle SI Step")
plt.plot(x_list[9],phi_SI_step_list[9],label="500 zone 4 angle SI Step")
plt.legend()
plt.show()

# Plot angular resolution solutions to SI Step Method
plt.plot(x_list[1],phi_SI_step_list[1],label="50 zone 2 angle SI Step")
plt.plot(x_list[6],phi_SI_step_list[6],label="50 zone 4 angle SI Step")
plt.plot(x_list[11],phi_SI_step_list[11],label="50 zone 8 angle SI Step")
plt.plot(x_list[16],phi_SI_step_list[16],label="50 zone 12 angle SI Step")
plt.plot(x_list[21],phi_SI_step_list[21],label="50 zone 18 angle SI Step")
plt.legend()
plt.show()


# Plot spatial resoltuion solutions to GMRES DD Method
plt.plot(x_list[5],phi_GMRES_dd_list[5],label="20 zone 4 angle GMRES DD")
plt.plot(x_list[6],phi_GMRES_dd_list[6],label="50 zone 4 angle GMRES DD")
plt.plot(x_list[7],phi_GMRES_dd_list[7],label="100 zone 4 angle GMRES DD")
plt.plot(x_list[8],phi_GMRES_dd_list[8],label="200 zone 4 angle GMRES DD")
plt.plot(x_list[9],phi_GMRES_dd_list[9],label="500 zone 4 angle GMRES DD")
plt.legend()
plt.show()

# Plot angular resolution solutions to GMRES DD Method
plt.plot(x_list[1],phi_GMRES_dd_list[1],label="50 zone 2 angle GMRES DD")
plt.plot(x_list[6],phi_GMRES_dd_list[6],label="50 zone 4 angle GMRES DD")
plt.plot(x_list[11],phi_GMRES_dd_list[11],label="50 zone 8 angle GMRES DD")
plt.plot(x_list[16],phi_GMRES_dd_list[16],label="50 zone 12 angle GMRES DD")
plt.plot(x_list[21],phi_GMRES_dd_list[21],label="50 zone 18 angle GMRES DD")
plt.legend()
plt.show()


# Plot spatial resolution solutions to GMRES Step Method
plt.plot(x_list[5],phi_GMRES_step_list[5],label="20 zone 4 angle GMRES Step")
plt.plot(x_list[6],phi_GMRES_step_list[6],label="50 zone 4 angle GMRES Step")
plt.plot(x_list[7],phi_GMRES_step_list[7],label="100 zone 4 angle GMRES Step")
plt.plot(x_list[8],phi_GMRES_step_list[8],label="200 zone 4 angle GMRES Step")
plt.plot(x_list[9],phi_GMRES_step_list[9],label="500 zone 4 angle GMRES Step")
plt.legend()
plt.show()

# Plot angular resolution solutions to GMRES Step Method
plt.plot(x_list[1],phi_GMRES_step_list[1],label="50 zone 2 angle GMRES Step")
plt.plot(x_list[6],phi_GMRES_step_list[6],label="50 zone 4 angle GMRES Step")
plt.plot(x_list[11],phi_GMRES_step_list[11],label="50 zone 8 angle GMRES Step")
plt.plot(x_list[16],phi_GMRES_step_list[16],label="50 zone 12 angle GMRES Step")
plt.plot(x_list[21],phi_GMRES_step_list[21],label="50 zone 18 angle GMRES Step")
plt.legend()
plt.show()


# Plot convergence for various methods
plt.plot(iter_SI_dd_list[24],error_SI_dd_list[24],label="SI DD")
plt.plot(iter_SI_step_list[24],error_SI_step_list[24],label="SI Step")
plt.plot(iter_GMRES_dd_list[24],error_GMRES_dd_list[24],label="GMRES DD")
plt.plot(iter_GMRES_step_list[24],error_GMRES_step_list[24],label="GMRES Step")
plt.yscale('log')
plt.legend()
plt.show()
