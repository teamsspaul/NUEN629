import numpy as np
import matplotlib.pyplot as plt


#Here is a diamond difference sweep
#
def sweep1D(I,hx,q,sigma_t,mu,boundary):
    """Compute a transport sweep for a given
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
    assert(np.abs(mu) > 1e-10)   #Make sure that the angle is not zero or really small
    psi = np.zeros(I)
    ihx = 1/hx
    if (mu > 0): 
        psi_left = boundary
        for i in range(I):
            psi_right = (q[i]*0.5 + mu*psi_left*ihx)/(sigma_t[i] + mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_left = psi_right
    else:
        psi_right = boundary
        for i in reversed(range(I)):
            psi_left = (q[i]*0.5 - mu*psi_right*ihx)/(sigma_t[i] - mu*ihx)
            psi[i] = 0.5*(psi_right + psi_left)
            psi_right = psi_left
    return psi



