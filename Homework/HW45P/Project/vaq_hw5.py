
# coding: utf-8

## Approximation to Exponential Using Parabolic Contour

# In[93]:

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt

#compute exp
Npoints = (2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32)
errors = np.zeros(len(Npoints))
pos = 0
for N in Npoints:
    theta = np.pi*np.arange(1,N,2)/N
    z = N*(.1309 - 0.1194*theta**2 + .2500j*theta)
    w = N*(- 2*0.1194*theta + .2500j)
    c = 1.0j/N*np.exp(z)*w
    plt.plot(np.real(z),np.imag(z),'o-')
    u = 0
    for k in range(int(N/2)):
        u -=  c[k]/((z[k] + 1))
    errors[pos] = np.abs(2*np.real(u)-0.3678794411714423215955238)
    pos += 1
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.axis([-40,10,0,30])
plt.savefig("6.pdf");
plt.close()
print("Errors =",errors)



## Depletion Example

# In[84]:

#cross-sections in barns
position = {'28':0,'29':1,'20':2,'39':3,'30':4,'49':5,'40':6,'41':7,'51':8}
previous_cap = {'28':-1,'29':0,'20':1,'39':-1,'30':3,'49':-1,'40':5,'41':6,'51':-1}
previous_beta = {'28':-1,'29':-1,'20':-1,'39':1,'30':2,'49':3,'40':4,'41':-1,'51':7}
sig_gamma = 1.0E-24*np.array([2.7,22.0,0,60,274,290,326,532])
sig_a = 1.0E-24*np.array([12.0,22,0,60,0,274+698,290+53,326+938,535])
lam = np.array([0,42.4737,1.17982,0.294956,138.629,0,0,0.000131877,0])

A = np.zeros((9,9))

phi = 1.0e14 * 60 * 60 * 24 #10^14 1/cm^2/s in 1/cm^2/day
for i in position:
    row = position[i]
    A[row,row] = -lam[row] - phi*sig_a[row]
    if previous_cap[i]>=0:
            A[row,previous_cap[i]] = phi*sig_gamma[previous_cap[i]]
    if previous_beta[i]>=0:
            A[row,previous_beta[i]] = lam[previous_beta[i]]
plt.spy(A)
plt.show()
b = np.zeros(9)
b[0] = 1.0
#plt.savefig("98.pdf");
#plt.close()



### Example Matrix Exponential

# In[89]:

Npoints = (32,) #(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32)

for N in Npoints:
    pos = 0
    theta = np.pi*np.arange(1,N,2)/N
    z = N*(.1309 - 0.1194*theta**2 + .2500j*theta)
    w = N*(- 2*0.1194*theta + .2500j)
    c = 1.0j/N*np.exp(z)*w
    #plt.plot(np.real(z),np.imag(z),'o-')
    #plt.savefig('hw5.pdf')
    #plt.close()
    u = np.zeros(9)
    for k in range(int(N/2)):
        n,code = splinalg.gmres(z[k]*sparse.identity(9)  - A*365,b, tol=1e-12, maxiter=2000)
        if (code):
            print(code)
        u = u- c[k]*n
    u = 2*np.real(u)
    print(u)


## HW 5, prob 2
isotopes = {'H3':0, 'He3':1, 'He4':2, 'He6':3, 'Li6':4, 'Li7':5, 'Li8':6, 'Li9':7,
            'Be8':8, 'Be9':9, 'Be10':10, 'B10':11}

def decay(isotopes, tope):
    return None

'''
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
"""Different Nuclear Reactions
Inputs:
    isotopes:        python dictionary containing matrix indices for isotopes
    tope:            nucleus that undergoes some nuclear reaction
Outputs:
    tope:            new nucleus created from nulcear reaction
    value:           either half-life [days]
                     or cross sections for 2.45 MeV and 14.1 MeV [barns]
"""

def betadecay(isotopes, tope):
    if   tope == 'F18':  return isotopes['O18'],  1.8295 #hours

    elif tope == 'H3':   return isotopes['He3'],  4500 #days
    elif tope == 'He6':  return isotopes['Li6'],  0.8067 #sec
    elif tope == 'Li8':  return isotopes['Be8'],  0.8403 #sec
    elif tope == 'Li9':  return isotopes['Be9'],  0.1783 #sec
    elif tope == 'Be10': return isotopes['B10'],  1387000 #years
    else: return -1, 0

def capture(isotopes, tope):
    if   tope == 'F19':  return isotopes['F20'],  8.649107E-5, 3.495035E-5

    else: return -1, 0

def n_alpha(isotopes, tope):

    else: return -1, 0

def n_p(isotopes, tope):
    if   tope == 'F19':  return isotopes['O19'],  0.0, 0.018438

    if   tope == 'Li6':  return isotopes['He6'],  0,         0.00604
    elif tope == 'Be9':  return isotopes['Li9'],  0,         0
    else: return -1, 0

def n_d(isotopes, tope):
    if   tope == 'F19':  return isotopes['O18'],  0.0, 0.022215

    if   tope == 'Li7':  return isotopes['He6'],  0,         0.010024
    elif tope == 'Be9':  return isotopes['Li8'],  0,         0
    else: return -1, 0

def n_t(isotopes, tope):
    if   tope == 'Be9':  return isotopes['Li7'],  0,         0.020327
    else: return -1, 0

def n_2n(isotopes, tope):
    if   tope == 'F19':  return isotopes['F18'],  0.0, 0.04162

    if   tope == 'Li7':  return isotopes['Li6'],  0,         0.031743
    elif tope == 'Be9':  return isotopes['Be8'],  0.0205,    0.485944
    else: return -1, 0

def n_nalpha(isotopes, tope):
    if   tope == 'Li7':  return isotopes['H3'],   0,         0.302149
    else: return -1, 0

def n_np(isotopes, tope):
    if   tope == 'F19':  return isotopes['O18'],  0.0, 0.061973

def n_nd(isotopes, tope):
    if   tope == 'Li6':  return isotopes['He4'],  0.05948,   0.473592
    else: return -1, 0
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


print(capture(isotopes,'H3'))
for tope in isotopes:
    print(isotopes[tope])
'''
