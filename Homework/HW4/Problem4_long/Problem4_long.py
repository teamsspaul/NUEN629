import numpy as np
from scipy.stats import norm
import pyDOE

def frm(x, b):
#    """
#    Converts given number x, from base 10 to base b 
#    x -- the number in base 10
#    b -- base to convert
#    """
    assert(x >= 0)
    assert(1< b < 37)
    r = ''
    import string
    while x > 0:
        r = string.printable[x % b] + r
        x //= b
    return r

def vandercorp(N,base):
  samples = np.empty(N)
  for i in range(0,N):
    n = frm(i+1,base)
    reflect = n[::-1]
    samples[i] = 0
    for j in range(0,len(reflect)):
      samples[i] += int(reflect[j])*pow(base,-j-1)
  
  return samples

def get_qoi(v,D,w):
  
  #X and T
  x = np.linspace(0,10,401)
  t = np.linspace(0,5,41)
  
  dx = x[1] - x[0]
  dt = t[1] - t[0]
  
  #i-1
  a = (-v/dx - D/pow(dx,2))
  #i
  b = (1/dt)+(v/dx) + 2*D/pow(dx,2) + w
  #i+1
  c = (-D/pow(dx,2))
  
  A = np.zeros([len(x),len(x)])
  
  for i in range(0,len(x)):
    if (i == 0):
        A[i][i] = b
        A[i][1] = c
    elif(i==(len(x)-1)):
      A[i][i-1] = a
      A[i][i] = b
    else:  
      A[i][i-1] = a
      A[i][i] = b
      A[i][i+1] = c
  
  #Boundary conditions
  A[0][len(x)-1] = A[0][0]
  A[len(x)-1][0] = A[len(x)-1][len(x)-1]
  
  
  u_t = np.zeros([len(x),len(t)],1/dt)
  u_t[:,0] = 1
  boundary_index = np.where(x == 2.5)
  boundary_index = int(boundary_index[0])
  u_t[boundary_index+1:len(x),0] = 0
  
  for i in range(1,len(t)):
    u_t[:,i] = np.linalg.solve(A,u_t[:,i-1]*(1/dt))
    
  x_low = int(np.where(x == 5)[0])
  x_high = int(np.where(x == 6)[0])
  
  t_low = 0
  t_high = int(np.where(t == 5)[0])
  
  r = u_t[x_low:x_high+1,t_low:t_high+1]
  
  
  integral_1 = np.empty(len(t))
  for i in range(t_low,t_high+1):
    integral_1[i] = np.trapz(r[:,i],x[x_low:x_high+1])
    
  integral = np.trapz(integral_1,t[t_low:t_high+1])
  integral = integral*w
  return integral


cube = pyDOE.lhs(3,50)
N = 50
greater_than = 0
for i in range(0,N):
  x_v = cube[i][0]
  x_D = cube[i][1]
  x_w = cube[i][2]
  
  #inverting to get a v
  v = norm.ppf(x_v,0.5,0.1)
  D = norm.ppf(x_D,0.125,0.03)
  w = norm.ppf(x_w,0.1,0.05)
  
  f = get_qoi(v,D,w)
  
  if f > 0.035:
    greater_than += 1

prob_lhs = float(greater_than/N)
print('Prob_lhs: ' ,prob_lhs)


base_2 = vandercorp(N,2)
base_3 = vandercorp(N,3)
base_5 = vandercorp(N,5)

halton = np.array([base_2,base_3,base_5]).T
greater_than = 0
for i in range(0,N):
  x_v = halton[i][0]
  x_D = halton[i][1]
  x_w = halton[i][2]
  
  #inverting to get a v
  v = norm.ppf(x_v,0.5,0.1)
  D = norm.ppf(x_D,0.125,0.03)
  w = norm.ppf(x_w,0.1,0.05)
  
  f = get_qoi(v,D,w)
  
  if f > 0.035:
    greater_than += 1

prob_halton = float(greater_than/N)
print('Prob_halton: ',prob_halton)

