import numpy as np
import numpy.random as rand

def get_qoi(mu_v,mu_D,mu_w):
  v = rand.uniform(mu_v-0.1*mu_v,mu_v+0.1*mu_v)
  D = rand.uniform(mu_D-0.1*mu_D,mu_D+0.1*mu_D)
  w = rand.uniform(mu_w-0.1*mu_w,mu_w+0.1*mu_w)
  
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

#Getting sensitity of v
v_51 = get_qoi(0.51,0.126,0.09)
v_49 = get_qoi(0.49,0.124,0.11)

coeff_v = 0.5*abs(v_51-v_49)/0.02
ind_v = 0.1*abs(v_51-v_49)/0.02


coeff_D = 0.125*abs(v_51-v_49)/0.002
ind_D = 0.03*abs(v_51-v_49)/0.002

coeff_w = 0.1*abs(v_51-v_49)/0.02
ind_w = 0.05*abs(v_51-v_49)/0.02

print(coeff_v,ind_v, coeff_D, ind_D, coeff_w, ind_w)
