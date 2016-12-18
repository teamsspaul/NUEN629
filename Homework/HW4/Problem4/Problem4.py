from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
import lhsmdu










#Latin Hypercube sampling

l = lhsmdu.sample(2,50)

c = (beta.ppf((l[0]),3,2))
d = (beta.ppf((l[1]),1.1,2))

LessThan10 = 0
for i in range(0,50):
    x = 2*c[0,i] - 1
    y = 2*d[0,i] - 1
    F = (1-x)**2 +100*(y-x**2)**2
    if F < 10.:
        LessThan10 += 1.

LessThan10 = LessThan10/50

print ("Hypercube probability =", LessThan10)





# Halton is basically the same as Van Der Corputs but in more dimensions so....
def vdc(n, base=2):
    vdc, denom = 0,1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

    
N=7
    
# This creates an NxN (49 point) matrix from a Halton sampling
a=[]
b=[]
for i in range(1,N+1):
    a.append (beta.ppf((vdc(i, base=2)),3,2))
    b.append (beta.ppf((vdc(i, base=2)),1.1,2))
    
t = beta.rvs(3, 2, size=100000)
s = beta.rvs(1.1, 2, size=100000)




LessThan10 = 0
for i in range(1,N):
    for j in range(1,N):
        x = 2*a[i] - 1
        y = 2*b[j] - 1
        F = (1-x)**2 +100*(y-x**2)**2
        if F < 10.:
            LessThan10 += 1.

LessThan10 = LessThan10/N**2

print ("Halton probability =", LessThan10)

LessThan10 = 0
for i in range(1,100000):
    x = 2*t[i] - 1
    y = 2*s[i] - 1
    F = (1-x)**2 +100*(y-x**2)**2
    if F < 10.:
        LessThan10 += 1.
LessThan10 = LessThan10/100000
print ("10^5 sample probability =", LessThan10)


plt.hist(t, bins=np.arange(51)/50) 
plt.title("Histogram of random data")
plt.savefig('C:\\PythonScripts\\UQhomework\\HW3P4Test1')
plt.clf()

plt.hist(s, bins=np.arange(51)/50) 
plt.title("Histogram of random data")
plt.savefig('C:\\PythonScripts\\UQhomework\\HW3P4Test2')
plt.clf()

plt.hist(c[0], bins=np.arange(11)/10) 
plt.title("Histogram of random data")
plt.savefig('C:\\PythonScripts\\UQhomework\\HW3P4Test3')
plt.clf()

plt.hist(d[0], bins=np.arange(11)/10) 
plt.title("Histogram of random data")
plt.savefig('C:\\PythonScripts\\UQhomework\\HW3P4Test4')
plt.clf()

