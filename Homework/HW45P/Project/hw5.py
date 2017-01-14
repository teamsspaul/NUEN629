import sys
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
from math import log

nuclides = {'H1':0, 'H2':1, 'H3':2, 'He3':3, 'He4':4, 'He6':5, 'Li6':6,
            'Li7':7, 'Li8':8, 'Be8':9, 'Be9':10, 'Be10':11,
            'Be11':12, 'B10':13, 'B11':14, 'B12':15, 'C12':16, 'C13':17,
            'C14':18, 'C15':19, 'N13':20, 'N14':21, 'N15':22, 'N16':23,
            'N17':24, 'O16':25, 'O17':26, 'O18':27, 'O19':28, 'F18':29,
            'F19':30, 'F20':31, 'Ne20':32}
nuclide_names = ('H1', 'H2', 'H3', 'He3', 'He4', 'He6', 'Li6',
            'Li7', 'Li8', 'Be8', 'Be9', 'Be10',
            'Be11', 'B10', 'B11', 'B12', 'C12', 'C13',
            'C14', 'C15', 'N13', 'N14', 'N15', 'N16',
            'N17', 'O16', 'O17', 'O18', 'O19', 'F18',
            'F19', 'F20', 'Ne20')

atom_mass = np.array([1.007825032, 2.014101778, 3.0160492779, 3.016029320,
             4.002603254, 6., 6.015122887, 7.0160034366, 8., 8. ,9.012183065,
             10., 11., 10.01293695, 11.00930536, 12., 12., 13.003354835,
             14.003241988, 15., 13., 14.003074004, 15.000108898, 16.,
             17., 15.994914619, 16.999131756, 17.999159612, 19.,
             18., 18.998403162, 20., 19.992440176])

"""Interaction functions
   @ In, nuclides:  dictionary with isotope keywords and 
                    corresponding indices
   @ In, parent:    parent nuclides undergoing a decay or interaction
   @Out, value:     new value in interaction matrix, either a half life
                    [secs] or 2.45 MeV and 14.1 MeV cross sections 
                    [barns]
"""

decay_consts = np.array([0., 0., log(2)/3.887896E8, 0., 0., log(2)/0.807, 0.,
            0., log(2)/0.840, log(2)/7.0E-17, 0., log(2)/4.73364E13, log(2)/13.8,
            0., 0., log(2)/0.0202, 0., 0., log(2)/1.803517E11, log(2)/2.45,
            log(2)/598.2, 0., 0., log(2)/7.13, log(2)/4.174, 0., 0., 0.,
            log(2)/26.9, log(2)/6586.2, 0., log(2)/11.1, 0.])

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
    if   parent == 'F19':  return nuclides['F20'],  8.649107E-5,    3.495035E-5
    elif parent == 'O16':  return nuclides['O17'],  1.0E-4,         1.0E-4
    elif parent == 'O17':  return nuclides['O18'],  2.2675E-4,      2.087114E-4
    elif parent == 'N14':  return nuclides['N15'],  2.397479E-5,    1.679535E-5
    elif parent == 'N15':  return nuclides['N16'],  8.121795E-6,    8.56E-6
    elif parent == 'Be9':  return nuclides['Be10'], 1.943574E-6,    1.660517E-6
    elif parent == 'Li6':  return nuclides['Li7'],  1.106851E-5,    1.017047E-5
    elif parent == 'Li7':  return nuclides['Li8'],  4.677237E-6,    4.105546E-6
    elif parent == 'He3':  return nuclides['He4'],  9.28775E-5,     3.4695E-5
    elif parent == 'H2':   return nuclides['H3'],   8.413251E-6,    9.471512E-6
    else: return -1, 0.0, 0.0

def n_2n(nuclides, parent):
    if   parent == 'F19':  return nuclides['F18'],  0.0,            0.04162
    elif parent == 'O17':  return nuclides['O16'],  0.0,            0.066113
    elif parent == 'N14':  return nuclides['N13'],  0.0,            0.006496
    elif parent == 'N15':  return nuclides['N14'],  0.0,            0.112284
    elif parent == 'B11':  return nuclides['B10'],  0.0,            0.018805
    elif parent == 'Be9':  return nuclides['Be8'],  0.0205,         0.484483
    elif parent == 'Li7':  return nuclides['Li6'],  0.0,            0.031603
    elif parent == 'H3':   return nuclides['H2'],   0.0,            0.0497
    elif parent == 'H2':   return nuclides['H1'],   0.0,            0.166767
    else: return -1, 0.0, 0.0

def n_alpha(nuclides, parent):
    if   parent == 'F19':  return [nuclides['N16'], nuclides['He4']], 2.166667E-5,    0.028393
    elif parent == 'O16':  return [nuclides['C13'], nuclides['He4']], 0.0,            0.144515
    elif parent == 'O17':  return [nuclides['C14'], nuclides['He4']], 0.117316,       0.260809
    elif parent == 'N14':  return [nuclides['B11'], nuclides['He4']], 0.104365,       0.080516
    elif parent == 'N15':  return [nuclides['B12'], nuclides['He4']], 0.0,            0.069240
    elif parent == 'B10':  return [nuclides['Li7'], nuclides['He4']], 0.281082,       0.044480
    elif parent == 'B11':  return [nuclides['Li8'], nuclides['He4']], 0.0,            0.031853
    else: return -1, 0.0, 0.0

def n_2alpha(nuclides, parent):
    if   parent == 'N14':  return [nuclides['Li7'], nuclides['He4']],  0.0,            0.031771
    elif parent == 'B10':  return [nuclides['H3'],  nuclides['He4']],  0.038439,       0.095487
    else: return -1, 0.0, 0.0

def n_nalpha(nuclides, parent):
    if   parent == 'F19':  return [nuclides['N15'], nuclides['He4']], 0.0,            0.3818
    elif parent == 'O17':  return [nuclides['C13'], nuclides['He4']], 0.0,            0.043420
    elif parent == 'N15':  return [nuclides['B11'], nuclides['He4']], 0.0,            0.012646
    elif parent == 'B11':  return [nuclides['Li7'], nuclides['He4']], 0.0,            0.286932
    elif parent == 'Be9':  return [nuclides['He6'], nuclides['He4']], 0.0825,         0.0104
    else: return -1, 0.0, 0.0

def n_2nalpha(nuclides, parent):
    if   parent == 'Li6':  return [nuclides['H1'], nuclides['He4']],  0.0,            0.0783
    elif parent == 'Li7':  return [nuclides['H2'], nuclides['He4']],  0.0,            0.020195
    else: return -1, 0.0, 0.0

def n_3nalpha(nuclides, parent):
    if   parent == 'Li7':  return [nuclides['H1'], nuclides['He4']],  0.0,            6.556330E-5
    else: return -1, 0.0, 0.0

def n_p(nuclides, parent):
    if   parent == 'F19':  return [nuclides['O19'],  nuclides['H1']], 0.0,            0.018438
    elif parent == 'O16':  return [nuclides['N16'],  nuclides['H1']], 0.0,            0.042723
    elif parent == 'O17':  return [nuclides['N17'],  nuclides['H1']], 0.0,            0.041838
    elif parent == 'N14':  return [nuclides['C14'],  nuclides['H1']], 0.014102,       0.043891
    elif parent == 'N15':  return [nuclides['C15'],  nuclides['H1']], 0.0,            0.019601
    elif parent == 'B10':  return [nuclides['Be10'], nuclides['H1']], 0.018860,       0.034093
    elif parent == 'B11':  return [nuclides['Be11'], nuclides['H1']], 0.0,            0.005564
    elif parent == 'Li6':  return [nuclides['He6'],  nuclides['H1']], 0.0,            0.00604
    elif parent == 'He3':  return [nuclides['H3'],   nuclides['H1']], 0.714941,       0.121
    else: return -1, 0.0, 0.0

def n_np(nuclides, parent):
    if   parent == 'F19':  return [nuclides['O18'],  nuclides['H1']], 0.0,            0.061973
    elif parent == 'N15':  return [nuclides['C14'],  nuclides['H1']], 0.0,            0.044827
    elif parent == 'B11':  return [nuclides['Be10'], nuclides['H1']], 0.0,            0.001016
    else: return -1, 0.0, 0.0

def n_d(nuclides, parent):
    if   parent == 'F19':  return [nuclides['O18'], nuclides['H2']],  0.0,            0.022215
    elif parent == 'O16':  return [nuclides['N15'], nuclides['H2']],  0.0,            0.017623
    elif parent == 'O17':  return [nuclides['N16'], nuclides['H2']],  0.0,            0.020579
    elif parent == 'N14':  return [nuclides['C13'], nuclides['H2']],  0.0,            0.042027
    elif parent == 'N15':  return [nuclides['C14'], nuclides['H2']],  0.0,            0.014926
    elif parent == 'B10':  return [nuclides['Be9'], nuclides['H2']],  0.0,            0.031270
    elif parent == 'Li7':  return [nuclides['He6'], nuclides['H2']],  0.0,            0.010199
    elif parent == 'He3':  return [nuclides['H2'],  nuclides['H2']],  0.0,            0.07609
    else: return -1, 0.0, 0.0

def n_t(nuclides, parent):
    if   parent == 'F19':  return [nuclides['O17'], nuclides['H3']],  0.0,            0.01303
    elif parent == 'N14':  return [nuclides['C12'], nuclides['H3']],  0.0,            0.028573
    elif parent == 'N15':  return [nuclides['C13'], nuclides['H3']],  0.0,            0.020163
    elif parent == 'B11':  return [nuclides['Be9'], nuclides['H3']],  0.0,            0.015172
    elif parent == 'Be9':  return [nuclides['Li7'], nuclides['H3']],  0.0,            0.020878
    elif parent == 'Li6':  return [nuclides['He4'], nuclides['H3']],  0.206155,       0.0258
    else: return -1, 0.0, 0.0

# NUEN 629 Homework 5 Problem 2

# Create Activation and Decay Matrix and initial nuclide quantity vector
A = np.zeros((len(nuclides),len(nuclides)))

hi_flux_frac = 0.5
lo_flux_frac = (1.0-hi_flux_frac)
phi = 1.0e14 * 60 * 60 * 24 #10^14 1/cm^2/s in 1/cm^2 /day
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
    row_lo_act_sum = row_n_gamma[1] + row_n_2n[1] + row_n_alpha[1] + \
                     row_n_2alpha[1] + row_n_nalpha[1] + row_n_2nalpha[1] + \
                     row_n_3nalpha[1] + row_n_p[1] + row_n_np[1] + row_n_d[1] +\
                     row_n_t[1]
    row_hi_act_sum = row_n_gamma[2] + row_n_2n[2] + row_n_alpha[2] + \
                     row_n_2alpha[2] + row_n_nalpha[2] + row_n_2nalpha[2] + \
                     row_n_3nalpha[2] + row_n_p[2] + row_n_np[2] + row_n_d[2] +\
                     row_n_t[2]
    if row_betanegdecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_betanegdecay[1] # [days^-1]
    elif row_betaposdecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_betaposdecay[1] # [days^-1]
    elif row_2alphadecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_2alphadecay[1] # [days^-1]
    else:
        row_lambda = 0.0
    # Diagonal Assignment
    A[row,row] = -row_lambda - phi_lo*row_lo_act_sum - phi_hi*row_hi_act_sum
    # Off Diagonal Assignment
    if row_betanegdecay[0] >= 0:
        A[row_betanegdecay[0],row] = log(2)*60*60*24/row_betanegdecay[1]
    if row_betaposdecay[0] >= 0:
        A[row_betaposdecay[0],row] = log(2)*60*60*24/row_betaposdecay[1]
    if row_2alphadecay[0] >= 0:
        A[row_2alphadecay[0],row] = log(2)*60*60*24/row_2alphadecay[1]
    if row_n_gamma[0] >= 0:
        A[row_n_gamma[0],row] = phi_lo*row_n_gamma[1] + phi_hi*row_n_gamma[2]
    if row_n_2n[0] >= 0:
        A[row_n_2n[0],row] = phi_lo*row_n_2n[1] + phi_hi*row_n_2n[2]
    if row_n_alpha[0] >= 0:
        for i in row_n_alpha[0]:
            A[i,row] = phi_lo*row_n_alpha[1] + phi_hi*row_n_alpha[2]
    if row_n_2alpha[0] >= 0:
        for i in row_n_2alpha[0]:
            A[i,row] = phi_lo*row_n_2alpha[1] + phi_hi*row_n_2alpha[2]
    if row_n_nalpha[0] >= 0:
        for i in row_n_nalpha[0]:
            A[i,row] = phi_lo*row_n_nalpha[1] + phi_hi*row_n_nalpha[2]
    if row_n_2nalpha[0] >= 0:
        for i in row_n_2nalpha[0]:
            A[i,row] = phi_lo*row_n_2nalpha[1] + phi_hi*row_n_2nalpha[2]
    if row_n_3nalpha[0] >= 0:
        for i in row_n_3nalpha[0]:
            A[i,row] = phi_lo*row_n_3nalpha[1] + phi_hi*row_n_3nalpha[2]
    if row_n_p[0] >= 0:
        for i in row_n_p[0]:
            A[i,row] = phi_lo*row_n_p[1] + phi_hi*row_n_p[2]
    if row_n_np[0] >= 0:
        for i in row_n_np[0]:
            A[i,row] = phi_lo*row_n_np[1] + phi_hi*row_n_np[2]
    if row_n_d[0] >= 0:
        for i in row_n_d[0]:
            A[i,row] = phi_lo*row_n_d[1] + phi_hi*row_n_d[2]
    if row_n_t[0] >= 0:
        for i in row_n_t[0]:
            A[i,row] = phi_lo*row_n_t[1] + phi_hi*row_n_t[2]
'''
    if row_betanegdecay[0] >= 0:
        A[row,row_betanegdecay[0]] = log(2)*60*60*24/row_betanegdecay[1]
    if row_betaposdecay[0] >= 0:
        A[row,row_betaposdecay[0]] = log(2)*60*60*24/row_betaposdecay[1]
    if row_2alphadecay[0] >= 0:
        A[row,row_2alphadecay[0]] = log(2)*60*60*24/row_2alphadecay[1]
    if row_n_gamma[0] >= 0:
        A[row,row_n_gamma[0]] = phi_lo*row_n_gamma[1] + phi_hi*row_n_gamma[2]
    if row_n_2n[0] >= 0:
        A[row,row_n_2n[0]] = phi_lo*row_n_2n[1] + phi_hi*row_n_2n[2]
    if row_n_alpha[0] >= 0:
        for i in row_n_alpha[0]:
            A[row,i] = phi_lo*row_n_alpha[1] + phi_hi*row_n_alpha[2]
    if row_n_2alpha[0] >= 0:
        for i in row_n_2alpha[0]:
            A[row,i] = phi_lo*row_n_2alpha[1] + phi_hi*row_n_2alpha[2]
    if row_n_nalpha[0] >= 0:
        for i in row_n_nalpha[0]:
            A[row,i] = phi_lo*row_n_nalpha[1] + phi_hi*row_n_nalpha[2]
    if row_n_2nalpha[0] >= 0:
        for i in row_n_2nalpha[0]:
            A[row,i] = phi_lo*row_n_2nalpha[1] + phi_hi*row_n_2nalpha[2]
    if row_n_3nalpha[0] >= 0:
        for i in row_n_3nalpha[0]:
            A[row,i] = phi_lo*row_n_3nalpha[1] + phi_hi*row_n_3nalpha[2]
    if row_n_p[0] >= 0:
        for i in row_n_p[0]:
            A[row,i] = phi_lo*row_n_p[1] + phi_hi*row_n_p[2]
    if row_n_np[0] >= 0:
        for i in row_n_np[0]:
            A[row,i] = phi_lo*row_n_np[1] + phi_hi*row_n_np[2]
    if row_n_d[0] >= 0:
        for i in row_n_d[0]:
            A[row,i] = phi_lo*row_n_d[1] + phi_hi*row_n_d[2]
    if row_n_t[0] >= 0:
        for i in row_n_t[0]:
            A[row,i] = phi_lo*row_n_t[1] + phi_hi*row_n_t[2]
'''


#plt.spy(A)
#plt.show()
#sys.exit("testing")

b = np.zeros(len(nuclides))

# N_0 expressed as kg nuclide per kg FLiBe
b[nuclides['F19']] = 0.7685
b[nuclides['Be9']] = 0.0911
b[nuclides['Li6']] = 0.01065636
b[nuclides['Li7']] = 0.12974364
"""
# N_0 expressed as % nuclide abundance per FLiBe molecule
b[nuclides['F19']] = 0.5714286
b[nuclides['Be9']] = 0.1428571
b[nuclides['Li6']] = 0.0216857
b[nuclides['Li7']] = 0.2640286
"""

# Backward Euler Method
time_iters = range(0,75)
iter_length = 9.74
ts = np.arange(0.0,731.0,9.74)
u_euler = np.zeros(len(nuclides))
u_set_euler = np.zeros((len(time_iters)+1,len(nuclides)))
u_set_euler[0,:] = b


b_new = b
for i_iter in time_iters:
    i_iter += 1
    u_euler = np.dot(np.linalg.inv(np.identity(len(nuclides)) - \
              A*iter_length),b_new)
    #print(u_euler)
    u_euler = np.absolute(u_euler)
    u_set_euler[i_iter,:] = u_euler
    b_new = u_euler

print("Amount of tritium at end of irradiation:", u_euler[2])

sort_index = np.argsort(u_set_euler[75,:])
# Plot the top seven isotopes that make up the irradiated material
plt.plot(ts,u_set_euler[:,sort_index[32]], label=nuclide_names[sort_index[32]])
plt.plot(ts,u_set_euler[:,sort_index[31]], label=nuclide_names[sort_index[31]])
plt.plot(ts,u_set_euler[:,sort_index[30]], label=nuclide_names[sort_index[30]])
plt.plot(ts,u_set_euler[:,sort_index[29]], label=nuclide_names[sort_index[29]])
plt.plot(ts,u_set_euler[:,sort_index[28]], label=nuclide_names[sort_index[28]])
plt.plot(ts,u_set_euler[:,sort_index[27]], label=nuclide_names[sort_index[27]])
plt.plot(ts,u_set_euler[:,sort_index[26]], label=nuclide_names[sort_index[26]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot the second top seven isotopes making up irradiated material
plt.plot(ts,u_set_euler[:,sort_index[25]], label=nuclide_names[sort_index[25]])
plt.plot(ts,u_set_euler[:,sort_index[24]], label=nuclide_names[sort_index[24]])
plt.plot(ts,u_set_euler[:,sort_index[23]], label=nuclide_names[sort_index[23]])
plt.plot(ts,u_set_euler[:,sort_index[22]], label=nuclide_names[sort_index[22]])
plt.plot(ts,u_set_euler[:,sort_index[21]], label=nuclide_names[sort_index[21]])
plt.plot(ts,u_set_euler[:,sort_index[20]], label=nuclide_names[sort_index[20]])
plt.plot(ts,u_set_euler[:,sort_index[19]], label=nuclide_names[sort_index[19]])
plt.yscale('log')
plt.legend()
plt.show()
'''
# Plot third top seven isotopes making up irradiated material
plt.plot(ts,u_set_euler[:,sort_index[18]], label=nuclide_names[sort_index[18]])
plt.plot(ts,u_set_euler[:,sort_index[17]], label=nuclide_names[sort_index[17]])
plt.plot(ts,u_set_euler[:,sort_index[16]], label=nuclide_names[sort_index[16]])
plt.plot(ts,u_set_euler[:,sort_index[15]], label=nuclide_names[sort_index[15]])
plt.plot(ts,u_set_euler[:,sort_index[14]], label=nuclide_names[sort_index[14]])
plt.plot(ts,u_set_euler[:,sort_index[13]], label=nuclide_names[sort_index[13]])
plt.plot(ts,u_set_euler[:,sort_index[12]], label=nuclide_names[sort_index[12]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot fourth top six isotopes making up irradated material
plt.plot(ts,u_set_euler[:,sort_index[11]], label=nuclide_names[sort_index[11]])
plt.plot(ts,u_set_euler[:,sort_index[10]], label=nuclide_names[sort_index[10]])
plt.plot(ts,u_set_euler[:,sort_index[9]], label=nuclide_names[sort_index[9]])
plt.plot(ts,u_set_euler[:,sort_index[8]], label=nuclide_names[sort_index[8]])
plt.plot(ts,u_set_euler[:,sort_index[7]], label=nuclide_names[sort_index[7]])
plt.plot(ts,u_set_euler[:,sort_index[6]], label=nuclide_names[sort_index[6]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot fifth top six isotopes making up irradated material
plt.plot(ts,u_set_euler[:,sort_index[5]], label=nuclide_names[sort_index[5]])
plt.plot(ts,u_set_euler[:,sort_index[4]], label=nuclide_names[sort_index[4]])
plt.plot(ts,u_set_euler[:,sort_index[3]], label=nuclide_names[sort_index[3]])
plt.plot(ts,u_set_euler[:,sort_index[2]], label=nuclide_names[sort_index[2]])
plt.plot(ts,u_set_euler[:,sort_index[1]], label=nuclide_names[sort_index[1]])
plt.plot(ts,u_set_euler[:,sort_index[0]], label=nuclide_names[sort_index[0]])
plt.yscale('log')
plt.legend()
plt.show()
'''

# Matrix Exponential Method
ts = np.arange(0., 730.51, 1.2175)
print(len(ts))
t_iter = 0
u_set_matexp = np.zeros((len(ts),len(nuclides)))


for t in ts:
    u_matexp = np.zeros(len(nuclides))
    temp = sp.linalg.expm(A*t)
    u_matexp = np.dot(temp,b)
    u_set_matexp[t_iter,:] = u_matexp
    t_iter += 1
    #print(u_matexp)

print("Amount of tritium at end of irradiation:", u_matexp[2])

sort_index = np.argsort(u_set_matexp[len(ts)-1,:])
# Plot the top seven isotopes that make up the irradiated material
plt.plot(ts,u_set_matexp[:,sort_index[32]], label=nuclide_names[sort_index[32]])
plt.plot(ts,u_set_matexp[:,sort_index[31]], label=nuclide_names[sort_index[31]])
plt.plot(ts,u_set_matexp[:,sort_index[30]], label=nuclide_names[sort_index[30]])
plt.plot(ts,u_set_matexp[:,sort_index[29]], label=nuclide_names[sort_index[29]])
plt.plot(ts,u_set_matexp[:,sort_index[28]], label=nuclide_names[sort_index[28]])
plt.plot(ts,u_set_matexp[:,sort_index[27]], label=nuclide_names[sort_index[27]])
plt.plot(ts,u_set_matexp[:,sort_index[26]], label=nuclide_names[sort_index[26]])
plt.yscale('log')
plt.legend()
plt.show()


# Matrix Exponential via Quadrature Method
ts = np.arange(0.0,730.51,9.74)
t_iter = 0
N = 32
#u_quad = np.zeros(len(nuclides))
u_set_quad = np.zeros((len(ts),len(nuclides)))

for t in ts:
    theta = np.pi*np.arange(1,N,2)/N
    z = N*(.1309 - 0.1194*theta**2 + .2500j*theta)
    w = N*(- 2*0.1194*theta + .2500j)
    c = 1.0j/N*np.exp(z)*w
    #plt.plot(np.real(z),np.imag(z),'o-')
    #plt.savefig('hw5.pdf')
    #plt.close()
    u_quad = np.zeros(len(nuclides))
    for k in range(int(N/2)):
        n,code = splinalg.gmres(z[k]*sparse.identity(len(nuclides))  - A*t,
                 b, tol=1e-12, maxiter=2000)
        if (code):
            print(code)
        u_quad = u_quad- c[k]*n
    u_quad = 2*np.real(u_quad)
    u_quad = np.absolute(u_quad)
    u_set_quad[t_iter,:] = u_quad
    t_iter += 1
    #print(u_quad)

print("Amount of tritium at end of irradiation:", u_quad[2])

sort_index = np.argsort(u_set_quad[75,:])
# Plot the top seven isotopes that make up the irradiated material
plt.plot(ts,u_set_quad[:,sort_index[32]], label=nuclide_names[sort_index[32]])
plt.plot(ts,u_set_quad[:,sort_index[31]], label=nuclide_names[sort_index[31]])
plt.plot(ts,u_set_quad[:,sort_index[30]], label=nuclide_names[sort_index[30]])
plt.plot(ts,u_set_quad[:,sort_index[29]], label=nuclide_names[sort_index[29]])
plt.plot(ts,u_set_quad[:,sort_index[28]], label=nuclide_names[sort_index[28]])
plt.plot(ts,u_set_quad[:,sort_index[27]], label=nuclide_names[sort_index[27]])
plt.plot(ts,u_set_quad[:,sort_index[26]], label=nuclide_names[sort_index[26]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot the second top seven isotopes making up irradiated material
plt.plot(ts,u_set_quad[:,sort_index[25]], label=nuclide_names[sort_index[25]])
plt.plot(ts,u_set_quad[:,sort_index[24]], label=nuclide_names[sort_index[24]])
plt.plot(ts,u_set_quad[:,sort_index[23]], label=nuclide_names[sort_index[23]])
plt.plot(ts,u_set_quad[:,sort_index[22]], label=nuclide_names[sort_index[22]])
plt.plot(ts,u_set_quad[:,sort_index[21]], label=nuclide_names[sort_index[21]])
plt.plot(ts,u_set_quad[:,sort_index[20]], label=nuclide_names[sort_index[20]])
plt.plot(ts,u_set_quad[:,sort_index[19]], label=nuclide_names[sort_index[19]])
plt.yscale('log')
plt.legend()
plt.show()
'''
# Plot third top seven isotopes making up irradiated material
plt.plot(ts,u_set_quad[:,sort_index[18]], label=nuclide_names[sort_index[18]])
plt.plot(ts,u_set_quad[:,sort_index[17]], label=nuclide_names[sort_index[17]])
plt.plot(ts,u_set_quad[:,sort_index[16]], label=nuclide_names[sort_index[16]])
plt.plot(ts,u_set_quad[:,sort_index[15]], label=nuclide_names[sort_index[15]])
plt.plot(ts,u_set_quad[:,sort_index[14]], label=nuclide_names[sort_index[14]])
plt.plot(ts,u_set_quad[:,sort_index[13]], label=nuclide_names[sort_index[13]])
plt.plot(ts,u_set_quad[:,sort_index[12]], label=nuclide_names[sort_index[12]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot fourth top six isotopes making up irradated material
plt.plot(ts,u_set_quad[:,sort_index[11]], label=nuclide_names[sort_index[11]])
plt.plot(ts,u_set_quad[:,sort_index[10]], label=nuclide_names[sort_index[10]])
plt.plot(ts,u_set_quad[:,sort_index[9]], label=nuclide_names[sort_index[9]])
plt.plot(ts,u_set_quad[:,sort_index[8]], label=nuclide_names[sort_index[8]])
plt.plot(ts,u_set_quad[:,sort_index[7]], label=nuclide_names[sort_index[7]])
plt.plot(ts,u_set_quad[:,sort_index[6]], label=nuclide_names[sort_index[6]])
plt.yscale('log')
plt.legend()
plt.show()
# Plot fifth top six isotopes making up irradated material
plt.plot(ts,u_set_quad[:,sort_index[5]], label=nuclide_names[sort_index[5]])
plt.plot(ts,u_set_quad[:,sort_index[4]], label=nuclide_names[sort_index[4]])
plt.plot(ts,u_set_quad[:,sort_index[3]], label=nuclide_names[sort_index[3]])
plt.plot(ts,u_set_quad[:,sort_index[2]], label=nuclide_names[sort_index[2]])
plt.plot(ts,u_set_quad[:,sort_index[1]], label=nuclide_names[sort_index[1]])
plt.plot(ts,u_set_quad[:,sort_index[0]], label=nuclide_names[sort_index[0]])
plt.yscale('log')
plt.legend()
plt.show()
'''

# NUEN 629 Homework 5 Problem 3

A_d = np.zeros((len(nuclides),len(nuclides)))

for isotope in nuclides:
    row = nuclides[isotope]
    row_betanegdecay =  betanegdecay(nuclides, isotope)
    row_betaposdecay =  betaposdecay(nuclides, isotope)
    row_2alphadecay =   twoalphadecay(nuclides, isotope)
    if row_betanegdecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_betanegdecay[1] # [days^-1]
    elif row_betaposdecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_betaposdecay[1] # [days^-1]
    elif row_2alphadecay[0] >= 0:
        row_lambda = log(2)*60*60*24/row_2alphadecay[1] # [days^-1]
    else:
        row_lambda = 0.0
    # Diagonal Assignment
    A_d[row,row] = -row_lambda
    # Off Diagonal Assignment
    if row_betanegdecay[0] >= 0:
        A_d[row_betanegdecay[0],row] = log(2)*60*60*24/row_betanegdecay[1]
    if row_betaposdecay[0] >= 0:
        A_d[row_betaposdecay[0],row] = log(2)*60*60*24/row_betaposdecay[1]
    if row_2alphadecay[0] >= 0:
        A_d[row_2alphadecay[0],row] = log(2)*60*60*24/row_2alphadecay[1]


b_d = u_euler

time_iters = range(0,500)
iter_length = 16524.
ts = np.arange(0.0,8.2620000001e6,16524.)
u_decay = np.zeros(len(nuclides))
b_new = b_d
Activities = np.zeros(len(ts))
Activities[0] = sum((6.0221409e23/atom_mass)*b_d*decay_consts)
print(Activities[0])
#u_set_decay = np.zeros((len(time_iters)+1,len(nuclides)))


for t_iter in time_iters:
    t_iter += 1
    u_decay = np.dot(np.linalg.inv(np.identity(len(nuclides)) - \
              A_d*iter_length), b_new)
    #print(u_decay)
    u_decay = np.absolute(u_decay)
    Activities[t_iter] = sum((6.0221409e23/atom_mass)*u_decay*decay_consts)
    print(Activities[t_iter])
    b_new = u_decay

plt.plot(ts,Activities)
plt.yscale('log')
plt.show()
