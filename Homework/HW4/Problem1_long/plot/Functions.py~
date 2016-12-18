#!/usr/bin/env python3

"""
FractionAM converts atom fractions to mass fractions
and mass fractions to atom fractions. Input is a 
single string with MCNP style fractions.
"""

__author__     =  "Paul Mendoza"
__copyright__  =  "Copyright 2016, Planet Earth"
__credits__    = ["Sunil Chirayath",
                  "Charles Folden",
                  "Jeremy Conlin"]
__license__    =  "GPL"
__version__    =  "1.0.1"
__maintainer__ =  "Paul Mendoza"
__email__      =  "paul.m.mendoza@gmail.com"
__status__     =  "Production"

################################################################
##################### Import packages ##########################
################################################################

import time
start_time = time.time()
import scipy.special as sps
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import random as rn
import Functions as fun

################################################################
######################### Functions ############################
################################################################

def GammaPDF(a,b,z):
    f_x=((z**(a-1))*np.exp(-z/b))\
                     /\
        (sps.gamma(a)*b**(a))
    return(f_x)


def Plot(theta,pi,f_theta):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(theta,pi,'k--',linewidth=2.0,label=r'$\pi(\theta)$')
    ax.plot(theta,f_theta,'.-',color='gray',linewidth=2.0,
            markersize=8,label=r'$f(\theta|x)$')
    
    ax.set_xlabel(r'$\boldsymbol{\theta}$',fontsize=18)
    ax.set_ylabel(r'\textbf{Probability}',fontsize=18)
    #ax.yaxis.labelpad=55
    
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    ax.grid(alpha=0.8,color='black',linestyle='dotted')
    ax.grid(alpha=0.8,color='black',linestyle='dotted')

    handles,labels=ax.get_legend_handles_labels()

    Lfont={'family':'monospace',
           'size':12}
    ax.legend(handles,labels,loc='best',
              fontsize=12,prop=Lfont)

    plt.savefig('P8F1.pdf')
    #return(fig,ax)
