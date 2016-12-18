#!/usr/bin/env python3

"""
My functions...to clean up the main code
"""

__author__     =  "Paul Mendoza"
__copyright__  =  "Copyright 2016, Planet Earth"
__credits__    = ["Ryan_McClarren"]
__license__    =  "GPL"
__version__    =  "1.0.1"
__maintainer__ =  "Paul Mendoza"
__email__      =  "paul.m.mendoza@gmail.com"
__status__     =  "Production"

################################################################
##################### Import packages ##########################
################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import matplotlib.mlab as mlab

#############################################################
######################### Variables #########################
#############################################################

# Basic information
FigureSize = (11, 6)              # Dimensions of the figure
TypeOfFamily='monospace'          # This sets the type of font for text
font = {'family' : TypeOfFamily}  # This sets the type of font for text
LegendFontSize = 6
Lfont = {'family' : TypeOfFamily}  # This sets up legend font
Lfont['size']=LegendFontSize

Title = ''
TitleFontSize = 22
TitleFontWeight = "bold"  # "bold" or "normal"

XFontSize=18          # X label font size
XFontWeight="normal"  # "bold" or "normal"

YFontSize=18                    # Y label font size
YFontWeight="normal"            # "bold" or "normal"

Colors=["aqua","gray","red","blue","black",
        "green","magenta","indigo","lime","peru","steelblue",
        "darkorange","salmon","yellow","lime","black"]

# If you want to highlight a specific item
# set its alpha value =1 and all others to 0.4
# You can also change the MarkSize (or just use the highlight option below)
Alpha_Value=[1  ,1  ,1  ,1  ,1  ,1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
MarkSize=   [8  ,8  ,8  ,8  ,8  ,8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8]

Linewidth=[1  ,1  ,1  ,1  ,1  ,1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

# Can change all these to "." or "" for nothing "x" isn't that good
MarkerType=["8","s","p","D","*","H","h","d","^",">"]

# LineStyles=["solid","dashed","dash_dot","dotted","."]
LineStyles=["solid"]

SquishGraph = 0.75
BBOXX = 1.24;BBOXY = 0.5       # Set legend on right side of graph
NumberOfLegendColumns = 1  

################################################################
######################### Functions ############################
################################################################

def loop_values(list1,index):
    """
    This function will loop through values in list even if 
    outside range (in the positive sense not negative)
    """
    while True:
        try:
            list1[index]
            break
        except IndexError:
            index=index-len(list1)
    return(list1[index])

def plot(x,y,ax,Check,label,fig,Ylabel,Xlabel,XScale,YScale):
	#Plot X and Y
    ax.plot(x,y,
            linestyle=loop_values(LineStyles,Check),
            marker=loop_values(MarkerType,Check),
            color=loop_values(Colors,Check),
            markersize=loop_values(MarkSize,Check),
            alpha=loop_values(Alpha_Value,Check),
            label=label)
    	
    #Log or linear scale?
    ax.set_xscale(XScale)
    ax.set_yscale(YScale)
    #Set Title
    fig.suptitle(Title,fontsize=TitleFontSize,
		 fontweight=TitleFontWeight,fontdict=font,ha='center')
    #Set X and y labels
    ax.set_xlabel(Xlabel,
		  fontsize=XFontSize,fontweight=XFontWeight,
		  fontdict=font)
    ax.set_ylabel(Ylabel,
		  fontsize=YFontSize,
		  fontweight=YFontWeight,
		  fontdict=font)
	
    return(fig,ax)

def Legend(ax):
    handles,labels=ax.get_legend_handles_labels()
    box=ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*SquishGraph,
                     box.height])
    ax.legend(handles,labels,loc='center',
              bbox_to_anchor=(BBOXX,BBOXY),
              fontsize=LegendFontSize,prop=font,
              ncol=NumberOfLegendColumns)
    return(ax)
