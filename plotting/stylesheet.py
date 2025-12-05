import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, colorsys
from matplotlib import rcParams as rc
import os, sys,subprocess, platform
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)
from os.path import relpath

Fontsize = 30
rc["legend.markerscale"] = 2.0
rc["legend.framealpha"] = 0
rc["legend.labelspacing"] = 0.1
rc['figure.figsize'] = (20,10)
rc['axes.autolimit_mode'] = 'data'
rc['axes.xmargin'] = 0
rc['axes.ymargin'] = 0.10
rc['axes.titlesize'] = 30
rc['axes.labelsize'] = Fontsize
rc['xtick.direction'] = 'in'
rc['ytick.direction'] = 'in'
rc['font.sans-serif'] = "Times New Roman"
rc['font.serif'] = "Times New Roman"
rc['xtick.labelsize'] = Fontsize
rc['ytick.labelsize'] = Fontsize
rc['axes.grid'] = True
rc['grid.linestyle'] = '-'
rc['grid.alpha'] = 0.2
rc['legend.fontsize'] = int(Fontsize*0.9)
rc['legend.loc'] = 'upper left'
rc["figure.autolayout"] = True
rc["savefig.dpi"] = 300
rc["text.usetex"] = True
rc["font.family"] = "Times New Roman"
rc["lines.markeredgecolor"] = matplotlib.colors.to_rgba('black', 0.5)
rc["lines.markeredgewidth"] = 0.01
rc["legend.markerscale"] = 2.0
rc['text.latex.preamble'] = r'\usepackage{amssymb}'
rc.update({
  "text.usetex": True,
  "font.family": "Times New Roman"
})

def set_fontsizes(fontsize):
  rc['axes.labelsize'] = fontsize
  rc['xtick.labelsize'] = fontsize
  rc['ytick.labelsize'] = fontsize
  rc['legend.fontsize'] = int(fontsize*0.9)

#Lets use the following markers: Triangle, Square, Circle, Star, Diamond
Markers = ['o','^', 's', '*', 'D']
MarkerScales = np.array([1.1, 1.25, 1., 1.5, 1.])


#Color dictionary
CD = { 
  "C0" : 'r', 
  "C1" :    "#FDEE00", 
  "C2" :    "#06D6A0", 
  "C3" :    "#FF4365", 
  "C4" :    "#14080E", 
  "C5" :    "#6320EE",
  "C6" :    "#963D5A", 

  "C7" :    "#7570b3", 
  "C8" :    "#d95f02", 
  "C9" :    "#e7298a", 
  "C10" :   "#66a61e", 
  "C11" :   "#8931EF", 

  "C12" :   "#1f77b4", 
  "C13" :   "#e377c2", 
  "C14" :   "#0D9276",
  "C15" :   "#8c564b",
}