#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:22:40 2021

@author: alessandro
"""
#%%Importing the necessary libraries
#standard libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors, colorbar, patches
#%%Plotting of the mask
def plot(holecentres, holesize, halfouterrect, **kwargs):
  fig = plt.figure(figsize=(12,12))
  ax = fig.gca()
  ax.set_facecolor('xkcd:steel')
  hsize = holesize / 2.
  for hole in holecentres:
    ax.add_patch(patches.Rectangle(hole - hsize, holesize[0], holesize[1], linewidth=0, facecolor='w'))
  ax.set_xlim((-halfouterrect[0],halfouterrect[0]))
  ax.set_ylim((-halfouterrect[1],halfouterrect[1]))
  plt.show()




