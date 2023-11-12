#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:44:36 2021

@author: ntosi
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors, colorbar, patches

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

def mkrandom(size, balance=0.5):
  arr = np.random.uniform(size=size)
  nwanted = int( np.floor( size[0] * size[1] * balance ) )
  s = np.sort(arr.flatten())
  cut = np.mean(s[nwanted-1:nwanted+1])
  return (arr > cut).astype(int)

def mkholes(mask, pitch):
  hdim = np.asarray(mask.shape) / 2. - 0.5
  hidx = np.argwhere(mask)
  return ( hidx - hdim ) * pitch

def writexml(holes):
  template ='''      <multiUnionNode name="Node-{0}">
        <solid ref="hole"/>
        <position name="UnitedBoxes0x7ffff7099560_pos" unit="mm" x="{1:.03f}" y="{2:.03f}" z="0"/>
      </multiUnionNode>'''
  print("BEGIN XML SEGMENT")
  for i, hole in enumerate( holes ):
    print(template.format(i, hole[0], hole[1]))
  print("END XML SEGMENT")

size = np.array((16, 16))

pitch = np.array((3.2, 3.2))

hsize = np.array((3.0, 3.0))

horect = np.array((30.0, 30.0))

mask = mkrandom(size)

holes = mkholes(mask, pitch)

plot(holes, hsize, horect)

writexml(holes)
