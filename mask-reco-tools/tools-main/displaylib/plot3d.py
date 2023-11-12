#!/usr/bin/env python3
#**************************************************************************
#*                                                                        *
#*   AUTHOR: nicolo.tosi@bo.infn.it                                       *
#*                                                                        *
#*                                                                        *
#*   This program is free software; you can redistribute it and/or modify *
#*   it under the terms of the GNU Lesser General Public License (LGPL)   *
#*   as published by the Free Software Foundation; either version 2 of    *
#*   the License, or (at your option) any later version.                  *
#*   for detail see the LICENCE text file.                                *
#*                                                                        *
#*   This program is distributed in the hope that it will be useful,      *
#*   but WITHOUT ANY WARRANTY; without even the implied warranty of       *
#*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
#*   GNU Library General Public License for more details.                 *
#*                                                                        *
#*   You should have received a copy of the GNU Library General Public    *
#*   License along with this program; if not, write to the Free Software  *
#*   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 *
#*   USA                                                                  *
#*                                                                        *
#*                                                                        *
#*                                                                        *
#**************************************************************************

import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm, colors, colorbar, patches

def voxels(fig, ax, ax_cb, data, **kwargs):
  shape = data.shape
  data = data.flatten(order='C')
  norm = colors.Normalize(vmin=int(kwargs['cut_low']), vmax=int(kwargs['cut_up']))
  cmap = 'viridis'
  cols = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(data)
  #apply cuts to data
  condition = np.ones(shape=data.shape, dtype=bool)
  if 'mask' in kwargs and kwargs['mask'] is not None:
    mask = kwargs['mask'].flatten(order='C')
    condition = np.logical_and(condition, mask > 0)
  if 'cut_low' in kwargs and kwargs['cut_low'] is not None:
    cut_low = kwargs['cut_low']
    condition = np.logical_and(condition, data >= cut_low)
  if 'cut_up' in kwargs and kwargs['cut_up'] is not None:
    cut_up = kwargs['cut_up']
    condition = np.logical_and(condition, data <= cut_up)
  cols = cols[condition]
  xform = kwargs.get('xform', np.eye(3))[0:3,0:3]
  scale = kwargs.get('scale', 100.)
  if False: # shape[0] * shape[1] * shape[2] > 25 ** 3:
    #We cant make pretty voxels and get somthing interactive, so scatter plot it is...
    hsize = np.array(shape) / 2
    xs = np.linspace(-hsize[0] * scale, hsize[0] * scale, shape[0])
    ys = np.linspace(-hsize[1] * scale, hsize[1] * scale, shape[1])
    zs = np.linspace(-hsize[2] * scale, hsize[2] * scale, shape[2])
    x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
    x = x.flatten(order='C')[condition]
    y = y.flatten(order='C')[condition]
    z = z.flatten(order='C')[condition]
    xyz = np.vstack([x, y, z])
    xyz = np.matmul(xform, xyz)
    x, y, z = np.vsplit(xyz, 3)
    ax.scatter(x, y, z, marker='o', color=cols, alpha=kwargs['alpha'], s=scale*1)
  else:
    C = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    C = np.array(C).astype(float)
    if 'spaced' in kwargs and kwargs['spaced']:
      space = 0.1
      C = C - C * 2 * space + space
    C -= 0.5
    faces = []
    for xi in range(shape[0]):
      for yi in range(shape[1]):
        for zi in range(shape[2]):
          xyz = np.array([xi,yi,zi]) - np.array(shape)/2 + (0.5, 0.5, 0.5)
          xyz = np.matmul(xform, xyz)
          faces.append((xyz + C) * scale)
    cubes = np.concatenate(faces)
    cubes = cubes[np.repeat(condition, 6, axis=0)]
    pc = Poly3DCollection(cubes)
    pc.set_facecolors(np.repeat(cols, 6, axis=0))
    pc.set_alpha(kwargs['alpha'])
    pc.set_edgecolors(None)
    pc.set_linewidths(0.25)
    ax.add_collection3d(pc)
  ax.set_xlim([-0.1, shape[0] + 0.1])
  ax.set_ylim([-0.1, shape[1] + 0.1])
  ax.set_zlim([-0.1, shape[2] + 0.1])
  if ax_cb is not None:
    bins, edges = np.histogram(data[condition], bins=100)
    bincolors = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba((edges[:-1] + edges[1:]) / 2.)
    egap = edges[1] - edges[0]
    ax_cb.barh(edges[:-1], bins, height=egap, align='center', linewidth=0, color=bincolors, log=True)
    #colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='vertical')


def frames(fig, ax, geotree, labels=False):
  origins = []
  xnormals = []
  ynormals = []
  znormals = []
  names = []
  maskverts = [[+88.8, +88.8, 0,1], [+88.8, -88.8, 0,1], [-88.8, -88.8, 0, 1],[-88.8, +88.8, 0, 1]]  #TODO: get mask size from geometry
  verts = []
  def rdp(geotree, o, x, y, z, n, depth):
    for volume in geotree.children.values():
      #ugly hack
      if 'geometry.Camera' in str(type(volume)):
        n.append(volume.path().split('/')[-1])
        xf = volume.gxform()
        o.append(xf[0:3,3])
        x.append(xf[0:3,0] * depth)
        y.append(xf[0:3,1] * depth)
        z.append(xf[0:3,2] * depth)
        v =[]
        for m in maskverts:
          vert = np.matmul(xf, m)
          v.append(vert[0:3])
        verts.append(v)
      #else:
        #n.append('')
      rdp(volume, o, x, y, z, n, depth * 0.8)
  rdp(geotree, origins, xnormals, ynormals, znormals, names, 1.0)
  for o, n in zip(origins, names):
    if len(n):
      ax.text(*o, n, fontsize='xx-small', ha='center')
  origins = np.asarray(origins).transpose()
  xnormals = np.asarray(xnormals).transpose() * 80.
  ynormals = np.asarray(ynormals).transpose() * 80.
  znormals = np.asarray(znormals).transpose() * 80.
  #ax.quiver(*origins, *xnormals, color='r')
  #ax.quiver(*origins, *ynormals, color='g')
  ax.quiver(*origins, *znormals, color='b')
  for v in verts: 
    quadr = Poly3DCollection([v], color='r', alpha=0.2, edgecolor='w')
    ax.add_collection3d(quadr)



def plot3d(geotree, **kwargs):
  #geotree -> FIDUCIAL VOLUME (lar_physical)
  gxform = geotree.gxform()
  voxel_size = geotree.voxel_size
  fig = plt.figure(figsize=(16,12))
  scalesz = 300
  if 'voxels' in kwargs:
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9], projection='3d')
    ax_cb = fig.add_axes([0.8, 0.05, 0.15, 0.9])
    if 'cut_low' in kwargs:
      cut_low = kwargs['cut_low']
    else:
      cut_low = None
    if 'cut_up' in kwargs:
      cut_up = kwargs['cut_up']
    else:
      cut_up = None  
    if 'alpha' in kwargs:
      alpha = kwargs['alpha']
    else:
      alpha = 0.3
    voxels(fig, ax, ax_cb, kwargs['voxels'], alpha = alpha, cut_low=cut_low, cut_up=cut_up,
           scale=voxel_size, xform = gxform, spaced=True)
    scalesz = np.max(kwargs['voxels'].shape) * voxel_size / 1.6
  else:
    ax = fig.gca(projection='3d')
  if 'frames' in kwargs:
    labels = 'labels' in kwargs
    frames(fig, ax, geotree, labels=labels)
  if 'vertex' in kwargs:
    vtx = kwargs['vertex']
    ax.scatter(*vtx, marker='o', color='r', alpha=1, s=20)
  if 'track' in kwargs:
    if kwargs['inputType'] == 'edepsim':
      for vertex in kwargs['track'].vertices:
        hitp = vertex.position
        for particle in vertex.particles:
          dirp = (particle.momentum / particle.mass)*50
          if particle.PDGCode == 13:
            ax.quiver(*hitp, *dirp, color='xkcd:lightblue', arrow_length_ratio=0.1, linewidth=2, label = "mu-")
          elif particle.PDGCode == -13:
            ax.quiver(*hitp, *dirp, color='xkcd:lightblue', arrow_length_ratio=0.1, linewidth=2, label = "mu+")
          elif particle.PDGCode == 11:
            ax.quiver(*hitp, *dirp, color='xkcd:red', arrow_length_ratio=0.1, linewidth=2, label = "e-")
          elif particle.PDGCode == -11:
            ax.quiver(*hitp, *dirp, color='xkcd:red', arrow_length_ratio=0.1, linewidth=2, label = "e+")
          elif particle.PDGCode == 111:
            ax.quiver(*hitp, *dirp, color='xkcd:darkgreen', arrow_length_ratio=0.1, linewidth=2, label = "pi0")
          elif particle.PDGCode == 211:
            ax.quiver(*hitp, *dirp, color='xkcd:coral', arrow_length_ratio=0.1, linewidth=2, label = "pi+")
          elif particle.PDGCode == -211:
            ax.quiver(*hitp, *dirp, color='xkcd:wheat', arrow_length_ratio=0.1, linewidth=2, label = "pi-")
          elif particle.PDGCode == 2112:
            ax.quiver(*hitp, *dirp, color='xkcd:sienna', arrow_length_ratio=0.1, linewidth=2, label = "n")
          elif particle.PDGCode == 2212:
            ax.quiver(*hitp, *dirp, color='xkcd:pink', arrow_length_ratio=0.1, linewidth=2, label = "p")
          else:
            ax.quiver(*hitp, *dirp, color='xkcd:black', arrow_length_ratio=0.1, linewidth=2, label = particle.PDGCode)
    else:
      hitp = kwargs['track'][0]
      dirp = kwargs['track'][1]
      ax.quiver(*hitp, *dirp, color='xkcd:lightblue', arrow_length_ratio=0.1, linewidth=2)
  ax.set_xlim([-scalesz, scalesz])
  ax.set_ylim([-scalesz, scalesz])
  ax.set_zlim([-scalesz, scalesz])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  ax.view_init(30, 45)

  # # event 13
  # x, y, z = [-198, -268], [30, -244], [-113, 87]
  # ax.scatter(x, y, z, c='red', s=50)
  # ax.plot(x,y,z, color='red')
  # x2, y2, z2 = [-397, -199], [-77, -6.5], [-15.4, -117]
  # ax.scatter(x2, y2, z2, c='red', s=50)
  # ax.plot(x2,y2,z2, color='red')
  
  # event 24
  # x, y, z = [45, -185], [291, 326], [128, -40]
  # ax.scatter(x, y, z, c='red', s=50)
  # ax.plot(x,y,z, color='red')
  # x2, y2, z2 = [-397, -199], [-77, -6.5], [-15.4, -117]
  # ax.scatter(x2, y2, z2, c='red', s=50)
  # ax.plot(x2,y2,z2, color='red')

  if 'title' in kwargs:
    title = kwargs['title'] + '\n'
    if 'track' in kwargs:
      if kwargs['inputType'] == 'edepsim':
        for vertex in kwargs['track'].vertices:
          title = title + vertex.reaction + '\n'
    ax.set_title(title, y=1.0, pad=-14)
  return fig



def hist(data, bins=50, title=None):
  fig = plt.figure(figsize=(12,12))
  ax = fig.gca()
  ax.hist(data, bins=bins)
  if title:
    ax.set_title(title)
  plt.show()


def analyze_time(hits):
  amplis = hits[:]['x']
  times = hits[:]['y']
  print('Arrival times: {0}'.format(len(times)))
  hist(times, bins=100, title='Arrival time')
  fig = plt.figure(figsize=(12,12))
  ax = fig.gca()
  ax.scatter(amplis, times, s=1)
  plt.show()


if __name__ == "__main__":
  

  fig = plt.figure(figsize=(14,12))
  ax = fig.add_axes([0.05, 0.05, 0.85, 0.9], projection='3d')
  ax_cb = fig.add_axes([0.9, 0.05, 0.05, 0.9])
  data = np.zeros(shape=(6,6,6))
  data[0,0,0] = 1
  data[0,0,5] = 2
  data[0,5,0] = 3
  data[0,5,5] = 4
  data[5,0,0] = 5
  data[5,0,5] = 6
  data[5,5,0] = 7
  data[5,5,5] = 8
  voxels(fig, ax, ax_cb, data, scale=10, spaced = True)
  ax.set_xlim([-30, 30])
  ax.set_ylim([-30, 30])
  ax.set_zlim([-30, 30])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.view_init(-35, 45)
  plt.show()
