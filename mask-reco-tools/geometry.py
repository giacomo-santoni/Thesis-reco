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


def transform(rot, trans):
  xform = np.column_stack([rot, trans])
  return np.vstack([xform, [0, 0, 0, 1]])


class Matrix:
  def __init__(self, count_or_array, size, pitch=None):
    self.cell_count = np.asarray(count_or_array, dtype=int)
    if self.cell_count.shape != (2,):
      self.array = np.asarray(count_or_array, dtype=float)
      self.cell_count = np.asarray(self.array.shape, dtype=int)
    self.cell_size = np.asarray(size, dtype=float)
    if pitch is None:
      self.cell_pitch = self.cell_size
    else:
      self.cell_pitch = np.asarray(pitch, dtype=float)

class Volume:
  """
  Base class for geometry objects.
  An instance of this class represents a physical instance of one volume.

  It has an associated name, which, when collated with that of the parent
  volumes, forms a unique identifier in the geometry.
  The path() function returns such a string.

  It also has a transform which describes its local coordinate system
  w.r.t. its parent.
  The convenience function gxform() returns the global transform to the
  root object.
  If T = self.xform, then T*(0,0,0,1) is the local origin in parent
  coordinates.

  An optional material is merely a dictionary of properties of the
  constituent material. None does not indicate a vacuum, but the lack
  of actual physical form. The parent material would apply if any.
  If a parent and children have different materials, the parent material
  exists anywhere inside the parent volume that isn't also inside the
  children volume.

  Finally, shape describes the actual physical extent of the bounding box of the
  volume regardless of its geometrical shape. 
  """
  def __init__(self, parent, name, xform, shape=None, material=None):
    self.parent = parent
    self.children = dict()
    self.name = name
    self.xform = xform
    self.shape = shape
    self.material = material
    if parent is not None:
      parent.children[name] = self

  def path(self):
    if self.parent is None:
      return '//' + self.name
    else:
      return self.parent.path() + '/' + self.name

  def gxform(self):
    if self.parent is None:
      return self.xform
    else:
      return np.matmul(self.parent.gxform(), self.xform)

  #Call after children parsing to finish constructing the object
  def build(self):
    pass

class Sensor(Volume):
  """Geometry class representing a photodetector."""
  def __init__(self, parent, name, xform, shape=None, material=None):
    Volume.__init__(self, parent, name, xform, shape, material)

class Mask(Volume):
  """Geometry class representing a Coded Aperture Mask."""
  def __init__(self, parent, name, xform, shape=None, material=None):
    Volume.__init__(self, parent, name, xform, shape, material)

class Camera(Volume):
  """
  A combination of a Coded Aperture Mask and a pixelated photodetector
  """
  def __init__(self, parent, name, xform, shape=None, material=None):
    Volume.__init__(self, parent, name, xform, shape, material)

  def build(self):
    for c in self.children:
      if isinstance(self.children[c], Mask):
        self.mask = self.children[c]
      elif isinstance(self.children[c], Sensor):
        self.sensor = self.children[c]
    if not hasattr(self, 'mask') or not hasattr(self, 'sensor'):
      raise ValueError("Camera is incomplete:", self.name)
    #Cameras should have sensors and masks with the same orientation and no scaling
    m = self.mask.gxform()
    s = self.sensor.gxform()
    if not np.allclose(m[0:2, 0:2], s[0:2, 0:2]):
      raise ValueError("Mask and Sensor have inconsistent transforms in camera", self.name)

class Fiducial(Volume):
  """Geometry class representing the active volume."""
  def __init__(self, parent, name, xform, shape=None, material=None):
    Volume.__init__(self, parent, name, xform, shape, material)

  def build(self):
    pass

  def voxelize_box(self, defs):
    self.voxel_size = defs['voxel_size']
    shape = np.ceil(self.shape * 1 / self.voxel_size).astype(int) #cut 10% on each side off the fiducial
    self.voxels = np.ones(shape=shape, dtype=np.int8)
    self.volume = np.sum(self.voxels)
    hx = shape[0] / 2.
    hy = shape[1] / 2.
    hz = shape[2] / 2.
    vmx = np.eye(4)
    vmx[0, 0] = vmx[1, 1] = vmx[2, 2] = self.voxel_size
    vmx[0, 3] = (-hx + 0.5) * self.voxel_size
    vmx[1, 3] = (-hy + 0.5) * self.voxel_size
    vmx[2, 3] = (-hz + 0.5) * self.voxel_size
    self.voxel_mapping_xform = vmx

  def voxelize_eltube(self, defs):
    self.voxel_size = defs['voxel_size']
    shape = np.ceil(self.shape * 1 / self.voxel_size).astype(int)
    print("fiducial voxels shape: ", shape )
    ellipse = np.zeros(shape=shape[:-1], dtype=np.int8)
    hx = shape[0] / 2.
    hy = shape[1] / 2.
    hz = shape[2] / 2.
    for ix, iy in np.ndindex(ellipse.shape):
      y = iy - hy + 0.5
      x = ix - hx + 0.5
      if y**2 / hy**2 + x**2 / hx**2 <= 1:
        ellipse[ix, iy] = 1
    self.voxels = np.broadcast_to(ellipse[..., None], shape=shape).copy() #copy is needed or cl.Buffer() creation fails
    self.volume = np.sum(self.voxels)
    vmx = np.eye(4)
    vmx[0, 0] = vmx[1, 1] = vmx[2, 2] = self.voxel_size
    vmx[0, 3] = (-hx + 0.5) * self.voxel_size
    vmx[1, 3] = (-hy + 0.5) * self.voxel_size
    vmx[2, 3] = (-hz + 0.5) * self.voxel_size
    self.voxel_mapping_xform = vmx 

  def voxelize_intersection(self, defs, eltube_size):
    """the geometry intersecting solids (box, eltube) MUST be centered on the same point"""
    self.voxel_size = defs['voxel_size']
    box_shape = np.ceil(self.shape * 1 / self.voxel_size).astype(int)
    box_cut = np.zeros(shape = box_shape[:-1], dtype=np.int8)
    self.voxels = np.ones(shape=box_shape, dtype=np.int8)
    hx = box_shape[0] / 2.
    hy = box_shape[1] / 2.
    hz = box_shape[2] / 2.
    vmx = np.eye(4)
    vmx[0, 0] = vmx[1, 1] = vmx[2, 2] = self.voxel_size
    vmx[0, 3] = (-hx + 0.5) * self.voxel_size
    vmx[1, 3] = (-hy + 0.5) * self.voxel_size
    vmx[2, 3] = (-hz + 0.5) * self.voxel_size
    self.voxel_mapping_xform = vmx
    el_hx = eltube_size[0] / 2.
    el_hy = eltube_size[1] / 2.
    for ix, iy in np.ndindex(box_cut.shape):
      vx_c = np.matmul( vmx, [ix, iy, 0, 1])
      x = vx_c[0] 
      y = vx_c[1] 
      if y**2 / el_hy**2 + x**2 / el_hx**2 <= 1:
        box_cut[ix, iy] = 1
    self.voxels = np.broadcast_to(box_cut[..., None], shape=box_shape).copy()
    self.volume = np.sum(self.voxels)
    print("fiducial voxels volume: ", self.volume)
    
    

def collect_by_type(node, Type, deep=False):
  collection = [] 
  def get_type(node, coll, deep):
    found = False
    if isinstance(node, Type):
      coll.append(node)
      found  = True
    if deep or not found:
      for name in node.children:
        get_type(node.children[name], coll, deep)
  get_type(node, collection, deep)
  return collection


class Geometry:
  def __init__(self, georoot, defs, verbose=True):
    self.root = georoot
    self.cameras = collect_by_type(georoot, Camera)
    if verbose:
      print('Loaded geometry with {0} cameras'.format(len(self.cameras)))
       
    self.fiducial = collect_by_type(georoot, Fiducial)
    if len(self.fiducial) != 1:
      raise ValueError('There should be one and only one fiducial: found', len(self.fiducial))
    self.fiducial = self.fiducial[0]
    # self.fiducial.voxelize(defs)    
    global_xform_fiducial = self.fiducial.gxform()
    global_inverse_xform = np.linalg.inv(global_xform_fiducial)
    global_xform_voxel = np.matmul(global_xform_fiducial, self.fiducial.voxel_mapping_xform)
    #Check inversion
    if not np.allclose(np.matmul(global_xform_fiducial, global_inverse_xform), np.eye(4)):
      raise ValueError('Could not invert global transform', global_xform_fiducial)
    #switch to transform_t, class just for naming clarity
    
    class _Transforms:
      pass
    self.transforms = _Transforms()
    self.transforms.xform_gf = global_xform_fiducial
    self.transforms.xform_fg = global_inverse_xform
    self.transforms.xform_fv = self.fiducial.voxel_mapping_xform
    self.transforms.xform_vf = np.linalg.inv(self.fiducial.voxel_mapping_xform)
    self.transforms.xform_gv = global_xform_voxel
    