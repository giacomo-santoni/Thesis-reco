#!/usr/bin/env python3
#**************************************************************************
#*                                                                        *
#*   AUTHOR: nicolo.tosi@bo.infn.it                                       *
#*                                                                        *
#*   PORTIONS OF THIS CODE ARE:                                           *
#*   Copyright (c) 2017 Keith Sloan <keith@sloan-home.co.uk>              *
#*             (c) Dam Lambert 2020                                       *
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

from lxml import etree

from geometry import transform, Volume, Sensor, Mask, Matrix, Camera, Fiducial, Geometry


def unit_multiplier(unit) :
  if unit == 'mm' or unit == None: return 1.
  elif unit == 'cm' : return 10.
  elif unit == 'm' : return 1000.
  elif unit == 'um' : return 0.001
  elif unit == 'nm' : return 0.000001
  elif unit == 'dm' : return 100.
  elif unit == 'm' : return 1000.
  elif unit == 'km' : return 1000000.
  elif unit == 'deg' : return np.pi / 180.
  else:
    print('unit not understood : ' + unit)

def parse_rotation(rotnode, scope = dict()):
  rot = np.eye(3)
  x = rotnode.get('x')
  if x is not None:
    angle = -eval(x, {"__builtins__": {}}, scope)
    mult = unit_multiplier(rotnode.get('unit'))
    s = np.sin(angle * mult)
    c = np.cos(angle * mult)
    xrot = np.eye(3)
    xrot[1][1] = c
    xrot[1][2] = -s
    xrot[2][1] = s
    xrot[2][2] = c
    rot = np.matmul(rot, xrot)
  y = rotnode.get('y')
  if y is not None:
    angle = -eval(y, {"__builtins__": {}}, scope)
    mult = unit_multiplier(rotnode.get('unit'))
    s = np.sin(angle * mult)
    c = np.cos(angle * mult)
    yrot = np.eye(3)
    yrot[0][0] = c
    yrot[0][2] = s
    yrot[2][0] = -s
    yrot[2][2] = c
    rot = np.matmul(rot, yrot)
  z = rotnode.get('z')
  if z is not None:
    angle = -eval(z, {"__builtins__": {}}, scope)
    mult = unit_multiplier(rotnode.get('unit'))
    s = np.sin(angle * mult)
    c = np.cos(angle * mult)
    zrot = np.eye(3)
    zrot[0][0] = c
    zrot[0][1] = -s
    zrot[1][0] = s
    zrot[1][1] = c
    rot = np.matmul(rot, zrot)
  return rot

def parse_position(posnode, scope = dict()):
  mult = unit_multiplier(posnode.get('unit'))
  xyz = [mult * eval(posnode.get(a), {"__builtins__": {}}, scope) for a in 'xyz']
  return np.array(xyz)

def parse_scale(scalenode, scope = dict()):
  return np.eye(3)

def process_defines(define):
  constants = dict()
  for cdefine in define.findall('constant') :
    name = str(cdefine.get('name'))
    value = eval(cdefine.get('value'), {"__builtins__": {}}, constants)
    constants[name] = value
  for vdefine in define.findall('variable') :
    name = str(vdefine.get('name'))
    value = eval(vdefine.get('value'), {"__builtins__": {}}, constants)
    constants[name] = value
  for qdefine in define.findall('quantity') :
    name = str(qdefine.get('name'))
    value = eval(qdefine.get('value'), {"__builtins__": {}}, constants)
    mult = unit_multiplier(qdefine.get('unit'))
    constants[name] = value * mult
  xforms = dict()
  for xform in define.findall('position') :
    name = str(xform.get('name'))
    xforms[name] = transform(np.eye(3), parse_position(xform))
  for xform in define.findall('rotation') :
    name = str(xform.get('name'))
    xforms[name] = transform(parse_rotation(xform), [0, 0, 0])
  return constants, xforms

def process_materials(materials, constants):
  mdict = dict()
  for m in materials.findall('material'):
    name = m.get('name')
    props = dict()
    props['name'] = name
    for p in m.findall('property'):
      pname = p.get('name')
      try:
        pval = p.get('value')
        if pval is None:
          pref = p.get('ref')
          pval = constants[pref]
      except:
        pval = None
      props[pname] = pval
    mdict[name] = props
  return mdict

def parse_matrix(aux, constants):
  try:
    ncells = aux.xpath("//auxiliary[@auxtype='cellcount']")[0].get('auxvalue')
    cellsizenode = aux.xpath("//auxiliary[@auxtype='cellsize']")[0]
    unit = unit_multiplier(cellsizenode.get('auxunit'))
    cellsize = eval(cellsizenode.get('auxvalue'), {"__builtins__": {}}, constants) * unit
    celledgenode = aux.xpath("//auxiliary[@auxtype='celledge']")[0]
    unit = unit_multiplier(celledgenode.get('auxunit'))
    celledge = eval(celledgenode.get('auxvalue'), {"__builtins__": {}}, constants) * unit
    cellpitch = cellsize + celledge
    return Matrix((ncells, ncells), (cellsize, cellsize), (cellpitch, cellpitch))
  except:
    raise ValueError('Matrix is Invalid')

def parse_box(box, constants):
  mult = unit_multiplier(box.get('lunit'))
  x = eval(box.get('x'), {"__builtins__": {}}, constants)
  y = eval(box.get('y'), {"__builtins__": {}}, constants)
  z = eval(box.get('z'), {"__builtins__": {}}, constants)
  return np.asarray([x, y, z]) * mult

def parse_ellipse(ellipse, constants):
  mult = unit_multiplier(ellipse.get('lunit'))
  dx = eval(ellipse.get('dx'), {"__builtins__": {}}, constants)
  dy = eval(ellipse.get('dy'), {"__builtins__": {}}, constants)
  dz = eval(ellipse.get('dz'), {"__builtins__": {}}, constants)
  return np.asarray([dx, dy, dz]) * mult


def process_volume(volume, constants, materials, solids, volumes, geotree, defs, verbose=False):
  if verbose:
    print('Processing Volume', volume.get('name'))
  try:
    mref = volume.find('materialref').get('ref')
    geotree.material = materials[mref]
    sref = volume.find('solidref').get('ref')
  except:
    pass
  #If an object has auxiliary properties then it has relevant behaviour for us.
  #Unfortunately by the time we know it, the object has already been created,
  #therefore we need to "promote" the object to the correct subclass.
  for aux in volume.findall('auxiliary'):
    if aux.get('auxtype') == 'Sensor':
      if verbose:
        print('Found Sensor')
      geotree.__class__ = Sensor
      geotree.matrix = parse_matrix(aux, constants)
      box = solids.xpath('//box[@name="' + sref + '"]')[0]
      geotree.shape = parse_box(box, constants)[0:2]
    elif aux.get('auxtype') == 'Mask':
      if verbose:
        print('Found Mask')
      geotree.__class__ = Mask
      geotree.matrix = parse_matrix(aux, constants)
      box = solids.xpath('//box[@name="codedApertureMask_solid"]')[0]
      geotree.shape = parse_box(box, constants)[0:2]
      hole = parse_box(solids.xpath('//box[@name="hole"]')[0], constants)[0:2]
      geotree.holes = np.array([parse_position(p, constants)[0:2] for p in
                               solids.xpath('//multiUnion/multiUnionNode/position')])
      geotree.rank = (geotree.matrix.cell_count + 1) // 2
      if not np.allclose(hole, geotree.matrix.cell_size):
        raise ValueError('Mismatch of hole size, got', hole, 'expected', geotree.matrix.cell_size)
    elif aux.get('auxtype') == 'Camera':
      if verbose:
        print('Found Camera Assembly')
      geotree.__class__ = Camera
    elif aux.get('auxtype') == 'Fiducial':
      fiducialsolid = solids.find('.//*[@name="'+ sref+ '"]')   
      if verbose:
        print('Found Fiducial Volume')
      geotree.__class__ = Fiducial      
      if fiducialsolid.tag == 'eltube':
        ellipse = solids.xpath('//eltube[@name="' + sref + '"]')[0]
        geotree.shape = parse_ellipse(ellipse, constants) * 2.0 #ellipses have half dimensions
        geotree.voxelize_eltube(defs)
      elif fiducialsolid.tag == 'box':
        box = solids.xpath('//box[@name="' + sref + '"]')[0]
        geotree.shape = parse_box(box, constants) #boxes have full dimensions
        geotree.voxelize_box(defs)
      elif fiducialsolid.tag == 'intersection':
        for child in fiducialsolid.getchildren():
          try: 
            box = solids.xpath('//box[@name="'+ child.get('ref') +'"]')[0] 
          except:
            ellipse = solids.xpath('//eltube[@name="' + child.get('ref') + '"]')[0]        
        geotree.shape = parse_box(box, constants) 
        ellipseshape = parse_ellipse(ellipse, constants) * 2.0
        geotree.voxelize_intersection(defs, ellipseshape)
      else:
        raise Exception("Fiducial Volume shape not recognized")

      
  for physvol in volume.findall('physvol'):
    if verbose:
      print('Entering physvol:', physvol.get('name'))
    r3d = np.eye(3)
    v3d = np.zeros(3)
    rot = physvol.find('rotation')
    if rot is not None:
      r3d = parse_rotation(rot, constants)
    pos = physvol.find('position')
    if pos is not None:
      v3d = parse_position(pos, constants)
    xform = transform(r3d, v3d)
    parent = Volume(geotree, physvol.get('name'), xform)
    #physvol either has a volumeref in same file or a file tag
    try:
      vname = physvol.find('volumeref').get('ref')
      if verbose:
        print('Found volume ref:', vname)
      for vol in volumes:
        if vol.get('name') == vname:
          process_volume(vol, constants, materials, solids, volumes, parent, defs)
    except:
      fname = physvol.find('file').get('name')
      if verbose:
        print('Found file ref:', fname)
      gdml_import(fname, parent, defs)
  geotree.build()

def gdml_import(filename, geotree, defs):
  """This function is not meant to import a generic gdml, only to find the
     relevant parts to place the objects relevant to reconstruction: sensors,
     masks, and sensitive volume boundaries.
     Because of this, things like materials are mostly ignored."""
  parser = etree.XMLParser(resolve_entities=True)
  root = etree.parse(filename, parser=parser)
  #print('Parsed', filename)
  constants, xforms = process_defines(root.find('define'))
  materials = process_materials(root.find('materials'), constants)
  solids = root.find('solids')
  worldref = root.find('setup').find('world').get('ref')
  structure = root.find('structure')
  volumes = structure.findall('volume') + structure.findall('assembly')
  for volume in volumes:
    if volume.get('name') == worldref:
      process_volume(volume, constants, materials, solids, volumes, geotree, defs)
      break

def print_volume_recursive(volume, depth=0):
  print('{0}+-{1}: {2}'.format(' ' * depth, volume.name, type(volume)))
  depth += 2
  for name in volume.children:
    print_volume_recursive(volume.children[name], depth)

def read_gdml_geometry(gdmlfile, defs, verbose=False):
  #in case the gdml has relative paths
  import os
  oldpath = os.getcwd()
  newpath = os.path.dirname(gdmlfile)
  os.chdir(newpath)
  #read geometry description
  root = Volume(None, 'world-root', np.eye(4))
  #gdml_import.gdml_import(gdmlfile, root)
  gdml_import(gdmlfile, root, defs)
  os.chdir(oldpath)
  return Geometry(root, defs, verbose)



if __name__ == "__main__":
  import os
  os.chdir('./geometry')
  filename = 'main.gdml'
  print('Importing GDML from', filename)
  root = Volume(None, 'world-root', np.eye(4))
  defs = {}
  defs['voxel_size'] = 10
  gdml_import(filename, root, defs)
  print_volume_recursive(root)
