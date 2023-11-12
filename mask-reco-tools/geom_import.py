import numpy as np
import os
import sys

import gdml_import
import geometry



def get_fiducial_recursive(volume):
  print(volume.name, type(volume))
  if isinstance(volume, geometry.Fiducial):
    return volume
  else:    
    for name in volume.children:   
      return get_fiducial_recursive(volume.children[name])

def load_fiducial(geopath, defs):
  #oldpath = os.getcwd()
  newpath = os.path.dirname(geopath + '/main.gdml')
  os.chdir(newpath)
  root = geometry.Volume(None, 'world-root', np.eye(4))
  gdml_import.gdml_import('main.gdml', root, defs)
  fiducial = geometry.collect_by_type(root, geometry.Fiducial)
  return fiducial

def load_geometry(geopath, defs):
  oldpath = os.getcwd()
  newpath = os.path.dirname(geopath + '/main.gdml')
  os.chdir(newpath)
  root = geometry.Volume(None, 'world-root', np.eye(4))
  gdml_import.gdml_import('main.gdml', root, defs) # main.gdml
  geom = geometry.Geometry(root, defs)
  os.chdir(oldpath)
  return geom


if __name__ == '__main__':
  defs = {}
  defs['voxel_size'] = 10
  fiducial = load_fiducial("./geometry", defs)
  print(fiducial)