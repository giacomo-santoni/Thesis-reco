import numpy as np
import ROOT
import sys
import pickle
import os 
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


import geom_import
import plot3d


def load_sensor_data(inFile, klist, evn, xform):
  ph_pos = []
  for treekey in klist:
    if (treekey.GetClassName() == "TNamed"):
      continue
    phcount = 0
    treename = treekey.GetName()
    tree = inFile.Get(treename)
    tree.GetEntry(evn)
    print(treename, len(tree.energy))
    for i in range(len(tree.energy)):
      xOrigin = tree.xOrigin[i]
      yOrigin = tree.yOrigin[i]
      zOrigin = tree.zOrigin[i]
      t = tree.time[i]
      if t <= 1000:
        origin_global = (xOrigin, yOrigin, zOrigin, 1)
        origin_fiducial = np.matmul(xform, origin_global)
        ph_pos.append(origin_fiducial)
        phcount +=1
    #print(treename, phcount)  
  return ph_pos


def make_photonmap(photons, geom):
  photons = np.array(photons)[:, 0:3]
  xform = geom.transforms.xform_fv
  # cx = ( [0, 0, 0, 1], [geom.fiducial.voxels.shape[0], 0, 0, 1] ) 
  # cy = ([0, 1, 0, 1] , [0, geom.fiducial.voxels.shape[1], 0, 1])
  # cz = ([0, 0, 1, 1], [0, 0, geom.fiducial.voxels.shape[2], 1])
  rangelist = []
  for axis in range(3):
    c0 = [0,0,0,1]
    c0[axis]+= 1
    r0 = np.matmul(xform, c0)[axis] - geom.fiducial.voxel_size /2
    c1 = [0,0,0,1]
    c1[axis]+= geom.fiducial.voxels.shape[axis]-1
    r1 = np.matmul(xform, c1)[axis] + geom.fiducial.voxel_size /2
    rangelist.append((r0,r1)) 
  hist, _ = np.histogramdd(photons, bins = geom.fiducial.voxels.shape, range=rangelist)
  return hist
 
def load_photonmap(fname, evn, geom):
  inFile = ROOT.TFile.Open(fname, "READ")
  klist = inFile.GetListOfKeys()
  photons = load_sensor_data(inFile, klist, evn, geom.transforms.xform_fg)
  photonmap = make_photonmap(photons, geom)
  return photonmap


if __name__ == '__main__':
  
  fname ='./data/initial-data/sensors.root'
  geopath ='./geometry'
  defs={}
  defs['voxel_size'] = 50
  
  
  inFile = ROOT.TFile.Open(fname, "READ")
  klist = inFile.GetListOfKeys()
  nEvents = inFile.Get(klist.Last().GetName()).GetEntries()
  geom = geom_import.load_geometry(geopath, defs)

  for evn in [265]:
    photons = load_sensor_data(inFile, klist, evn, geom.transforms.xform_fg)
  
  print("number of detected photons: ", len(photons))
  hist = make_photonmap(photons, geom)
  
  plot3d.plot3d(geom, voxels = hist, cut_low =1, cut_up = np.amax(hist), title = "photon distribution", alpha = 1)#ho tolto geom.fiducial
  plt.show()
