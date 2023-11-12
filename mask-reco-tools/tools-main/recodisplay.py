from cv2 import vconcat
import numpy as np
import ROOT
import sys
from matplotlib import pyplot as plt 
import argparse
import os

from displaylib.plot3d import plot3d
sys.path.insert(1, os.path.join(sys.path[0], '../volumereco'))
import gdml_import
import geometry

class Particle:
  def __init__(self, PDGCode, momentum):
    self.PDGCode = PDGCode
    self.momentum = momentum
    
    if PDGCode == 13 or PDGCode == -13:
      self.mass = 105.7
    elif PDGCode == 11 or PDGCode == -11:
      self.mass = 0.511
    elif PDGCode == 211 or PDGCode == -211:
      self.mass = 139.6
    elif PDGCode == 111:
      self.mass = 135
    elif PDGCode == 2112:
      self.mass = 939.6
    elif PDGCode == 2212:
      self.mass = 938.3
    else:
      self.mass = 100

class Vertex:
  def __init__(self, reaction, position, particles):
    self.reaction = reaction
    self.position = position
    self.particles = particles

class Hit:
  def __init__(self, start, stop, energy):
    self.start = start
    self.stop = stop
    self.energy = energy

class Event:
  def __init__(self, eventID, vertices, hits):
    self.eventID = eventID
    self.vertices = vertices
    self.hits = hits


  def print(self):
    print("Checking event {}, it has {} vertices and {} energy deposits.".format(self.eventID, len(self.vertices), len(self.hits)))
    for idx, vertex in enumerate(self.vertices):
      print("  Checking vertex {}, it has: ".format(idx))
      print("    reaction in {}".format(vertex.reaction))
      print("    position in {}".format(vertex.position))
      print("    {} particles".format(len(vertex.particles)))
      for idx2, particle in enumerate(vertex.particles):
        print("      Checking particle {}, it has: ".format(idx2))
        print("        PDGCode {}".format(particle.PDGCode))
        print("        momentum {}".format(particle.momentum/particle.mass))


def load_primaries_edepsim(fname):
  tfile = ROOT.TFile.Open(fname)
  events = tfile.Get('EDepSimEvents')

  ROOT.gSystem.Load("libGeom")
  ROOT.TGeoManager.Import(fname)
    
  # new GRAIN geometry 
  ROOT.gGeoManager.cd('/volWorld_PV_1/rockBox_lv_PV_0/volDetEnclosure_PV_0/volSAND_PV_0/' +
                      'MagIntVol_volume_PV_0/sand_inner_volume_PV_0/GRAIN_lv_PV_0/GRAIN_Ext_vessel_outer_layer_lv_PV_0/' +
                      'GRAIN_Honeycomb_layer_lv_PV_0/GRAIN_Ext_vessel_inner_layer_lv_PV_0/' +
                      'GRAIN_gap_between_vessels_lv_PV_0/GRAIN_inner_vessel_lv_PV_0/GRAIN_LAr_lv_PV_0')

  local = np.array([0.,0.,0.])
  master = np.array([0.,0.,0.])
  ROOT.gGeoManager.LocalToMaster(local, master)

  event_list = []
  for i in range(events.GetEntries()):
    events.GetEntry(i)
    event = events.Event
    vertices = []
    for idx, vertex in enumerate(event.Primaries):
      reaction = vertex.GetReaction()
      position = np.array([vertex.GetPosition().X(), vertex.GetPosition().Y(), vertex.GetPosition().Z()])
      position = position - master
      particles = []
      for idx2, particle in enumerate(vertex.Particles):
        momentum = np.array([particle.GetMomentum().Px(), particle.GetMomentum().Py(), particle.GetMomentum().Pz()])
        PDGCode = particle.GetPDGCode()
        particles.append(Particle(PDGCode, momentum))
      vertices.append(Vertex(reaction, position, particles))
    # Load energy deposits
    hits = []
    for idx, volume in enumerate(event.SegmentDetectors):
      if (volume[0] == "LArHit"):
        for idx2, hit in enumerate(volume[1]):
          start = np.array([hit.GetStart().X() - master[0], hit.GetStart().Y() - master[1], hit.GetStart().Z() - master[2]])
          stop  = np.array([hit.GetStop().X()  - master[0], hit.GetStop().Y()  - master[1], hit.GetStop().Z()  - master[2]])
          energy = hit.GetSecondaryDeposit()
          hits.append(Hit(start, stop, energy))

    event_list.append(Event(event.EventId, vertices, np.asarray(hits)))
  
  
  # for event in event_list:
  #     event.print()
  return event_list


def mask_voxels(data):
  shape = data.shape
  ellipse_mask = np.zeros(shape = shape[:-1])
  hx = shape[0] / 2.
  hy = shape[1] / 2.
  hz = shape[2] / 2.
  for ix, iy in np.ndindex(ellipse_mask.shape):
    y = iy - hy + 0.5
    x = ix - hx + 0.5
    if y**2 / (hy - 5)**2 + x**2 / (hx-5)**2 <= 1:  
      ellipse_mask[ix, iy] = 1
  mask = np.broadcast_to(ellipse_mask[..., None], shape=shape).copy() #copy is needed or cl.Buffer() creation fails
  print("volume = ", np.sum(mask))
  data *= mask
  return data, np.sum(mask)


def load_primaries(fname):
  tfile = ROOT.TFile.Open(fname)
  l = tfile.primaries.GetEntries()
  positions = np.zeros((l, 2, 3))
  for entries in tfile.primaries:
    positions[entries.idEvent] = ((entries.xVertex, entries.yVertex, entries.zVertex),
                                  (entries.px[0], entries.py[0], entries.pz[0]))
  return positions


def load_pickle(fname, evn):
    data = np.load(fname, allow_pickle = True)
    uuuid = data.keys()
    for u in uuuid:
      print(data[u])
      shaep = data[u][evn]['CAM_NE_X2_Y0']['amplitude'].shape
      print(shaep)
      eventreco = np.zeros(shape = shaep, dtype = np.float32())
      cams = data[u][evn].keys()
      remove_cams = []
      for c in cams:
        if c in remove_cams:
          print("skipped cam: ", c)
          continue
        else:
          eventreco += data[u][evn][c]['amplitude']
    return eventreco


def load_fiducial(geopath, defs):
    oldpath = os.getcwd()
    newpath = os.path.dirname(geopath + '/main.gdml')
    print(newpath)
    os.chdir(newpath)
    #read geometry description 
    root = geometry.Volume(None, 'world-root', np.eye(4))
    gdml_import.gdml_import('main.gdml', root)
    #gdml_import.print_volume_recursive(root)
    #COLDDEMO:
    #lar_phy = root.children['vessel_ext_physical'].children['dewar_thickness_physical'].children['lar_physical']
    #GRAIN:
    lar_phy = root.children['vessel_ext_physical'].children['air_physical'].children['vessel_int_physical'].children['lar_physical']
    #print(lar_phy.shape)
    lar_phy.voxelize(defs)
    #print(lar_phy.voxel_size)
    print(lar_phy.voxel_mapping_xform)
    #print(lar_phy.voxels.shape)
    print(lar_phy.gxform())
    return lar_phy


def loadEnergyDeposits(fname, data, voxelSize):
  deposits = np.zeros_like(data)
  size = np.array([deposits.shape[2], deposits.shape[1], deposits.shape[0]])
  for hit in fname.hits:
    deposit_center = ((hit.start + hit.stop) / 2)
    deposit_center = deposit_center / voxelSize
    deposit_center = deposit_center + (size / 2)
    deposit_ampl = hit.energy
    deposits[int(-deposit_center[2])][int(deposit_center[1])][int(deposit_center[0])] = deposits[int(deposit_center[2])][int(deposit_center[1])][int(deposit_center[0])] + deposit_ampl
  return deposits

  
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('recofile', help = '.pkl reconstruction output file')
    parser.add_argument('evn', help = 'event number to display')
    #parser.add_argument('camname', help = 'cam name to display')
    parser.add_argument('-p', '--primary', help ='.root primary particles file to display track') 
    parser.add_argument('-c', '--cut', help = 'set display amplitude lower treshold')
    parser.add_argument('-uc', '--uppercut', help='set display amplitude upper threshold')
    #parser.add_argument('-ph', '--phase', action='store_true', help='diplay phase')
    args = parser.parse_args()

    defs = {}
    defs['voxel_size'] = 10
    # geometryPath = '/home/cicero/Projects/dune/sand-optical/geometry/'
    # geometryPath = '/home/valentina/projects/sandoptical/geometry/'
    geometryPath = '/mnt/c/Users/vpia/OneDrive - Alma Mater Studiorum Università di Bologna/Dottorato/Simulazioni/git/OpticalPhoton/geometries/grain3mm'

    # parse args 
    fpkl = args.recofile
    evn =int(args.evn)
    #camname = args.camname
    primaryfile = False
    if args.primary:
        primaryfile = args.primary
    #if args.phase:
    #    show = 'phase'
    #else:
    #    show ='amplitude'

    #load data 
    data = load_pickle(fpkl, evn)
    recodata, volume = mask_voxels(data)
    print("DATA SHAPE", recodata.shape)
    
    #load fiducial volume from geometry and get gxform 
    fiducial = load_fiducial(geometryPath, defs)


    #define cuts
    maxampl = np.amax(recodata)
    
    reco_cut_old = recodata[recodata > 0]

    x = []
    height = []
    for c in range(0, 199, 1):
      c2 = c / 200
      x.append(c2)
      cut = c2 *maxampl
      reco_cut = recodata[recodata > cut]
      print(reco_cut.shape[0] / volume)
      height.append(reco_cut.shape[0] / volume)
      print("{} / {}".format(c2, reco_cut_old.shape[0] / reco_cut.shape[0]))
      reco_cut_old = recodata[recodata > cut]


    x = np.asarray(x)
    height = np.asarray(height)
    ratio = (height[:-1] / height[1:])
    print(x)
    print(ratio)
    # plt.bar(x,height,width=0.005)
    # plt.ylabel("Number of voxels on")
    # plt.xlabel("Cut")
    # plt.yscale('log')
    # plt.show()

    uppercut = maxampl
    cut = 0.875 * maxampl
    if args.cut:
        cut = float(args.cut) * maxampl
    if args.uppercut:
        uppercut = float(args.uppercut) * maxampl
    
    showcameras = False

    if primaryfile:
      MCtruth = load_primaries_edepsim(primaryfile)
      energyDeposits = loadEnergyDeposits(MCtruth[evn], recodata, defs['voxel_size'])

      # plot3d(fiducial, voxels = recodata, cut_low = cut, cut_up = uppercut, track = MCtruth[evn], vertex = MCtruth[evn][0], frames= showcameras)
      plot3d(fiducial, voxels = recodata, cut_low = cut, cut_up = maxampl, inputType = 'edepsim', frames = True, title = "Reconstructed voxels", alpha = 0.1)

      plot3d(fiducial, voxels = energyDeposits, cut_low = 0.01, cut_up = np.amax(energyDeposits), track = MCtruth[evn], inputType = 'edepsim', frames = True, title = "Energy deposits", alpha = 1)

    else:
      plot3d(fiducial, voxels = recodata, cut_low = cut, cut_up = uppercut, frames = showcameras)
    
    plt.show()

