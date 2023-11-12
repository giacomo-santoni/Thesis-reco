import numpy as np 
import ROOT
import sys

#import pytraversal


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



def loadPrimariesEdepSim(fname, ev):
  tfile = ROOT.TFile.Open(fname)
  events = tfile.Get('EDepSimEvents')
  ROOT.gSystem.Load("libGeom")
  ROOT.TGeoManager.Import(fname)
  # new GRAIN geometry 
  ROOT.gGeoManager.cd('/volWorld_PV_1/rockBox_lv_PV_0/volDetEnclosure_PV_0/volSAND_PV_0/MagIntVol_volume_PV_0/sand_inner_volume_PV_0/GRAIN_lv_PV_0/GRAIN_Ext_vessel_outer_layer_lv_PV_0/GRAIN_Honeycomb_layer_lv_PV_0/GRAIN_Ext_vessel_inner_layer_lv_PV_0/GRAIN_gap_between_vessels_lv_PV_0/GRAIN_inner_vessel_lv_PV_0/GRIAN_LAr_lv_PV_0')
  local = np.array([0.,0.,0.])
  master = np.array([0.,0.,0.])
  ROOT.gGeoManager.LocalToMaster(local, master)
  event_list = []
  events.GetEntry(ev)
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
  return Event(event.EventId, vertices, np.asarray(hits))

def load_primaries(fname):
  tfile = ROOT.TFile.Open(fname)
  l = tfile.primaries.GetEntries()
  positions = np.zeros((l, 2, 3))
  for entries in tfile.primaries:
    positions[entries.idEvent] = ((entries.xVertex, entries.yVertex, entries.zVertex),
                                  (entries.px[0], entries.py[0], entries.pz[0]))
  return positions


def loadEnergyDeposits(event, datashape, geom, args):
  shape = geom.fiducial.voxels.shape
  if args.maskvoxels:
    bound_min = np.matmul(geom.transforms.xform_fv, [0,5,5,1]) - geom.fiducial.voxel_size / 2
    bound_max = np.matmul(geom.transforms.xform_fv, [shape[0], shape[1] - 6 , shape[2] - 6 ,1]) + geom.fiducial.voxel_size / 2
    # grid = pytraversal.Grid3D(bound_min[0:3], bound_max[0:3], datashape)  
  else:
    bound_min = np.matmul(geom.transforms.xform_fv, [0,0,0,1]) - geom.fiducial.voxel_size / 2
    bound_max = np.matmul(geom.transforms.xform_fv, [shape[0], shape[1] , shape[2] ,1]) + geom.fiducial.voxel_size / 2
    # grid = pytraversal.Grid3D(bound_min[0:3], bound_max[0:3], geom.fiducial.voxels.shape)  
  
  deposits = np.zeros(datashape)
  size = np.array([deposits.shape[0], deposits.shape[1], deposits.shape[2]])
  print(size)
  for hit in event.hits:
    start = np.array([0,0,0,1]) 
    start[0:3] =+ hit.start 
    start = np.matmul(geom.transforms.xform_fg, start)
    stop = np.array([0,0,0,1]) 
    stop[0:3] =+ hit.stop 
    stop = np.matmul(geom.transforms.xform_fg, stop)
  #   traversed = grid.traverse(start[0:3],stop[0:3])
  #   for i,j,k in traversed:
  #     deposits[i,j,k] += hit.energy / len(traversed) 
  # return deposits


if __name__ == '__main__':
  fname = sys.argv[1]
  primaries = loadPrimariesEdepSim(fname, 20)
  print(primaries)