import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt 

import argparse
import pickle
import sys

from plot3d import plot3d
from mctruth import loadPrimariesEdepSim, loadEnergyDeposits, load_primaries
from recoprocessing import equalizeHistogram, compute_ssim
from photonmap import load_photonmap
import geom_import

  
def parseArgs():
  parser = argparse.ArgumentParser()
  parser.add_argument('recofile', help = '.pkl reconstruction output file')
  parser.add_argument('evn', help = 'event number to display')
  parser.add_argument('geometryPath', help = 'geometry folder path')
  parser.add_argument('-pg', '--primaryGEANT', help ='.root primary particles root file to display tracks') 
  parser.add_argument('-pe', '--primaryEDEPSIM', help ='.root primary particles edepsim root file to display track') 
  parser.add_argument('-e', '--energy', help ='.pkl energy deposits file')
  parser.add_argument('-c', '--cut', help = 'set display amplitude lower treshold')
  parser.add_argument('-uc', '--uppercut', help='set display amplitude upper threshold')
  parser.add_argument('-m', '--maskvoxels', action="store_true", help = "mask voxels on fiducial volume edges")
  parser.add_argument('-ph', '--photonmap', help='diplay 3d map of detected photons from sensors.root file')
  #parser.add_argument('-cam', '--cameraOutput', action='store_true', help='camera recontruction is stored separately in output file')
  parser.add_argument('-s', '--save', help="save file name")
  parser.add_argument('-ssim', '--ssim', action ='store_true', help='compute structural similarity index with photon map')
  #parser.add_argument('-pca', '--pca', action='store_true', help="execute principal component analysis")
  parser.add_argument('-it', '--iteration', help=" display iteration number"  )
  parser.add_argument('-eq', '--equalize', action='store_true', help="equalize histogram" )
  parser.add_argument('-v', '--voxelsize', help="reco voxel size, default: 10 mm")
  args = parser.parse_args()
  return args

def define_cuts(data, args):
  maxampl = np.amax(recodata)   
  if args.cut and args.equalize:
    cut = float(args.cut) * maxampl
  elif args.cut:
    cut = float(args.cut)
  else:
    cut = 0.95 * maxampl
  if args.uppercut and args.equalize:
    uppercut = float(args.uppercut) * maxampl
  elif args.uppercut:
    uppercut = float(args.uppercut)
  else: 
    uppercut =  maxampl
  return cut, uppercut

def plot_histo(vox, t=False):
  plt.figure()
  plt.hist(vox.flatten(), bins=200)
  plt.xlabel("voxel amplitude")
  plt.yscale("log")
  plt.ylabel("n")
  
def preprocess_data(data, args):
  if args.maskvoxels:
    data = data[5:-5,5:-5,5:-5]
  if args.equalize:
    data = equalizeHistogram(data).reshape(data.shape)
  return data

def get_primaries(evn, args): 
  if args.primaryGEANT:
    primaryfile = args.primaryGEANT
    MCtruth = load_primaries(primaryfile, evn)
    primarytype = 'root'
  elif args.primaryEDEPSIM:
    primaryfile = args.primaryEDEPSIM
    MCtruth = loadPrimariesEdepSim(primaryfile, evn)
    primarytype = 'edepsim'
  else:
    MCtruth = None
    primaryfile = None
    primarytype = None
  return MCtruth, primarytype 

def load_pickle(fname, evn):
    data = np.load(fname, allow_pickle = True)
    uuuid = data.keys()
    for u in uuuid:
        eventreco = data[u][evn]
    return eventreco

def gradient(data, args):
  recodata = preprocess_data(data, args)
  gx_recodata, gy_recodata, gz_recodata = np.gradient(recodata)
  gx_recodata, gy_recodata, gz_recodata = np.asarray(gx_recodata), np.asarray(gy_recodata), np.asarray(gz_recodata)
  g_recodata = np.abs(np.sqrt((gx_recodata**2)+(gy_recodata**2)+(gz_recodata**2)))
  return g_recodata





if __name__ == '__main__':
    args = parseArgs()   
    defs = {}
    defs['voxel_size'] = 10
    if args.voxelsize:
      defs['voxel_size'] = int(args.voxelsize)
    
    geometryPath = args.geometryPath
    fpkl = args.recofile
    evn =int(args.evn) 
    cameraOutput = False
       
    """load data""" 
    recodata = load_pickle(fpkl, evn)
    print(recodata.keys())
    
    if args.iteration:
      it = int(args.iteration)
      recodata = recodata[it]
      
    #recodata, volume = mask_voxels(data)
    print("DATA SHAPE", recodata.shape)
    print("DATA MAX", np.amax(recodata))
    print("VOXEL SUM", np.sum(recodata))

    
    """load fiducial volume from geometry and get gxform""" 
    geom = geom_import.load_geometry(geometryPath, defs)
    fiducial = geom.fiducial
    
    MCtruth, primarytype = get_primaries(evn, args)
   
    recodata = preprocess_data(recodata, args)
    g_recodata = gradient(recodata, args)

    cut, uppercut = define_cuts(recodata, args)
    #sys.exit(0)



    scale = 1
    if args.equalize:
      scale=100
    showcameras = False

    if args.photonmap:
      photonmap = load_photonmap(args.photonmap, evn, geom)
      if args.maskvoxels:
        photonmap = photonmap[:,5:-5,5:-5 ]
      plot3d(geom, voxels = photonmap, cut_low =1, cut_up = np.amax(photonmap), frames= True, title = "photon distribution", alpha = 1) 
      if args.ssim:
        ssim = compute_ssim(recodata, photonmap)
        print("structural similarity index: ", ssim)  
    
    if MCtruth is not None:
      plot3d(geom, voxels = g_recodata * scale, cut_low = cut*scale, cut_up = uppercut * scale, inputType = primarytype, track = MCtruth,  title = "Reconstructed voxels", frames=True, alpha = 0.4)     
      plot3d(geom, voxels = recodata * scale, cut_low = cut*scale, cut_up = uppercut * scale, inputType = primarytype, track = MCtruth,  title = "Reconstructed voxels", frames=True, alpha = 0.4) 
      # if primarytype=='edepsim':
      #   Edeposits = loadEnergyDeposits(MCtruth, recodata.shape, geom, args )
      #   print("edeposit sum ", np.sum(Edeposits))
      #   plot3d(geom, voxels = Edeposits, cut_low = 0.01, cut_up = np.amax(Edeposits), inputType = primarytype, track = MCtruth,  title = "Reconstructed voxels", frames=True, alpha = 0.4)     
      #   if args.ssim:
      #     ssim = compute_ssim(recodata, Edeposits)
      #     print("structural similarity index with edep: ", ssim )
    else:
      plot3d(geom, voxels = g_recodata* scale, cut_low = cut* scale, cut_up = uppercut* scale, frames=True , alpha = 0.4)
      plot3d(geom, voxels = recodata* scale, cut_low = cut* scale, cut_up = uppercut* scale, frames=True , alpha = 0.4)

    if args.save:
      #plt.legend(["This is my legend"], fontsize="x-small")
      plt.savefig(args.save)
    else:
      plt.show()
    


