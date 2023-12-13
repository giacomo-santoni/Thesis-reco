import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import clusterclass
from recodisplay import load_pickle
from geom_import import load_geometry

all_parameters = {"epsilon" : 36, "min_points" : 6, "bandwidth" : 40, "stepsize" : 40, "cuts" : [100,500], "g_cut_perc" : 0.05, "it" : 500, "g_it" : 200}

def EventNumberList(fname):
   f = open(fname, "r")
   events_string = []
   all_events = []

   for x in f:
      events_string.append(x)
   
   for i in events_string:
    new_ev = "".join(x for x in i if x.isdecimal())
    all_events.append(int(new_ev))
   return all_events

def gradient(data):
  gx_recodata, gy_recodata, gz_recodata = np.gradient(data)
  gx_recodata, gy_recodata, gz_recodata = np.asarray(gx_recodata), np.asarray(gy_recodata), np.asarray(gz_recodata)
  g_recodata = np.abs(np.sqrt((gx_recodata**2)+(gy_recodata**2)+(gz_recodata**2)))
  return g_recodata

def getVoxelsCut(data, geotree, cut):
  """returns voxel centers and amplitude of voxels whose amplitude is above a given threshold"""
  xform = geotree.gxform()[0:3,0:3]
  scale = geotree.voxel_size
  shape = data.shape
  data = data.flatten(order='C')
  condition = np.ones(shape=data.shape, dtype=bool)
  condition = np.logical_and(condition, data >= cut)
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
  amps = data[(data>=cut).astype(bool)]
  return xyz, amps

"""creo array con i centri e le ampiezze di tutti gli eventi selezionati"""
def AllCentersAllEvents(events, fname1, fname2, geotree, applyGradient):
  centers_all_ev = []
  amps_all_ev = []
  recodata_all_ev = []
  for ev in events:
    data = load_pickle(fname1, ev)
    if all_parameters["it"] not in data.keys():
      data = load_pickle(fname2, ev)
    if applyGradient:
       recodata = data[all_parameters["g_it"]]
       data_masked = recodata[5:-5,5:-5,5:-5]
       gradient_recodata = gradient(data_masked)
       g_cut = all_parameters["g_cut_perc"]*np.max(gradient_recodata)
       centers, amps = getVoxelsCut(gradient_recodata, geotree, g_cut)
    else:
       recodata = data[all_parameters["it"]]
       data_masked = recodata[5:-5,5:-5,5:-5]
       cut = all_parameters["cuts"][0]
       centers, amps = getVoxelsCut(data_masked, geom.fiducial, cut)
    centers = np.transpose(centers)
    recodata_all_ev.append(recodata)
    centers_all_ev.append(centers)
    amps_all_ev.append(amps)
  recodata_all_ev = np.asarray(recodata_all_ev)
  return centers_all_ev, amps_all_ev, recodata_all_ev

def Clustering(centers, amps):
  db_algo = DBSCAN(eps = all_parameters["epsilon"], min_samples = all_parameters["min_points"])
  y_pred = db_algo.fit_predict(centers)
  labels = np.unique(y_pred)

  all_clusters_in_ev = []
  for i in labels:
      if i != -1:#because I want to exclude -1 that is the noise label
        cluster_centers = []
        cluster_amps = []
        for j in range(len(y_pred)):
          if y_pred[j] == i:
            cluster_centers.append(centers[j])
            cluster_amps.append(amps[j])
        cluster_centers = np.asarray(cluster_centers)
        cluster_amps = np.asarray(cluster_amps)
        cluster = clusterclass.VoxelCluster(cluster_centers, cluster_amps)
        all_clusters_in_ev.append(cluster) 
  return all_clusters_in_ev, y_pred

def ExtendedLPC(all_clusters_in_ev, geotree):
  new_all_curves_in_ev = []
  for cluster in all_clusters_in_ev:
    cluster.ExtendLPC(cluster.LPCs[0], geotree)
    curve = cluster.LPCs
    print("len: ", len(curve))
    curve = np.concatenate(curve)
    print("len: ", len(curve))
    new_all_curves_in_ev.append(curve)
  return new_all_curves_in_ev

def LPCDistances(all_curves):
  all_clusters_all_distances = []#questo è una lista di liste che ha tutte le distanze interne a ogni cluster di lpc
  for i in range(len(all_curves)):#loop sui cluster di lpc
    all_distances = []#inizializzo qua perchè voglio le distanze tra punti per un singolo cluster
    for j in range(len(all_curves[i])):#loop sui singoli punti di un cluster di lpc
      if j < len(all_curves[i])-1:
        distance = all_curves[i][j+1] - all_curves[i][j]
        all_distances.append(distance)
    all_clusters_all_distances.append(all_distances)
  return all_clusters_all_distances

def Angles(curve, all_LPCindex_points):
  all_clusters_all_distances = LPCDistances(all_curves)
  all_scalar_products = []
  all_LPC_vertex_nrpoints = []
  for i in range(len(all_clusters_all_distances)):#loop sui cluster di distanze (sono i cluster di lpc)
    scalar_products = []
    for j in range(len(all_clusters_all_distances[i])):#loop sulle singole distanze di un singolo cluster
      if j < (len(all_clusters_all_distances[i])-1):
        scalar_prod = np.dot(all_clusters_all_distances[i][j], all_clusters_all_distances[i][j+1])
        norm_product = np.linalg.norm(all_clusters_all_distances[i][j])*np.linalg.norm(all_clusters_all_distances[i][j+1])
        quantity_to_plot = norm_product*(1 - np.abs(scalar_prod/norm_product))
      scalar_products.append(quantity_to_plot)
      lpc_point_angles = {k:v for (k,v) in zip(range(1,len(all_LPCindex_points[i])),scalar_products)}#faccio partire da 1 il conto perchè voglio che l'indice che conta gli angoli parta dal secondo punto lpc
    LPC_vertex_nrpoint = int([k for k, v in lpc_point_angles.items() if v == np.max(scalar_products)][0])
    all_LPC_vertex_nrpoints.append(LPC_vertex_nrpoint)
    all_scalar_products.append(scalar_products)
  return all_scalar_products, all_LPC_vertex_nrpoints

if __name__ == '__main__':
  eventNumbers = EventNumberList("./data/initial-data/EDepInGrain_1.txt")
  # ev selected: 229,265,440,1173,1970,3344,3453,3701,4300
  selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]
  # selectedEvents = [0,1,2,4,5,6,7,8,9]
  # selectedEvents = [eventNumbers[4],eventNumbers[7],eventNumbers[16],eventNumbers[18],eventNumbers[19]]

  defs = {}
  defs['voxel_size'] = 12
  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/initial-data/reco_data/3dreco-0-10.pkl" #reconstruction data file
  fpkl2 = "./data/initial-data/reco_data/3dreco-10-30.pkl"
  fpkl3 = "./data/other_data/3dreco.pkl"
  fpkl4 = "./data/data_1-12/3dreco_ccqe_mup.pkl"
  geom = load_geometry(geometryPath, defs)

  # g_centers_all_ev, g_amps_all_ev, g_recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, applyGradient=True)
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)

  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    new_all_curves_in_ev = ExtendedLPC(all_clusters_in_ev, geom.fiducial)
    #print("nuovi lpc: ", new_all_curves_in_ev)

    print("evento: ", selectedEvents[i])

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    scalesz = np.max(recodata_all_ev[i].shape) * 12 / 1.6
    ax.set_xlim([-scalesz, scalesz])
    ax.set_ylim([-scalesz, scalesz])
    ax.set_zlim([-scalesz, scalesz])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter3D(centers_all_ev[i][:, 0], centers_all_ev[i][:,1], centers_all_ev[i][:,2], c = y_pred, cmap = 'cividis',s=15)
    # for c in range(len(new_all_curves_in_ev)):
    #   ax.scatter3D(new_all_curves_in_ev[c][:, 0], new_all_curves_in_ev[c][:,1], new_all_curves_in_ev[c][:,2], color = 'red')#allLPCpoints
    for curve in new_all_curves_in_ev:
      #lpc_points = np.asarray(curve)
      # lpc_points = np.squeeze(curve)
      ax.scatter3D(curve[:,0], curve[:,1], curve[:,2], color = 'red')#allLPCpoints
      #ax.scatter3D(externalPoints[:,0],externalPoints[:,1],externalPoints[:,2], color = 'darkorange')#remainingcenters
 
  #   fig2 = plt.figure()
  #   for c in range(len(all_sorted_curves_in_ev)):
  #     plt.xlabel("lpc point number")
  #     plt.ylabel("|a||b|(1-|cos$\Phi$|)")
  #     plt.scatter(range(1,len(all_LPCindex_points[c])),all_scalar_products[c])
    
    plt.title(selectedEvents[i])
    plt.legend()
    plt.show()

  """only amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)

  """gradient amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=True)