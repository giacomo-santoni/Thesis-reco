import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

import clusterclass
from recodisplay import load_pickle
from geom_import import load_geometry

defs = {}
defs['voxel_size'] = 12
#bandwidth = 30, stepsize = 55
all_parameters = {"epsilon" : 3*defs['voxel_size'], "min_points" : 6, "bandwidth" : 2.5*defs['voxel_size'], "stepsize" : 4.6*defs['voxel_size'], "cuts" : [100,500], "g_cut_perc" : 0.05, "it" : 500, "g_it" : 200}

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

def Collinearity(dist1, dist2):
  scalar_product = np.dot(dist1, dist2)
  norm_product = np.linalg.norm(dist1)*np.linalg.norm(dist2)
  cosine = scalar_product/norm_product
  return cosine

def ComputeTotalDistance(LPCclusters):
  distance_with_cluster = []
  for curve in LPCclusters:
    curve = np.asarray(curve)
    tot_dist = curve[-1] - curve[0]#distance vector
    distance_with_cluster.append((tot_dist,curve))
  #   tot_distances.append(tot_dist)
  #   print("dist: ",tot_distances)
  # distance_with_cluster = list(zip(tot_distances, LPCclusters))
  # print("dict: ", distance_with_cluster)
  return distance_with_cluster

def FindCollinearCurves(clusters):
  distances_in_ev = []
  all_distances_tuple = []
  for c in clusters:
    c.BreakLPCs()
    if c.break_point != 0:
      distances_clust = ComputeTotalDistance(c.broken_lpccurve)
    if len(c.LPCs)>1:
      distances_clust = ComputeTotalDistance(c.LPCs[1:])
    elif c.break_point == 0:
      distances_clust = ComputeTotalDistance(c.LPCs)
    distances_in_ev.append(distances_clust)
  
  for tuple in distances_in_ev:
    for subtuple in tuple:
      all_distances_tuple.append(subtuple)
  #print("len: ", all_distances_tuple[0])
  
  all_collinears = []
  for dist1,lpc1 in all_distances_tuple:
    collinear_lpc = []
    print(i)
    dist1_index = next(i for i, t in enumerate(all_distances_tuple) if np.all(t[0]) == np.all(dist1))
    collinear_lpc.append(lpc1)
    for dist2,lpc2 in all_distances_tuple[dist1_index:]:
      if np.abs(Collinearity(dist1,dist2)) > 0.906:#angle is less than 25°
        collinear_lpc.append(lpc2)
    all_collinears.append(collinear_lpc)
  print("all: ", all_collinears)
  return all_collinears

if __name__ == '__main__':
  #eventNumbers = EventNumberList("./data/initial-data/EDepInGrain_1.txt")
  eventNumbers = EventNumberList("./data/data_1-12/idlist_ccqe_mup.txt")
  # ev selected: 229,265,440,1173,1970,3344,3453,3701,4300
  #selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]
  #selectedEvents = [0,1,2,4,5,6,7,8,9]
  selectedEvents = [eventNumbers[4],eventNumbers[7],eventNumbers[16],eventNumbers[18],eventNumbers[19]]

  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/initial-data/reco_data/3dreco-0-10.pkl" #reconstruction data file
  fpkl2 = "./data/initial-data/reco_data/3dreco-10-30.pkl"
  fpkl3 = "./data/other_data/3dreco.pkl"
  fpkl4 = "./data/data_1-12/3dreco_ccqe_mup.pkl"
  geom = load_geometry(geometryPath, defs)

  # g_centers_all_ev, g_amps_all_ev, g_recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, applyGradient=True)
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl4, fpkl2, geom.fiducial, applyGradient=False)

  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    all_collinears = FindCollinearCurves(all_clusters_in_ev)
    #print("coll: ", all_collinears)

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
    for cluster in all_clusters_in_ev:
      single_curve = np.asarray(cluster.LPCs[0])
      cluster.FindBreakPoint()
      ax.scatter3D(single_curve[:, 0], single_curve[:,1], single_curve[:,2], color = 'red')#allLPCpoints
      if cluster.break_point != 0:
        cluster.BreakLPCs()
        broken_curve = np.asarray(cluster.broken_lpccurve[0])
        broken_curve2 = np.asarray(cluster.broken_lpccurve[1])
        ax.scatter3D(broken_curve[:,0], broken_curve[:,1], broken_curve[:,2], color = 'green')
        ax.scatter3D(broken_curve2[:,0], broken_curve2[:,1], broken_curve2[:,2], color = 'purple')
        ax.scatter3D(single_curve[cluster.break_point][0], single_curve[cluster.break_point][1], single_curve[cluster.break_point][2], color = 'blue', marker='D')
    
    for coll_lpcs in all_collinears:
      ax.scatter3D(coll_lpcs[0][:,0], coll_lpcs[0][:,1], coll_lpcs[0][:,2], color = 'green')
      # ax.scatter3D(coll_lpcs[1][:,0], coll_lpcs[1][:,1], coll_lpcs[1][:,2], color = 'blue')
      # ax.scatter3D(coll_lpcs[2][:,0], coll_lpcs[2][:,1], coll_lpcs[2][:,2], color = 'black')
    
    plt.title(selectedEvents[i])
    plt.legend()
    plt.show()

  """only amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)

  """gradient amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=True)




  # def FindCollinearCurves(clusters):
  # #broken_curves_distances = self.BreakLPCs()

  # # new_curves_distances = []
  # # for curve in self.LPCs[1:]:#since I want to consider the lpc clusters obtained in ExtendLPC()
  # #     distance = curve[-1] - curve[0]
  # #     new_curves_distances.append(distance)

  # # collinearities = []
  # # for i, b_distance in enumerate(broken_curves_distances):
  # #     for j, n_distance in enumerate(new_curves_distances):
  # #         # scalar_product = np.dot(b_distance, n_distance)
  # #         # norm_product = np.linalg.norm(b_distance)*np.linalg.norm(n_distance)
  # #         cosine = Collinearity(b_distance, n_distance)
  # #         collinearity = np.abs(cosine)#collinearity is the absolute value of cosine
  # #         collinearities.append((collinearity, (self.broken_lpccurve[i], self.LPCs[1:][j])))
  # # sorted_collinearities = {k:v for (k,v) in collinearities}
  # # print("nr coll: ", len(sorted_collinearities))
  # # max_collinearities = sorted(list(sorted_collinearities.keys()), reverse=True)[:len(self.LPCs[1:])]
  # # print("max collinearities: ", max_collinearities)
  # # for value in max_collinearities:
  # #     self.collinear_clusters = sorted_collinearities[value]

  # all_clusters_distances = []
  # for c in clusters:
  #   print("lunghezza: ", len(c.LPCs[0]))
  #   print(c.LPCs[1])
  #   distance_vector = c.LPCs[0][-1] - c.LPCs[0][0]
  #   all_clusters_distances.append(distance_vector)
  
  # collinearities = []
  # for i, distance in enumerate(all_clusters_distances[1:]):
  #     cosine = Collinearity(distance, all_clusters_distances[0])
  #     collinearity = np.abs(cosine)
  #     collinearities.append((collinearity, (clusters[0].LPCs[0], clusters[i+1].LPCs[0])))
  # sorted_collinearities = {k:v for (k,v) in collinearities}
  # print("nr coll: ", len(sorted_collinearities))
  # max_collinearities = np.max(sorted(list(sorted_collinearities.keys()), reverse=True))
  # print("max collinearities: ", max_collinearities)
  # #for value in max_collinearities:
  # collinear_clusters = sorted_collinearities[max_collinearities]
  # print(len(collinear_clusters))
  # return collinear_clusters