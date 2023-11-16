import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
import kneed as kn
import sys

from lpcclass import LPCSolver
from recodisplay import load_pickle, parseArgs
from geom_import import load_geometry
from recoprocessing import compute_pca

def SetParameters():
   epsilon = 36
   min_points = 6
   bandwidth = 40
   stepsize = 40

   cuts = [100,500]
   g_cut_perc = 0.05
   it = 500
   g_it = 200
   return epsilon, min_points, bandwidth, stepsize, it, g_it, cuts, g_cut_perc

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
def AllCentersAllEvents(events, fname1, fname2, applyGradient):
  _, _, _, _, it, g_it, cuts, g_cut_perc = SetParameters()
  centers_all_ev = []
  amps_all_ev = []
  recodata_all_ev = []
  for ev in events:
    data = load_pickle(fname1, ev)
    if it not in data.keys():
      data = load_pickle(fname2, ev)
    if applyGradient:
       recodata = data[g_it]
       data_masked = recodata[5:-5,5:-5,5:-5]
       gradient_recodata = gradient(data_masked)
       g_cut = g_cut_perc*np.max(gradient_recodata)
       centers, amps = getVoxelsCut(gradient_recodata, geom.fiducial, g_cut)
    else:
       recodata = data[it]
       data_masked = recodata[5:-5,5:-5,5:-5]
       cut = cuts[0]
       centers, amps = getVoxelsCut(data_masked, geom.fiducial, cut)
    centers = np.transpose(centers)
    recodata_all_ev.append(recodata)
    centers_all_ev.append(centers)
    amps_all_ev.append(amps)
  recodata_all_ev = np.asarray(recodata_all_ev)
  return centers_all_ev, amps_all_ev, recodata_all_ev

def Clustering(centers, amps):
    epsilon, min_points, _, _, _, _, _, _ = SetParameters()
    # Calculate NN
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    neighbors = nearest_neighbors.fit(centers)
    distances, indices = neighbors.kneighbors(centers)
    distances = np.sort(distances, axis=0)
    # Get distances
    distances = distances[:,1]
    i = np.arange(len(distances))
    # sns.lineplot(x = i, y = distances)
    # plt.xlabel("Points")
    # plt.ylabel("Distance")
    kneedle = kn.KneeLocator(x = range(1, len(distances)+1), y = distances, S = 1.0, 
                      curve = "concave", direction = "increasing", online=True)
    # get the estimate of knee point
    # epsilon = kneedle.knee_y
    # print("eps: ", epsilon)
    # kneedle.plot_knee()
    #plt.show()

    db_algo = DBSCAN(eps = epsilon, min_samples = min_points)
    y_pred = db_algo.fit_predict(centers)
    labels = np.unique(y_pred)

    all_clusters_in_ev = []
    all_amps_in_ev = []
    for i in labels: 
       cluster = []
       amp = []
       for j in range(len(y_pred)):
          if y_pred[j] == i:
             cluster.append(centers[j])
             amp.append(amps[j])
       cluster = np.asarray(cluster)
       all_clusters_in_ev.append(cluster) 
       all_amps_in_ev.append(amp)
    return all_clusters_in_ev, all_amps_in_ev, y_pred, labels

def LPC(centers, amps, startPoint):
    _,_,h,t,_,_,_,_ = SetParameters()
    """compute LPC"""
    lpc = LPCSolver()
    lpc.setPoints(centers, amps)
    lpc.setBandwidth(h) #find right parameter
    lpc.setStepSize(t) #find right parameter  
    lpc.solve(startPoint)      
    #lpc.plot()
    curve = lpc.lpcPoints
    inversionPoint = lpc.inversionPoint
    inversionPoint = np.asarray(inversionPoint)
    endPoint = curve[len(curve)-1]
    #initialPoint = np.average(centers, axis = 0, weights = amps)
    #initialPoint = lpc.setMaxAmpPoint(centers,amps)
    #plt.show()  
    return curve, inversionPoint, endPoint

def ClustersLPC(clusters, amps, labels, startPoint):
    all_curves = []
    all_inv_points = []
    all_end_points = []
    #all_initial_points = []
    for i in range(len(clusters)):
      if labels[i] != -1:
        curve, inversionPoint, endPoint = LPC(clusters[i], amps[i], startPoint[i])
        curve = np.asarray(curve)
        all_curves.append(curve)
        all_inv_points.append(inversionPoint)
        all_end_points.append(endPoint)
        #all_initial_points.append(initialPoint)
    all_inv_points = np.asarray(all_inv_points)
    all_end_points = np.asarray(all_end_points)
    #all_initial_points = np.asarray(all_initial_points)
    return all_curves, all_inv_points, all_end_points#, all_initial_points

# def NewLPC(clusters, amps, labels):
  # all_curves, all_inv_points, all_end_points = ClustersLPC(clusters, amps, labels)
  # for i in range(len(all_curves)): #IMPORTANTE!! nr di clusters è diverso dal numero di curves, dal momento nei clusters è contato anche il cluster del noise mentre nelle curves no
  #   original_stdout = sys.stdout
  #   with open('./points.txt', 'a') as f:
  #       sys.stdout = f
  #       print('cluster nr: ', i+1)
  #       print('inversion point: ', all_inv_points[i])
  #       print('end point: ', all_end_points[i])
  #       sys.stdout = original_stdout
  #   n = 0
  #   print('cluster nr: ', i+1)
  #   for points in clusters[i]:
  #      if all_inv_points[i][0] <= points[0] <= all_end_points[i][0] and all_inv_points[i][1] <= points[1] <= all_end_points[i][1] and all_inv_points[i][2] <= points[2] <= all_end_points[i][2]:
  #         n += 1
  #         print(n)

def NewClustersLPC(clusters, amps, startPoint):
  all_curves_new = []
  #all_initial_points = []
  for i in range(len(clusters)-1):
      curve, _, _ = LPC(clusters[i], amps[i], startPoint[i])
      curve = np.asarray(curve)
      all_curves_new.append(curve)
  return all_curves_new

def PlotClusters(events, fname1, fname2, applyGradient):
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(events, fname1, fname2, applyGradient)
  for i in range(len(events)):
    all_clusters_in_ev, all_amps_in_ev, y_pred, labels= Clustering(centers_all_ev[i], amps_all_ev[i])
    all_curves_in_ev, all_inv_points, all_end_points = ClustersLPC(all_clusters_in_ev,all_amps_in_ev,labels)
    #original_stdout = sys.stdout
    # with open('./points.txt', 'a') as f:
    #     sys.stdout = f
    #     print('event: ', events[i])
    #     sys.stdout = original_stdout
    # print('event: ', events[i])
    # NewLPC(all_clusters_in_ev, all_amps_in_ev, labels)
  
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
    for c in range(len(all_curves_in_ev)):
      #ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')
      ax.scatter3D(all_curves_in_ev[c][0,0], all_curves_in_ev[c][0,1], all_curves_in_ev[c][0,2], color = 'darkorange')
      ax.scatter3D(all_inv_points[c][0],all_inv_points[c][1],all_inv_points[c][2], color = 'cyan')
      ax.scatter3D(all_end_points[c][0],all_end_points[c][1],all_end_points[c][2], color = 'black')
    plt.title(events[i])
    plt.legend()
  plt.show()
  


if __name__ == '__main__':
  ###example structure, fill the gaps
  
  eventNumbers = EventNumberList("./data/initial-data/EDepInGrain_1.txt")
  # ev selected: 229,265,440,1173,1970,3344,3453,3701,4300
  selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]

  defs = {}
  defs['voxel_size'] = 12
  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/initial-data/reco_data/3dreco-0-10.pkl" #reconstruction data file
  fpkl2 = "./data/initial-data/reco_data/3dreco-10-30.pkl"
  geom = load_geometry(geometryPath, defs)
  
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, applyGradient=False)
  #print(list(centers_all_ev[0]))

  data = load_pickle(fpkl1, 229)
  recodata = data[500]
  data_masked = recodata[5:-5,5:-5,5:-5]
  pca_results = compute_pca(data_masked, geom.fiducial, 100, plot=True)
  print(pca_results)
 

  # # g_centers_all_ev, g_amps_all_ev, g_recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, applyGradient=True)
  
  for i in range(len(selectedEvents)):
     all_clusters_in_ev, all_amps_in_ev, y_pred, labels = Clustering(centers_all_ev[i], amps_all_ev[i])
     all_initial_points = []
     for c in range(len(all_clusters_in_ev)):
       initialPoint = np.average(all_clusters_in_ev[c], axis = 0, weights = all_amps_in_ev[c])
       all_initial_points.append(initialPoint)
     all_curves_in_ev, all_inv_points, all_end_points = ClustersLPC(all_clusters_in_ev, all_amps_in_ev, labels, all_initial_points)
    
     all_curves_in_ev2 = NewClustersLPC(all_clusters_in_ev, all_amps_in_ev, all_inv_points)

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
     for c in range(len(all_curves_in_ev)):
       ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')
       ax.scatter3D(all_curves_in_ev2[c][:, 0], all_curves_in_ev2[c][:,1], all_curves_in_ev2[c][:,2], color = 'darkgoldenrod')
       # ax.scatter3D(all_curves_in_ev[c][0,0], all_curves_in_ev[c][0,1], all_curves_in_ev[c][0,2], color = 'darkorange')
       # ax.scatter3D(all_inv_points[c][0],all_inv_points[c][1],all_inv_points[c][2], color = 'cyan')
       # ax.scatter3D(all_curves_in_ev[c][len(all_curves_in_ev[c])-1,0], all_curves_in_ev[c][len(all_curves_in_ev[c])-1,1], all_curves_in_ev[c][len(all_curves_in_ev[c])-1,2], color = 'black')
     plt.title(selectedEvents[i])
     plt.legend()
     plt.show()
  




  """only amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, applyGradient=False)

  """gradient amps"""
  #PlotClusters(selectedEvents, fpkl1, fpkl2, applyGradient=True)
