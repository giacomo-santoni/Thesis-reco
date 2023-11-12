import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
import kneed as kn

from lpcclass import LPCSolver
from recodisplay import load_pickle, parseArgs
from geom_import import load_geometry

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

def Clustering(centers, amps):
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
    #plt.show()
    kneedle = kn.KneeLocator(x = range(1, len(distances)+1), y = distances, S = 1.0, 
                      curve = "concave", direction = "increasing", online=True)
    # get the estimate of knee point
    epsilon = kneedle.knee_y
    print(epsilon)
    kneedle.plot_knee()
    #plt.show()

    dbscan_opt=DBSCAN(eps=epsilon, min_samples=6)
    y_pred = dbscan_opt.fit_predict(centers)
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
    return all_clusters_in_ev, all_amps_in_ev, y_pred

def PlotClusters(centers, amps):
  # Plotting the resulting clusters
    _,_,y_pred = Clustering(centers, amps)
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.scatter3D(centers[:, 0], centers[:,1], centers[:,2], c = y_pred, cmap = 'cividis',s=15)
    #ax.scatter3D(curve[:, 0], curve[:,1], curve[:,2], color = 'red')
    plt.title("event: ", )
    plt.legend()
    plt.show()

def LPC(centers, amps):
    """compute LPC"""
    lpc = LPCSolver()
    lpc.setPoints(centers, amps)
    lpc.setBandwidth(30) #find right parameter
    lpc.setStepSize(30) #find right parameter  
    lpc.solve()      
    #lpc.plot()
    curve = lpc.lpcPoints
    #plt.show()  
    return curve

def ClustersLPC(clusters, amps):
    all_curves = []
    for i in range(len(clusters)):
      if i != 0:
        curve = LPC(clusters[i], amps[i])
        curve = np.asarray(curve)
        all_curves.append(curve)
    return all_curves
    
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
   






if __name__ == '__main__':
  ###example structure, fill the gaps
  
  eventNumbers = EventNumberList("./data/initial-data/EDepInGrain_1.txt")
  #ev selected: 229,265,440,1173,1970,3344,3453,3701,4300
  selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]

  defs = {}
  defs['voxel_size'] = 12
  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/initial-data/reco_data/3dreco-0-10.pkl" #reconstruction data file
  fpkl2 = "./data/initial-data/reco_data/3dreco-10-30.pkl"
  #eventNumber = selectedEvents[0]
  geom = load_geometry(geometryPath, defs)
  """normal pars"""
  it = 500
  cuts = [100, 500] #cuts on voxel amplitude: higher and lower
  cut = cuts[1] 


  centers_all_ev = []
  amps_all_ev = []
  for ev in selectedEvents:
    data = load_pickle(fpkl1, ev)
    if it not in data.keys():
      data = load_pickle(fpkl2, ev)
    recodata = data[it]
    data_masked = recodata[5:-5,5:-5,5:-5]
    centers, amps = getVoxelsCut(data_masked, geom.fiducial, cut)
    centers = np.transpose(centers)
    centers_all_ev.append(centers)
    amps_all_ev.append(amps)

  # """gradient pars"""
  # g_it = 240
  # g_cut = 50 #cut on voxel amplitude
  # g_recodata = data[g_it]
  # g_data_masked = g_recodata[5:-5,5:-5,5:-5]
  # gradient_recodata = gradient(g_data_masked)
  # g_centers, g_amps = getVoxelsCut(gradient_recodata, geom.fiducial, g_cut)
  # g_centers = np.transpose(g_centers)

  for i in range(len(centers_all_ev)):
    all_clusters_in_ev, all_amps_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])
    # print(len(all_amps_in_ev))
    all_curves_in_ev = ClustersLPC(all_clusters_in_ev,all_amps_in_ev)
    

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.scatter3D(centers_all_ev[i][:, 0], centers_all_ev[i][:,1], centers_all_ev[i][:,2], c = y_pred, cmap = 'cividis',s=15)
    #DA SISTEMARE IL PLOT DELLE CURVE!!!!
    for c in range(len(all_curves_in_ev)):
      ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')
    plt.title(selectedEvents[i])
    plt.legend()
    plt.show()
  
  # """normal"""
  # fig1 = plt.figure()
  # ax = fig1.add_subplot(projection='3d')
  # # for i in range(len(all_curves)):
  # #   ax.scatter3D(all_curves[i][:, 0], all_curves[i][:,1], all_curves[i][:,2], color = 'red')
  # #   # ax.scatter3D(all_curves[i][0,0], all_curves[i][:,1], all_curves[i][:,2], color = 'green')
  # ax.scatter3D(centers[:, 0], centers[:,1], centers[:,2], c = amps, cmap = 'viridis', alpha = 0.3)
  # plt.legend()
  # #plt.show()

  # """gradient"""
  # fig2 = plt.figure()
  # ax = fig2.add_subplot(projection='3d')
  # # for i in range(len(all_curves)):
  # #   ax.scatter3D(all_curves[i][:, 0], all_curves[i][:,1], all_curves[i][:,2], color = 'red')
  # #   # ax.scatter3D(all_curves[i][0,0], all_curves[i][:,1], all_curves[i][:,2], color = 'green')
  # ax.scatter3D(g_centers[:, 0], g_centers[:,1], g_centers[:,2], c = g_amps, cmap = 'cividis', alpha = 0.3)
  # plt.legend()
  # #plt.show()


