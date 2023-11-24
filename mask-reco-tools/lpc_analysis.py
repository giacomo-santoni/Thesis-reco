import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
import kneed as kn
import sys
import copy

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
def AllCentersAllEvents(events, fname1, fname2, geotree, applyGradient):
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
       centers, amps = getVoxelsCut(gradient_recodata, geotree, g_cut)
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
      amp = np.asarray(amp)
      all_clusters_in_ev.append(cluster) 
      all_amps_in_ev.append(amp)
  return all_clusters_in_ev, all_amps_in_ev, y_pred, labels

def LPC(centers, amps):
  _,_,h,t,_,_,_,_ = SetParameters()
  """compute LPC"""
  lpc = LPCSolver()
  lpc.setPoints(centers, amps)
  lpc.setBandwidth(h) #find right parameter
  lpc.setStepSize(t) #find right parameter  
  lpc.solve()      
  #lpc.plot()
  curve = lpc.lpcPoints
  #plt.show()  
  return curve

def ClustersLPC(clusters, amps, labels):#, startPoints):
  all_curves = []
  for i in range(len(clusters)):
    if labels[i] != -1:
      curve = LPC(clusters[i], amps[i])#, startPoints[i])
      curve = np.asarray(curve)
      all_curves.append(curve)
  return all_curves

def is_point_in_array(point, array):
    return any(np.all(point == i) for i in array)

def ExcludingPoints(all_clusters, all_amps, labels, geotree):
  all_curves = ClustersLPC(all_clusters, all_amps, labels)
  R = 5.5*(geotree.voxel_size)
  all_inner_points = []
  all_inner_amps = []
  all_point_masks = []
  for i in range(len(all_curves)):
    innerPoints = []
    innerAmps = []
    print(all_clusters[i+1].shape)
    point_mask = np.zeros((len(all_clusters[i+1])))
    for lpc in all_curves[i]:
      for j,p in enumerate(all_clusters[i+1]):#faccio +1 perchè in all_clusters il primo elemento è sempre il cluster che contiene il noise. Dato che sui punti del noise non facciamo la lpc, non ha senso includerli in questa operazione. Non mi da errore perchè tanto la i sta ciclando su all_curves che è 1 dimensione più corto.
        diff = p - lpc
        distance = np.abs(np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2))
        if distance < R:
          if not is_point_in_array(p,innerPoints):
            innerPoints.append(p)
            innerAmps.append(all_amps[i+1][j])
          point_mask[j] = 1
    innerPoints = np.asarray(innerPoints)
    print(innerPoints.shape)
    innerAmps = np.asarray(innerAmps)
    all_inner_points.append(innerPoints)
    all_inner_amps.append(innerAmps)
    all_point_masks.append(point_mask)
  return all_inner_points, all_inner_amps, all_point_masks

def RemainingPoints(all_clusters, all_amps, all_inner_points, all_inner_amps, all_point_masks):
  #print("all clusters: ", len(all_clusters))
  remaining_clusters = copy.deepcopy(all_inner_points)
  remaining_amps = copy.deepcopy(all_inner_amps)
  #per i centers non posso usare la funzione np.isin perchè considera uguali due punti che hanno coordinate in ordine diverso. per questo motivo nella funzione precedente ho riempito una matrice lunga quanto i punti del cluster che è 1 in corrispondenza degli innerPoints e 0 negli altri punti. Poi li trasformo in booleani e li inverto, in modo che poi la applico come maschera sulla matrice totale di punti e ci siano valori False in corrispondenza degli innerPoints e True negli altri.
  for cluster_index in range(len(all_inner_points)):
    #print("inner pt: ", len(all_inner_points))
    all_point_masks[cluster_index] = np.invert(all_point_masks[cluster_index].astype(bool))
    remaining_clusters[cluster_index] = all_clusters[cluster_index+1][all_point_masks[cluster_index]]
    remaining_clusters[cluster_index] = np.asarray(remaining_clusters[cluster_index])
    print("rem clusters points: ", len(remaining_clusters[cluster_index]))

  for cluster_index, inner_amps in enumerate(all_inner_amps):
    amp_mask = np.isin(all_amps[cluster_index+1], inner_amps)#creo una matrice uguale a all_cluster[i] (n_points, 3) di True e False, dove True sono gli elementi che sono presenti in entrambi gli array, e False sono quelli che rimangono. Noi però stiamo lavorando con punti che hanno 3 colonne, per cui dobbiamo accertarci che tutte le colonne di un punto siano True. 
    remaining_amps[cluster_index] = all_amps[cluster_index+1][~amp_mask]#dato che a noi interessano i punti NON presenti in inner_points, invertiamo la matrice e così i True diventano i punti che non sono presenti in inner_points
    remaining_amps[cluster_index] = np.asarray(remaining_amps[cluster_index])
    #print("rem amps: ", remaining_amps[cluster_index].shape)
  return remaining_clusters, remaining_amps

def NewClustersLPC(remaining_clusters, remaining_amps):#, all_inner_points, all_inner_amps):
  #remaining clusters ha sempre la stessa lunghezza di all_inner_points anche se dentro non c'è nessuno punto, vedi evento 1173
  #print("nr rem clusters: ", len(remaining_clusters))
  all_new_curves = []
  for i in range(len(remaining_clusters)):
    if len(remaining_clusters[i]) != 0:
      new_curve = LPC(remaining_clusters[i], remaining_amps[i])
      new_curve = np.asarray(new_curve)
    else: new_curve = np.zeros((2,3))
    all_new_curves.append(new_curve)
  return all_new_curves

# def PlotClusters(events, fname1, fname2, geotree, applyGradient):
  # centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(events, fname1, fname2, geotree, applyGradient)
  # for i in range(len(events)):
  #   all_clusters_in_ev, all_amps_in_ev, y_pred, labels= Clustering(centers_all_ev[i], amps_all_ev[i])
  #   all_curves_in_ev, = ClustersLPC(all_clusters_in_ev,all_amps_in_ev,labels)
  #   #original_stdout = sys.stdout
  #   # with open('./points.txt', 'a') as f:
  #   #     sys.stdout = f
  #   #     print('event: ', events[i])
  #   #     sys.stdout = original_stdout
  #   # print('event: ', events[i])
  
  #   fig1 = plt.figure()
  #   ax = fig1.add_subplot(projection='3d')
  #   scalesz = np.max(recodata_all_ev[i].shape) * 12 / 1.6
  #   ax.set_xlim([-scalesz, scalesz])
  #   ax.set_ylim([-scalesz, scalesz])
  #   ax.set_zlim([-scalesz, scalesz])
  #   ax.set_xlabel('x')
  #   ax.set_ylabel('y')
  #   ax.set_zlabel('z')
  #   ax.scatter3D(centers_all_ev[i][:, 0], centers_all_ev[i][:,1], centers_all_ev[i][:,2], c = y_pred, cmap = 'cividis',s=15)
  #   for c in range(len(all_curves_in_ev)):
  #     ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')
  #     ax.scatter3D(all_curves_in_ev[c][0,0], all_curves_in_ev[c][0,1], all_curves_in_ev[c][0,2], color = 'darkorange')
  #     # ax.scatter3D(all_inv_points[c][0],all_inv_points[c][1],all_inv_points[c][2], color = 'cyan')
  #     # ax.scatter3D(all_end_points[c][0],all_end_points[c][1],all_end_points[c][2], color = 'black')
  #   plt.title(events[i])
  #   plt.legend()
  # plt.show()
  

if __name__ == '__main__':
  ###example structure, fill the gaps
  
  eventNumbers = EventNumberList("./data/initial-data/EDepInGrain_1.txt")
  # ev selected: 229,265,440,1173,1970,3344,3453,3701,4300
  #selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]
  selectedEvents = [0,1,2,4,5,6,7,8,9]

  defs = {}
  defs['voxel_size'] = 12
  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/initial-data/reco_data/3dreco-0-10.pkl" #reconstruction data file
  fpkl2 = "./data/initial-data/reco_data/3dreco-10-30.pkl"
  fpkl3 = "./data/other_data/3dreco.pkl"
  geom = load_geometry(geometryPath, defs)

  # data = load_pickle(fpkl1, 229)
  # recodata = data[500]
  # data_masked = recodata[5:-5,5:-5,5:-5]
  # pca_results = compute_pca(data_masked, geom.fiducial, 100, plot=True)
  # print(pca_results)

  # # g_centers_all_ev, g_amps_all_ev, g_recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, applyGradient=True)
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl3, fpkl2, geom.fiducial, applyGradient=False)


  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, all_amps_in_ev, y_pred, labels = Clustering(centers_all_ev[i], amps_all_ev[i])

    # all_initial_points = []
    # for c in range(len(all_clusters_in_ev)):
    #   initialPoint = np.average(all_clusters_in_ev[c], axis = 0, weights = all_amps_in_ev[c])
    #   all_initial_points.append(initialPoint)


    all_inner_points, all_inner_amps, all_point_masks = ExcludingPoints(all_clusters_in_ev, all_amps_in_ev,labels, geom.fiducial)

    all_curves_in_ev = ClustersLPC(all_clusters_in_ev, all_amps_in_ev, labels)

    all_remaining_clusters, all_remaining_amps = RemainingPoints(all_clusters_in_ev, all_amps_in_ev, all_inner_points, all_inner_amps, all_point_masks)

    new_all_curves_in_ev = NewClustersLPC(all_remaining_clusters, all_remaining_amps)
    
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
    for c in range(len(all_curves_in_ev)):
      ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')
      ax.scatter3D(new_all_curves_in_ev[c][:, 0], new_all_curves_in_ev[c][:,1], new_all_curves_in_ev[c][:,2], color = 'blue')
      # ax.scatter3D(all_clusters_in_ev[0][:,0], all_clusters_in_ev[0][:,1], all_clusters_in_ev[0][:,2], color = 'yellow')
      ax.scatter3D(all_remaining_clusters[c][:,0],all_remaining_clusters[c][:,1],all_remaining_clusters[c][:,2], color = 'darkorange')
      ax.scatter3D(all_inner_points[c][:, 0], all_inner_points[c][:, 1], all_inner_points[c][:, 2], color = 'forestgreen')
      #ax.scatter3D(all_curves_in_ev[0][0,0], all_curves_in_ev[0][0,1], all_curves_in_ev[0][0,2], color = 'cyan')
    # ax.scatter3D(all_inv_points[c][0],all_inv_points[c][1],all_inv_points[c][2], color = 'cyan')
    # ax.scatter3D(all_curves_in_ev[c][len(all_curves_in_ev[c])-1,0], all_curves_in_ev[c][len(all_curves_in_ev[c])-1,1], all_curves_in_ev[c][len(all_curves_in_ev[c])-1,2], color = 'black')
    plt.title(selectedEvents[i])
    plt.legend()
  plt.show()




  """only amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)

  """gradient amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=True)