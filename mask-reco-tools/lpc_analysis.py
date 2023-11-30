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
  nrLPCpoints = range(1,len(curve))#perchè nel plot voglio considerare l'angolo tra segmento precedente e successivo
  inversionPoint = lpc.inversionPoint
  endPoint = lpc.endPoint
  #plt.show()
  return curve, inversionPoint, endPoint, nrLPCpoints

def ClustersLPC(clusters, amps, labels):#, startPoints):
  all_curves = []
  all_inv_points = []
  all_end_points = []
  all_nrLPCpoints = []
  for i in range(len(clusters)):
    if labels[i] != -1:
      curve, inversionPoint, endPoint, nrLPCpoints = LPC(clusters[i], amps[i])#, startPoints[i])
      curve = np.asarray(curve)
      all_curves.append(curve)
      all_inv_points.append(inversionPoint)
      all_end_points.append(endPoint)
      all_nrLPCpoints.append(nrLPCpoints)
  return all_curves, all_inv_points, all_end_points, all_nrLPCpoints

def is_point_in_array(point, array):
    return any(np.all(point == i) for i in array)

def ExcludingPoints(all_clusters, all_amps, labels, geotree):
  all_curves, _, _, _ = ClustersLPC(all_clusters, all_amps, labels)
  R = 7*(geotree.voxel_size)
  all_inner_points = []
  all_inner_amps = []
  all_point_mask = []
  for i in range(len(all_curves)):
    innerPoints = []
    innerAmps = []
    point_mask = np.zeros((len(all_clusters[i+1])))
    for j,p in enumerate(all_clusters[i+1]):#faccio +1 perchè in all_clusters il primo elemento è sempre il cluster che contiene il noise. Dato che sui punti del noise non facciamo la lpc, non ha senso includerli in questa operazione. Non mi da errore perchè tanto la i sta ciclando su all_curves che è 1 dimensione più corto.
      all_distances = []
      for lpc in all_curves[i]:
        diff = p - lpc
        distance = np.abs(np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2))
        all_distances.append(distance)
      bool_distances = np.asarray(all_distances) < R
      if np.count_nonzero(bool_distances) >= 3:
        if not is_point_in_array(p,innerPoints):
          innerPoints.append(p)
          innerAmps.append(all_amps[i+1][j])
        point_mask[j] = 1
    innerPoints = np.asarray(innerPoints)
    innerAmps = np.asarray(innerAmps)
    all_inner_points.append(innerPoints)
    all_inner_amps.append(innerAmps)
    all_point_mask.append(point_mask)
  return all_inner_points, all_inner_amps, all_point_mask

def RemainingPoints(all_clusters, all_amps, all_inner_points, all_inner_amps, all_point_mask):
  remaining_clusters = copy.deepcopy(all_inner_points)
  remaining_amps = copy.deepcopy(all_inner_amps)
  # per i centers non posso usare la funzione np.isin perchè considera uguali due punti che hanno coordinate in ordine diverso. per questo motivo nella funzione precedente ho riempito una matrice lunga quanto i punti del cluster che è 1 in corrispondenza degli innerPoints e 0 negli altri punti. Poi li trasformo in booleani e li inverto, in modo che poi la applico come maschera sulla matrice totale di punti e ci siano valori False in corrispondenza degli innerPoints e True negli altri.
  remaining_points = []
  for cluster_index in range(len(all_inner_points)):
    all_point_mask[cluster_index] = np.invert(all_point_mask[cluster_index].astype(bool))
    remaining_clusters[cluster_index] = all_clusters[cluster_index+1][all_point_mask[cluster_index]]
    for points in remaining_clusters[cluster_index]:
      remaining_points.append(points)

  list_remaining_amps = []
  for cluster_index, inner_amps in enumerate(all_inner_amps):
    amp_mask = np.isin(all_amps[cluster_index+1], inner_amps)#creo una matrice uguale a all_cluster[i] (n_points, 3) di True e False, dove True sono gli elementi che sono presenti in entrambi gli array, e False sono quelli che rimangono. Noi però stiamo lavorando con punti che hanno 3 colonne, per cui dobbiamo accertarci che tutte le colonne di un punto siano True. 
    remaining_amps[cluster_index] = all_amps[cluster_index+1][~amp_mask]#dato che a noi interessano i punti NON presenti in inner_points, invertiamo la matrice e così i True diventano i punti che non sono presenti in inner_points
    remaining_amps[cluster_index] = np.asarray(remaining_amps[cluster_index])
    for amps in remaining_amps[cluster_index]:
      list_remaining_amps.append(amps)
  return remaining_clusters, remaining_points, remaining_amps, list_remaining_amps

def NewClustersLPC(remaining_points, list_remaining_amps, all_clusters,all_amps,labels):
  if remaining_points:
    new_remaining_clusters, new_remaining_amps, _, new_labels = Clustering(remaining_points, list_remaining_amps)
  else: 
    new_remaining_clusters = all_clusters
    new_remaining_amps = all_amps
    new_labels = labels
  
  all_new_curves = []
  new_all_inv_points = []
  new_all_end_points = []
  new_all_nrLPCpoints = []
  for i in range(len(new_remaining_clusters)):
    if new_labels[i] != -1:
      new_curve,new_invPoint,new_endPoint,new_nrLPCpoints = LPC(new_remaining_clusters[i], new_remaining_amps[i])
      new_curve = np.asarray(new_curve)
      all_new_curves.append(new_curve)
      new_all_inv_points.append(new_invPoint)
      new_all_end_points.append(new_endPoint)
      new_all_nrLPCpoints.append(new_nrLPCpoints)
  return all_new_curves, new_all_inv_points, new_all_end_points, new_all_nrLPCpoints

def LPCDistances(all_curves, all_inv_points):
  all_distances_to_end = []
  all_distances_to_inv = []
  all_distances_inv_to_end = []
  for i in range(len(all_curves)):
    distances_to_inv = []
    distances_to_end = []
    for lpc in range(len(all_curves[i])):
      if not np.array_equal(all_curves[i][lpc+1],all_inv_points[i]):
        distance = all_curves[i][lpc+1] - all_curves[i][lpc]
        distances_to_inv.append(distance)
      else: 
        distance = all_inv_points[i] - all_curves[i][lpc]
        inversion_index = lpc+1
        distances_to_inv.append(distance)
        distances_to_inv_rev = distances_to_inv[::-1]
        break
    all_distances_to_inv.append(distances_to_inv)#serve perchè posso avere più clusters di punti LPC in un evento

    for lpc in range(inversion_index+1,len(all_curves[i])):
      if lpc < len(all_curves[i])-1:
        if distances_to_end == []: #and inv_to_next > origin_to_inv:
          distance = all_curves[i][lpc] - all_curves[i][0]#basta lpc come indice perchè nel loop parto già da (inversion_index +1)
          distances_to_end.append(distance)
          distance2 = all_curves[i][lpc+1] - all_curves[i][lpc]
          distances_to_end.append(distance2)
        else: 
          distance = all_curves[i][lpc+1] - all_curves[i][lpc]
          distances_to_end.append(distance)
      
    distance_inv_to_end = np.concatenate((distances_to_inv_rev, distances_to_end))
    all_distances_to_end.append(distances_to_end)
    all_distances_inv_to_end.append(distance_inv_to_end)
  return all_distances_to_inv, all_distances_to_end, all_distances_inv_to_end

def ConnectingLPCclusters(clusters, amps, remaining_points, list_remaining_amps, labels):#?????????
  all_curves, all_inv_points, all_end_points,_ = ClustersLPC(clusters, amps, labels)
  all_new_curves, new_all_inv_points, new_all_end_points,_ = NewClustersLPC(remaining_points, list_remaining_amps, clusters, amps, labels)
  all_clusters_all_diffs = []
  for c in range(len(all_curves)):
    all_diffs = []
    for new_c in range(len(all_new_curves)):
      diff1 = new_all_end_points[new_c] - all_end_points[c]
      all_diffs.append(diff1)
      diff2 = new_all_end_points[new_c] - all_inv_points[c]
      all_diffs.append(diff2)
      diff3 = new_all_inv_points[new_c] - all_inv_points[c]
      all_diffs.append(diff3)
      diff4 = new_all_inv_points[new_c] - all_end_points[c]
      all_diffs.append(diff4)
    all_clusters_all_diffs.append(all_diffs)
  print(len(all_clusters_all_diffs))
  
  all_clusters_all_distances = []
  for all_diffs in all_clusters_all_diffs:
    all_distances = []
    for diff in all_diffs:
      distance = np.abs(np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2))
      all_distances.append(distance)
    all_distances = np.asarray(all_distances)
    all_clusters_all_distances.append(all_distances)
  print(len(all_clusters_all_distances))

  all_connections = []
  for i in range(len(all_clusters_all_distances)):
    min_distance = np.min(all_clusters_all_distances[i])
    index_matching = all_clusters_all_distances[i].index(min_distance)
    connection = all_clusters_all_diffs[i][index_matching]
    all_connections.append(connection)

  return index_matching, all_connections

def Angles(all_curves, all_inv_points):
  _, _, all_distances_inv_to_end = LPCDistances(all_curves, all_inv_points)
  all_scalar_products = []
  for i in range(len(all_distances_inv_to_end)):
    scalar_products = []
    for j in range(len(all_distances_inv_to_end[i])):
      if j < (len(all_distances_inv_to_end[i])-1):
        scalar_product = np.dot(all_distances_inv_to_end[i][j], all_distances_inv_to_end[i][j+1])#/(np.linalg.norm(v1_u)*np.linalg.norm(v2_u))
        norm_product = np.linalg.norm(all_distances_inv_to_end[i][j])*np.linalg.norm(all_distances_inv_to_end[i][j+1])
        quantity_to_plot = norm_product*(1 - np.abs(scalar_product/norm_product))
      scalar_products.append(quantity_to_plot)
    all_scalar_products.append(scalar_products)
    print("scalar prod: ",len(all_scalar_products[i]))
  return all_scalar_products

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
  selectedEvents = [eventNumbers[0],eventNumbers[1],eventNumbers[2],eventNumbers[6],eventNumbers[9],eventNumbers[20],eventNumbers[21],eventNumbers[24],eventNumbers[25]]
  #selectedEvents = [0,1,2,4,5,6,7,8,9]

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
  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)


  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, all_amps_in_ev, y_pred, labels = Clustering(centers_all_ev[i], amps_all_ev[i])

    # all_initial_points = []
    # for c in range(len(all_clusters_in_ev)):
    #   initialPoint = np.average(all_clusters_in_ev[c], axis = 0, weights = all_amps_in_ev[c])
    #   all_initial_points.append(initialPoint)


    all_inner_points, all_inner_amps, all_point_mask = ExcludingPoints(all_clusters_in_ev, all_amps_in_ev,labels, geom.fiducial)

    all_curves_in_ev, all_inv_points, all_end_points, all_nrLPCpoints = ClustersLPC(all_clusters_in_ev, all_amps_in_ev, labels)

    all_remaining_clusters, all_remaining_points, all_remaining_amps, list_all_remaining_amps = RemainingPoints(all_clusters_in_ev, all_amps_in_ev, all_inner_points, all_inner_amps, all_point_mask)

    new_all_curves_in_ev, new_all_inv_points, new_all_end_points, new_all_nrLPCpoints = NewClustersLPC(all_remaining_points, list_all_remaining_amps, all_clusters_in_ev, all_amps_in_ev, labels)

    all_distances_to_inv, all_distances_to_end, _ = LPCDistances(all_curves_in_ev, all_inv_points)
   
    new_all_distances_to_inv, new_all_distances_to_end, _ = LPCDistances(new_all_curves_in_ev, new_all_inv_points)

    # index_matching, all_connections = ConnectingLPCclusters(all_clusters_in_ev, all_amps_in_ev, all_remaining_points, list_all_remaining_amps, labels)

    all_scalar_products = Angles(all_curves_in_ev, all_inv_points)

    # print("indice per la distanza minima: ", index_matching)
    # print("vettori di collegamento: ", all_connections)

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
    ax.quiver(all_curves_in_ev[0][0][0],all_curves_in_ev[0][0][1],all_curves_in_ev[0][0][2], all_distances_to_inv[0][0][0], all_distances_to_inv[0][0][1], all_distances_to_inv[0][0][2], color='black')
    ax.quiver(all_curves_in_ev[0][1][0],all_curves_in_ev[0][1][1],all_curves_in_ev[0][1][2], all_distances_to_inv[0][1][0], all_distances_to_inv[0][1][1], all_distances_to_inv[0][1][2], color='forestgreen')
    ax.quiver(all_curves_in_ev[0][2][0],all_curves_in_ev[0][2][1],all_curves_in_ev[0][2][2], all_distances_to_inv[0][2][0], all_distances_to_inv[0][2][1], all_distances_to_inv[0][2][2], color='deeppink')
    # ax.quiver(all_curves_in_ev[0][3][0],all_curves_in_ev[0][3][1],all_curves_in_ev[0][3][2], all_distances_to_inv[0][3][0], all_distances_to_inv[0][3][1], all_distances_to_inv[0][3][2], color='olive')
    ax.quiver(all_curves_in_ev[0][0][0],all_curves_in_ev[0][0][1],all_curves_in_ev[0][0][2], all_distances_to_end[0][0][0], all_distances_to_end[0][0][1], all_distances_to_end[0][0][2], color='red')
    ax.quiver(all_curves_in_ev[0][5][0],all_curves_in_ev[0][5][1],all_curves_in_ev[0][5][2], all_distances_to_end[0][1][0], all_distances_to_end[0][1][1], all_distances_to_end[0][1][2], color='cyan')
    ax.quiver(all_curves_in_ev[0][6][0],all_curves_in_ev[0][6][1],all_curves_in_ev[0][6][2], all_distances_to_end[0][2][0], all_distances_to_end[0][2][1], all_distances_to_end[0][2][2], color='darkorange')
    # ax.quiver(all_curves_in_ev[0][7][0],all_curves_in_ev[0][7][1],all_curves_in_ev[0][7][2], all_distances_to_end[0][3][0], all_distances_to_end[0][3][1], all_distances_to_end[0][3][2], color='blue')
  #   ax.quiver(all_curves_in_ev[0][8][0],all_curves_in_ev[0][8][1],all_curves_in_ev[0][8][2], all_distances_to_end[0][4][0], all_distances_to_end[0][4][1], all_distances_to_end[0][4][2], color='grey')
  #   ax.quiver(all_curves_in_ev[0][9][0],all_curves_in_ev[0][9][1],all_curves_in_ev[0][9][2], all_distances_to_end[0][5][0], all_distances_to_end[0][5][1], all_distances_to_end[0][5][2], color='teal')
    
  #   ax.quiver(new_all_curves_in_ev[0][0][0],new_all_curves_in_ev[0][0][1],new_all_curves_in_ev[0][0][2], new_all_distances_to_inv[0][0][0], new_all_distances_to_inv[0][0][1], new_all_distances_to_inv[0][0][2], color='black')
  #   ax.quiver(new_all_curves_in_ev[0][1][0],new_all_curves_in_ev[0][1][1],new_all_curves_in_ev[0][1][2], new_all_distances_to_inv[0][1][0], new_all_distances_to_inv[0][1][1], new_all_distances_to_inv[0][1][2], color='forestgreen')
    
  #   ax.quiver(new_all_curves_in_ev[0][0][0],new_all_curves_in_ev[0][0][1],new_all_curves_in_ev[0][0][2], new_all_distances_to_end[0][0][0], new_all_distances_to_end[0][0][1], new_all_distances_to_end[0][0][2], color='red')
  #   ax.quiver(new_all_curves_in_ev[0][3][0],new_all_curves_in_ev[0][3][1],new_all_curves_in_ev[0][3][2], new_all_distances_to_end[0][1][0], new_all_distances_to_end[0][1][1], new_all_distances_to_end[0][1][2], color='olive')



  #   ax.quiver(new_all_curves_in_ev[1][0][0],new_all_curves_in_ev[1][0][1],new_all_curves_in_ev[1][0][2], new_all_distances_to_inv[1][0][0], new_all_distances_to_inv[1][0][1], new_all_distances_to_inv[1][0][2], color='black')
  #   ax.quiver(new_all_curves_in_ev[1][1][0],new_all_curves_in_ev[1][1][1],new_all_curves_in_ev[1][1][2], new_all_distances_to_inv[1][1][0], new_all_distances_to_inv[1][1][1], new_all_distances_to_inv[1][1][2], color='forestgreen')
  #   ax.quiver(new_all_curves_in_ev[1][2][0],new_all_curves_in_ev[1][2][1],new_all_curves_in_ev[1][2][2], new_all_distances_to_inv[1][2][0], new_all_distances_to_inv[1][2][1], new_all_distances_to_inv[1][2][2], color='deeppink')
  #   ax.quiver(new_all_curves_in_ev[1][3][0],new_all_curves_in_ev[1][3][1],new_all_curves_in_ev[1][3][2], new_all_distances_to_inv[1][3][0], new_all_distances_to_inv[1][3][1], new_all_distances_to_inv[1][3][2], color='olive')

  #   ax.quiver(new_all_curves_in_ev[1][0][0],new_all_curves_in_ev[1][0][1],new_all_curves_in_ev[1][0][2], new_all_distances_to_end[1][0][0], new_all_distances_to_end[1][0][1], new_all_distances_to_end[1][0][2], color='red')
  #   ax.quiver(new_all_curves_in_ev[1][5][0],new_all_curves_in_ev[1][5][1],new_all_curves_in_ev[1][5][2], new_all_distances_to_end[1][1][0], new_all_distances_to_end[1][1][1], new_all_distances_to_end[1][1][2], color='olive')
  #   ax.quiver(new_all_curves_in_ev[1][6][0],new_all_curves_in_ev[1][6][1],new_all_curves_in_ev[1][6][2], new_all_distances_to_end[1][2][0], new_all_distances_to_end[1][2][1], new_all_distances_to_end[1][2][2], color='deeppink')
  #   ax.quiver(new_all_curves_in_ev[1][7][0],new_all_curves_in_ev[1][7][1],new_all_curves_in_ev[1][7][2], new_all_distances_to_end[1][3][0], new_all_distances_to_end[1][3][1], new_all_distances_to_end[1][3][2], color='olive')

    #ax.quiver(*new_all_inv_points[i], all_connections[0][0],all_connections[0][1],all_connections[0][2])

    for c in range(len(all_curves_in_ev)):
      print(len(all_nrLPCpoints[c]))
      ax.scatter3D(all_curves_in_ev[c][:, 0], all_curves_in_ev[c][:,1], all_curves_in_ev[c][:,2], color = 'red')#allLPCpoints
    #   # ax.scatter3D(all_curves_in_ev[c][0, 0], all_curves_in_ev[c][0,1], all_curves_in_ev[c][0,2], color = 'darkorange')#firstLPCpoints
    #   # ax.scatter3D(all_remaining_clusters[c][:,0],all_remaining_clusters[c][:,1],all_remaining_clusters[c][:,2], color = 'darkorange')#remainingcenters
    #   # ax.scatter3D(all_inner_points[c][:, 0], all_inner_points[c][:, 1], all_inner_points[c][:, 2], color = 'forestgreen')#alreadypassedcenters
      ax.scatter3D(all_end_points[c][0],all_end_points[c][1],all_end_points[c][2], color = 'blue')#endLPCpoints
      ax.scatter3D(all_inv_points[c][0],all_inv_points[c][1],all_inv_points[c][2], color = 'cyan')#inversionLPCpoints
    for new_c in range(len(new_all_curves_in_ev)):
      ax.scatter3D(new_all_curves_in_ev[new_c][:, 0], new_all_curves_in_ev[new_c][:,1], new_all_curves_in_ev[new_c][:,2], color = 'blue')
      ax.scatter3D(new_all_curves_in_ev[new_c][0, 0], new_all_curves_in_ev[new_c][0,1], new_all_curves_in_ev[new_c][0,2], color = 'black')
      # ax.scatter3D(new_all_curves_in_ev[new_c][len(new_all_curves_in_ev[new_c])-1, 0], new_all_curves_in_ev[new_c][len(new_all_curves_in_ev[new_c])-1,1], new_all_curves_in_ev[new_c][len(new_all_curves_in_ev[new_c])-1,2], color = 'blue')
      ax.scatter3D(new_all_end_points[new_c][0],new_all_end_points[new_c][1],new_all_end_points[new_c][2], color = 'blue')
      ax.scatter3D(new_all_inv_points[new_c][0],new_all_inv_points[new_c][1],new_all_inv_points[new_c][2], color = 'cyan')
    
    fig2 = plt.figure()
    for c in range(len(all_curves_in_ev)):
      plt.xlabel("lpc point number")
      plt.ylabel("1-|cos(Phi)|")
      plt.scatter(all_nrLPCpoints[c],all_scalar_products[c])
    
    plt.title(selectedEvents[i])
    plt.legend()
    plt.show()




  """only amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=False)

  """gradient amps"""
  # PlotClusters(selectedEvents, fpkl1, fpkl2, geom.fiducial, applyGradient=True)