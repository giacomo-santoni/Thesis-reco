import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy as scp
import skimage as sk

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
        
def HoughTransform(points, theta_resolution=5, rho_resolution=3*defs["voxel_size"]):
  #theta and rho limits
  max_rho = int(np.ceil(np.sqrt(235**2 + 735**2)))#half the diagonal
  thetas = np.deg2rad(np.arange(0,180,theta_resolution))
  rhos = np.arange(-max_rho, max_rho, rho_resolution)

  accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
  
  par_points = []
  for coord1,coord2 in points:
    for theta_index, theta in enumerate(thetas):
      rho = coord1 * np.cos(theta) + coord2 * np.sin(theta)
      rho_index = np.argmin(np.abs(rhos - rho))
      accumulator[rho_index, theta_index] += 1
      par_points.append((rho_index,theta_index,coord1,coord2))
  return accumulator, rhos, thetas, par_points

def FindLocalMaxima(accumulator, neighborhood_size = 10, threshold = 6):
  data_max = scp.ndimage.maximum_filter(accumulator, neighborhood_size)
  maxima = (accumulator == data_max)
  #data_min = scp.ndimage.minimum_filter(accumulator, neighborhood_size)
  diff = (data_max > threshold)
  maxima[diff == 0] = 0

  labeled, num_objects = scp.ndimage.label(maxima)
  slices = scp.ndimage.find_objects(labeled)
  local_max_indices = []
  for dy,dx in slices:#dy sono le righe (rho), dx sono le colonne (theta)
      subarray = accumulator[dy, dx]
      idx_max = np.unravel_index(np.argmax(subarray), subarray.shape)
      local_max_indices.append((dy.start + idx_max[0], dx.start + idx_max[1]))

  #2
  # Comparison between image_max and im to find the coordinates of local maxima
  # local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=5)
 
  #3
  # window_size = (5, 5)
  # local_max = scp.ndimage.maximum_filter(accumulator, size=window_size)
  # local_max_indices = np.argwhere((local_max == accumulator) & (local_max > 10))

  print(local_max_indices)
  return local_max_indices

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
      # if cluster.break_point != 0:
      #   cluster.BreakLPCs()
      #   broken_curve = np.asarray(cluster.broken_lpccurve[0])
      #   broken_curve2 = np.asarray(cluster.broken_lpccurve[1])
      #   ax.scatter3D(broken_curve[:,0], broken_curve[:,1], broken_curve[:,2], color = 'green')
      #   ax.scatter3D(broken_curve2[:,0], broken_curve2[:,1], broken_curve2[:,2], color = 'purple')
      #   ax.scatter3D(single_curve[cluster.break_point][0], single_curve[cluster.break_point][1], single_curve[cluster.break_point][2], color = 'blue', marker='D')
    plt.title(selectedEvents[i])
    plt.legend()

    fig3 = plt.figure()
    all_lpcs = np.concatenate([cluster.LPCs[0] for cluster in all_clusters_in_ev])
    pointsYZ = zip(all_lpcs[:,1],all_lpcs[:,2])
    pointsXZ = zip(all_lpcs[:,0],all_lpcs[:,2])
    accumulator, rhos, thetas, indices_points = HoughTransform(pointsYZ)

    local_max_indices = FindLocalMaxima(accumulator)
    print("indices_local_max (rows, columns): ", local_max_indices)
    
    for i in local_max_indices:
      local_max_values = accumulator[i[0],i[1]]
      print("max_values: ", local_max_values)
      print("theta: ", np.rad2deg(thetas[i[1]]))
      print("rho: ", rhos[i[0]])

    all_collinear_pt = []
    for index in local_max_indices:
      collinear_points = []
      print("altro indice: ", index)
      for t in indices_points:
        if t[0] == index[0] and t[1] == index[1]:
          print(t)
          point = np.asarray([t[2],t[3]])
          #print(point)
          collinear_points.append(point)
      all_collinear_pt.append(collinear_points)
    # print("primo: ", all_collinear_pt[0], len(all_collinear_pt[0]))
    # print("secondo: ", all_collinear_pt[1], len(all_collinear_pt[1]))
    # print("terzo: ", all_collinear_pt[2], len(all_collinear_pt[2]))
    #print("quarto: ", all_collinear_pt[3], len(all_collinear_pt[3]))
    
    plt.imshow(accumulator, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
    plt.xlabel('theta')
    plt.ylabel('rho')
    plt.colorbar()

    for i in local_max_indices:
      plt.plot(np.rad2deg(thetas[i[1]]),rhos[i[0]], 'ro')


    fig2 = plt.figure()
    ax = fig2.add_subplot()
    for cluster in all_clusters_in_ev:
      single_curve = np.asarray(cluster.LPCs[0])
      ax.scatter(single_curve[:,2], single_curve[:,1], color = 'red')#allLPCpoints
    #for collinear_points in all_collinear_pt:
    all_collinear_pt[0] = np.asarray(all_collinear_pt[0])
    ax.scatter(all_collinear_pt[0][:,1], all_collinear_pt[0][:,0], color = 'green')
    all_collinear_pt[1] = np.asarray(all_collinear_pt[1])
    plt.scatter(all_collinear_pt[1][:,1], all_collinear_pt[1][:,0], color = 'blue')
    # all_collinear_pt[2] = np.asarray(all_collinear_pt[2])
    # plt.scatter(all_collinear_pt[2][:,1], all_collinear_pt[2][:,0], color = 'grey')
    # all_collinear_pt[3] = np.asarray(all_collinear_pt[3])
    # plt.scatter(all_collinear_pt[3][:,1], all_collinear_pt[3][:,0], color = 'yellow')
    # all_collinear_pt[4] = np.asarray(all_collinear_pt[4])
    # # plt.scatter(all_collinear_pt[4][:,1], all_collinear_pt[4][:,0], color = 'black')
    # # all_collinear_pt[5] = np.asarray(all_collinear_pt[5])
    # # plt.scatter(all_collinear_pt[5][:,1], all_collinear_pt[5][:,0], color = 'silver')
    # # all_collinear_pt[6] = np.asarray(all_collinear_pt[6])
    # # plt.scatter(all_collinear_pt[6][:,1], all_collinear_pt[6][:,0], color = 'brown')
    # # all_collinear_pt[7] = np.asarray(all_collinear_pt[7])
    # # plt.scatter(all_collinear_pt[7][:,1], all_collinear_pt[7][:,0], color = 'aqua')
    # all_collinear_pt[8] = np.asarray(all_collinear_pt[8])
    # plt.scatter(all_collinear_pt[8][:,1], all_collinear_pt[8][:,0], color = 'chartreuse')
    # all_collinear_pt[9] = np.asarray(all_collinear_pt[9])
    # plt.scatter(all_collinear_pt[9][:,1], all_collinear_pt[9][:,0], color = 'indigo')
    # all_collinear_pt[10] = np.asarray(all_collinear_pt[10])
    # plt.scatter(all_collinear_pt[10][:,1], all_collinear_pt[10][:,0], color = 'lightpink')
    # all_collinear_pt[11] = np.asarray(all_collinear_pt[11])
    # plt.scatter(all_collinear_pt[11][:,1], all_collinear_pt[11][:,0], color = 'magenta')
    # all_collinear_pt[12] = np.asarray(all_collinear_pt[12])
    # plt.scatter(all_collinear_pt[12][:,1], all_collinear_pt[12][:,0], color = 'darkkhaki')
    # all_collinear_pt[13] = np.asarray(all_collinear_pt[3])
    # plt.scatter(all_collinear_pt[13][:,1], all_collinear_pt[13][:,0], color = 'coral')
    plt.ylim(-700,700)
    plt.xlim(-200,200)
    plt.gca().set_aspect('equal', adjustable='box')

    # from matplotlib.patches import Circle

    # center = (0, 0)
    # radius = 220
    # circle = Circle(center, radius,facecolor = None, edgecolor = 'blue',alpha=0.1)
    # ax.add_patch(circle)
    
    plt.title("y-z plane")
    plt.xlabel("z")
    plt.ylabel("y")

    plt.show()