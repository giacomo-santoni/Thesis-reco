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
        
def HoughTransform(points, theta_resolution=5, rho_resolution=3*defs["voxel_size"]):
  #theta and rho limits
  max_rho = int(np.ceil(np.sqrt(235**2 + 735**2)))#half the diagonal
  thetas = np.deg2rad(np.arange(0,360,theta_resolution))
  rhos = np.arange(-max_rho, max_rho, rho_resolution)

  accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
  
  par_points = []
  for coord1,coord2 in points:
    for theta_index, theta in enumerate(thetas):
      rho = coord1 * np.cos(theta) + coord2 * np.sin(theta)
      rho_index = np.argmin(np.abs(rhos - rho))
      #print(rho_index,theta_index)
      accumulator[rho_index, theta_index] += 1
      par_points.append((rho_index,theta_index,coord1,coord2))
  return accumulator, rhos, thetas, par_points

def FindNextMaximum(matrix):
    all_max = []
    for _ in range(2):
      max = np.max(matrix)
      flat_matrix = matrix.flatten()
      matrix = flat_matrix[flat_matrix < max]
      next_max = np.max(matrix) 
      print("max: ", next_max)
      all_max.append(next_max)
    return all_max

def is_point_in_array(point, array):
    return any(np.all(point == i) for i in array)


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

    global_max_indices = np.argwhere(accumulator == np.max(accumulator))
    all_max = FindNextMaximum(accumulator)
    print(all_max[0], all_max[1])
    local_max_indices = np.argwhere(accumulator == all_max[0])
    local_max_indices2 = np.argwhere(accumulator == all_max[1])

    print("global: ", global_max_indices)
    print("local: ", local_max_indices)
    print("local2: ", local_max_indices2)
    all_indices = np.concatenate((global_max_indices,local_max_indices, local_max_indices2))
    print("all: ", all_indices)

    print("values: ", rhos[global_max_indices[0][0]], np.rad2deg(thetas[global_max_indices[0][1]]))
    print(np.max(accumulator))
    print(accumulator[global_max_indices[0][0],global_max_indices[0][1]])
    #print(accumulator)

    all_collinear_pt = []
    for index in all_indices:
      collinear_points = []
      for t in indices_points:
        if t[0] == index[0] and t[1] == index[1]:
          point = np.asarray([t[2],t[3]])
          collinear_points.append(point)
      all_collinear_pt.append(collinear_points)
    print(len(all_collinear_pt))

    import scipy as scp
    neighborhood_size = 5
    threshold = 10
    data_max = scp.ndimage.maximum_filter(accumulator, neighborhood_size)
    print("max: ", data_max)
    maxima = (accumulator == data_max)
    data_min = scp.ndimage.minimum_filter(accumulator, neighborhood_size)
    print("min: ", data_min)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    print(maxima)

    labeled, num_objects = scp.ndimage.label(maxima)
    #print("lab: ",num_objects)
    slices = scp.ndimage.find_objects(labeled)
    print("slices: ", slices)
    x, y = [], []
    for dy,dx in slices:
        print(dy,dx)
        print(dx.start, dx.stop)
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
        print("indices_local_max: ", (x,y))

    plt.imshow(accumulator, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
    plt.xlabel('theta')
    plt.ylabel('rho')
    plt.colorbar()

   #plt.autoscale(False)
    plt.plot(np.rad2deg(thetas[6]),rhos[15], 'ro')
    plt.savefig('/Users/giacomosantoni/Desktop/result.png', bbox_inches = 'tight')

    
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
    all_collinear_pt[2] = np.asarray(all_collinear_pt[2])
    plt.scatter(all_collinear_pt[2][:,1], all_collinear_pt[2][:,0], color = 'grey')
    all_collinear_pt[3] = np.asarray(all_collinear_pt[3])
    plt.scatter(all_collinear_pt[3][:,1], all_collinear_pt[3][:,0], color = 'yellow')
    plt.ylim(-400,400)
    plt.xlim(-200,200)

    from matplotlib.patches import Circle

    center = (0, 0)
    radius = 220
    circle = Circle(center, radius,facecolor = None, edgecolor = 'blue',alpha=0.1)
    ax.add_patch(circle)
    
    plt.title("y-z plane")
    plt.xlabel("z")
    plt.ylabel("y")

    plt.show()