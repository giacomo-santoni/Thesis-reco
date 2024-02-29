import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy as scp
import skimage as sk
import scipy as scp

import clusterclass
from recodisplay import load_pickle
from geom_import import load_geometry
from mctruth import loadPrimariesEdepSim

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
def AllCentersAllEvents(events, fname1):
  centers_all_ev = []
  amps_all_ev = []
  recodata_all_ev = []
  for ev in events:
    data = load_pickle(fname1, ev)
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
  
  #coord1=z, coord2=y oppure coord2=x
  for coord1,coord2 in points:
    for theta_index, theta in enumerate(thetas):
      rho = coord2 * np.cos(theta) + coord1 * np.sin(theta)
      rho_index = np.argmin(np.abs(rhos - rho))
      accumulator[rho_index, theta_index] += 1
  return accumulator, rhos, thetas

def FindLocalMaxima(accumulator):
  local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=4, threshold_rel = 0.6, exclude_border=False)
  
  if len(local_max_indices) > 2:
    local_max_indices = local_max_indices[:2]
  
  rho_thetas_max = []
  for i in local_max_indices:
    theta_max = thetas[i[1]]
    rho_max = rhos[i[0]]
    rho_thetas_max.append((rho_max, theta_max))
  return local_max_indices, rho_thetas_max

def FindClosestToLinePointsZY(points, rho_thetas_max):
  all_collinear_pointsZY = []
  collinear_points1 = []
  collinear_points2 = []
  all_distancesZY = []
  n=0
  #coord1=x, coord2=y, coord3=z
  for coord1,coord2,coord3 in points:
    n+=1
    dist_point_to_lines = []
    for rho,theta in rho_thetas_max:
      d = abs((np.cos(theta)*coord2 + np.sin(theta)*coord3 - rho)) / (np.sqrt(np.cos(theta)*np.cos(theta) + np.sin(theta)*np.sin(theta)))
      dist_point_to_lines.append((d,rho,theta))
    min_dist = np.min(np.asarray(dist_point_to_lines)[:,0])
    
    closest_line = []
    for tuple in dist_point_to_lines: 
      if tuple[0] == min_dist and tuple[0]<45:
        closest_line = (tuple[1],tuple[2])
      if tuple[0] == min_dist:
        all_distancesZY.append(tuple[0])

    if closest_line == rho_thetas_max[0]:
      collinear_points1.append((coord1,coord2,coord3))
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))

  all_collinear_pointsZY.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0:
    all_collinear_pointsZY.append(collinear_points2)
  return all_collinear_pointsZY, all_distancesZY

def FindClosestToLinePointsZX(points, rho_thetas_max):
  all_collinear_pointsZX = []
  collinear_points1 = []
  collinear_points2 = []
  all_distancesZX = []
  n=0
  #coord1=x, coord2=y, coord3=z
  for coord1,coord2,coord3 in points:
    n+=1
    dist_point_to_lines = []
    for rho,theta in rho_thetas_max:
      d = abs((np.cos(theta)*coord1 + np.sin(theta)*coord3 - rho)) / (np.sqrt(np.cos(theta)*np.cos(theta) + np.sin(theta)*np.sin(theta)))
      dist_point_to_lines.append((d,rho,theta))
      all_distancesZX.append(d)
    min_dist = np.min(np.asarray(dist_point_to_lines)[:,0])
    
    closest_line = []
    for tuple in dist_point_to_lines: 
      if tuple[0] == min_dist and tuple[0]<45:
        closest_line = (tuple[1],tuple[2])

    if closest_line == rho_thetas_max[0]:
      collinear_points1.append((coord1,coord2,coord3))
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))

  all_collinear_pointsZX.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0:
    all_collinear_pointsZX.append(collinear_points2)
  return all_collinear_pointsZX, all_distancesZX

def ExtractTrueParameters(fname, ev):
  true_event = loadPrimariesEdepSim(fname, ev)
  true_vertices = true_event.vertices
  vertices_coord = []
  for vertex in true_vertices:
    vertices_coord.append(vertex.position)
    particles = vertex.particles
    directions = []
    for particle in particles:
      momentum = particle.momentum
      directions.append(momentum)
  return vertices_coord, directions

def Fit(points1, points2):
  slope, intercept = np.polyfit(points1, points2, 1)
  #plt.plot(points1, slope*points2 + intercept, color='red')
  return slope, intercept

def GetRecoVertex(all_collinear_points):
  all_slopesZY = []
  all_interceptsZY = []
  all_slopesZX = []
  all_interceptsZX = []
  for collinear_points in all_collinear_points:
    if len(collinear_points) != 0:
      #ZY
      collinear_points = np.asarray(collinear_points)
      slopeZY, interceptZY = Fit(collinear_points[:,2], collinear_points[:,1])
      all_slopesZY.append(slopeZY)
      all_interceptsZY.append(interceptZY)
      #ZX
      slopeZX, interceptZX = Fit(collinear_points[:,2], collinear_points[:,0])
      all_slopesZX.append(slopeZX)
      all_interceptsZX.append(interceptZX)
    
  if len(all_interceptsZY)>1 or len(all_interceptsZX)>1:
    z_vertex1 = (all_interceptsZY[1] - all_interceptsZY[0])/(all_slopesZY[0] - all_slopesZY[1])
    y_vertex = all_slopesZY[0]*z_vertex1 + all_interceptsZY[0]
    #print("ZY vertex: ", z_vertex1, y_vertex)

    z_vertex2 = (all_interceptsZX[1] - all_interceptsZX[0])/(all_slopesZX[0] - all_slopesZX[1])
    x_vertex = all_slopesZX[0]*z_vertex2 + all_interceptsZX[0]
    #print("ZX vertex: ", z_vertex2, x_vertex)

    z_vertex = (z_vertex1 + z_vertex2)/2

    x_vertex = slopeZX*z_vertex + interceptZX
    reco_vertex = (x_vertex, y_vertex, z_vertex)
  else: reco_vertex = "Vertex not found or not present"
  return reco_vertex

def GetRecoDirections(reco_vertex, all_collinear_points):
  all_reco_directions = []
  for collinear_points in all_collinear_points: 
    if len(collinear_points) != 0:
      collinear_points = np.asarray(collinear_points)
      slopeZY, interceptZY = Fit(collinear_points[:,2], collinear_points[:,1])
      slopeZX, interceptZX = Fit(collinear_points[:,2], collinear_points[:,0])
      z = collinear_points[0,2]
      point2 = np.asarray((slopeZX*z + interceptZX, slopeZY*z + interceptZY, z))#x,y,z
      if isinstance(reco_vertex, np.ndarray):
        reco_vertex = np.asarray(reco_vertex)
        direction = (reco_vertex - point2)
      elif not isinstance(reco_vertex, np.ndarray):
        if len(collinear_points)>1:
          z1 = collinear_points[1,2]
          point1 = np.asarray((slopeZX*z1 + interceptZX, slopeZY*z1 + interceptZY, z1))#x,y,z
          direction = (point1 - point2)
          all_reco_directions.append(direction)  
  return all_reco_directions 

def GetRecoAngle(reco_directions, true_directions):
  min_thetas = []
  for r_dir in reco_directions:
    all_thetas = []
    for t_dir in true_directions:
      theta = np.arccos(np.abs(np.dot(r_dir,t_dir)/(np.linalg.norm(r_dir)*np.linalg.norm(t_dir))))
      print("angle: ", np.rad2deg(theta))
      all_thetas.append(theta)
      # print("all_thetas: ", np.rad2deg(all_thetas))
    min_thetas.append(np.rad2deg(np.min(all_thetas)))
  return min_thetas


if __name__ == '__main__':
  #eventNumbers = EventNumberList("./data/data_19-2/id-list/idlist.1.txt")
  eventNumbers = EventNumberList("./data/data_1-12/idlist_ccqe_mup.txt")
  selectedEvents = [eventNumbers[4],eventNumbers[7],eventNumbers[16],eventNumbers[18],eventNumbers[19]]

  geometryPath = "./geometry" #path to GRAIN geometry
  #fpkl = "./data/data_19-2/pickles/3dreco_1.pkl"
  fpkl = "./data/data_1-12/3dreco_ccqe_mup.pkl"


  #edepsim_file = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.1.edep-sim.root"
  edepsim_file = "./data/data_1-12/events-in-GRAIN_LAr_lv.999.edep-sim.root"
  geom = load_geometry(geometryPath, defs)

  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl)

  all_distances_all_evZY = []
  all_distances_all_evZX = []
  final_dist = []

  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    print("*******************NUOVO EVENTO***********************")
    print("evento: ", selectedEvents[i])

    #************************MC TRUTH**************************
    true_vertices, true_directions = ExtractTrueParameters(edepsim_file, selectedEvents[i])
    print("---------------------MC TRUTH-------------------------")
    print("TRUE vertex: ", true_vertices)
    print("TRUE directions: ", true_directions)
    #**********************************************************

    #**************************ACCUMULATOR***************************
    if len(all_clusters_in_ev) != 0:
      all_lpcs = np.concatenate([cluster.LPCs[0] for cluster in all_clusters_in_ev])
      points = list(zip(all_lpcs[:,0],all_lpcs[:,1],all_lpcs[:,2]))
      pointsZY = list(zip(all_lpcs[:,2],all_lpcs[:,1]))
      pointsZX = list(zip(all_lpcs[:,2],all_lpcs[:,0]))
    
    accumulatorZY, rhos, thetas = HoughTransform(pointsZY)
    accumulatorZX, _, _ = HoughTransform(pointsZX)

    local_max_indicesZY, rho_thetas_maxZY = FindLocalMaxima(accumulatorZY)
    local_max_indicesZX, rho_thetas_maxZX = FindLocalMaxima(accumulatorZX)
    print("indices_local_max (rows, columns) ZY: ", local_max_indicesZY)
    print("len: ", len(local_max_indicesZY))
    print("indices_local_max (rows, columns) ZX: ", local_max_indicesZX)
    print("rhos thetas max ZY: ", rho_thetas_maxZY)
    print("rhos thetas max ZX: ", rho_thetas_maxZX)


    for j in local_max_indicesZY:
      local_max_values = accumulatorZY[j[0],j[1]]
      # print("max_values: ", local_max_values)
      # print("theta: ", np.rad2deg(thetas[i[1]]))
      # print("rho: ", rhos[i[0]])

    # fig1 = plt.figure()
    # plt.imshow(accumulatorZY, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
    # plt.xlabel('theta')
    # plt.ylabel('rho')
    # plt.title('ZY')
    # plt.colorbar()

    # for k in local_max_indicesZY:
    #   plt.plot(np.rad2deg(thetas[k[1]]),rhos[k[0]], 'ro')

    # fig2 = plt.figure()
    # plt.imshow(accumulatorZX, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
    # plt.xlabel('theta')
    # plt.ylabel('rho')
    # plt.title('ZX')
    # plt.colorbar()

    # for w in local_max_indicesZX:
    #   plt.plot(np.rad2deg(thetas[w[1]]),rhos[w[0]], 'bo')
    #*********************************************************************

    #***********************************2D PLOT ZY***************************
    all_collinear_pointsZY, all_distancesZY = FindClosestToLinePointsZY(points, rho_thetas_maxZY)
    print("all_dist zy: ", all_distancesZY)
    all_distances_all_evZY.append(all_distancesZY)

    fig3 = plt.figure()
    ax = fig3.add_subplot()
    for cluster in all_clusters_in_ev:
      single_curve = np.asarray(cluster.LPCs[0])
      plt.scatter(single_curve[:,2], single_curve[:,1], color = 'red')#allLPCpoints
    
    #i collinear points sono organizzati come x,y,z; li disegno nel piano z-y
    if all_collinear_pointsZY[0] != []:
      all_collinear_pointsZY[0] = np.asarray(all_collinear_pointsZY[0])
      plt.scatter(all_collinear_pointsZY[0][:,2], all_collinear_pointsZY[0][:,1], color = 'green')
      if len(all_collinear_pointsZY)>1:
        if all_collinear_pointsZY[1] != []:
          all_collinear_pointsZY[1] = np.asarray(all_collinear_pointsZY[1])
          plt.scatter(all_collinear_pointsZY[1][:,2], all_collinear_pointsZY[1][:,1], color = 'blue')
    plt.ylim(-700,700)
    plt.xlim(-200,200)
    plt.gca().set_aspect('equal', adjustable='box')

    # """HOUGH TRANSFORM LINES"""
    # for rho,theta in rho_thetas_maxZY:
    #   z = np.linspace(-400, 400, 10000)
    #   y = -(np.cos(theta)/np.sin(theta))*z + rho/np.sin(theta)
    #   plt.plot(y, z)

    # for collinear_points in all_collinear_pointsZY:
    #   if len(collinear_points) != 0:
    #     collinear_points = np.asarray(collinear_points)
    #     slope, intercept = Fit(collinear_points[:,2], collinear_points[:,1])
    #     plt.plot(collinear_points[:,2], slope*collinear_points[:,2] + intercept, color = 'red')
  
    plt.grid()
    plt.title("z-y plane")
    plt.xlabel("z (mm)")
    plt.ylabel("y (mm)")
    #*********************************************************************

    #***********************************2D PLOT ZX***************************
    all_collinear_pointsZX, all_distancesZX = FindClosestToLinePointsZX(points, rho_thetas_maxZX)
    all_distances_all_evZX.append(all_distancesZX)

    # fig4 = plt.figure()
    # ax = fig4.add_subplot()
    # for cluster in all_clusters_in_ev:
    #   single_curve = np.asarray(cluster.LPCs[0])
    #   plt.scatter(single_curve[:,2], single_curve[:,0], color = 'red')#allLPCpoints

    # #i collinear points sono organizzati come x,y,z; li disegno nel piano z-x
    # if all_collinear_pointsZX[0] != []:
    #   all_collinear_pointsZX[0] = np.asarray(all_collinear_pointsZX[0])
    #   plt.scatter(all_collinear_pointsZX[0][:,2], all_collinear_pointsZX[0][:,0], color = 'green')
    #   if len(all_collinear_pointsZX)>1:
    #     if all_collinear_pointsZX[1] != []:
    #       all_collinear_pointsZX[1] = np.asarray(all_collinear_pointsZX[1])
    #       plt.scatter(all_collinear_pointsZX[1][:,2], all_collinear_pointsZX[1][:,0], color = 'blue')
    # plt.ylim(-1000,1000)
    # plt.xlim(-200,200)
    # plt.gca().set_aspect('equal', adjustable='box')

    # """HOUGH TRANSFORM LINES"""
    # for rho,theta in rho_thetas_maxZX:
    #   z = np.linspace(-400, 400, 10000)
    #   x = -(np.cos(theta)/np.sin(theta))*z + rho/np.sin(theta)
    #   plt.plot(x, z)

    # for collinear_points in all_collinear_pointsZX:
    #   if len(collinear_points) != 0:
    #     collinear_points = np.asarray(collinear_points)
    #     slope, intercept = Fit(collinear_points[:,2], collinear_points[:,0])
    #     plt.plot(collinear_points[:,2], slope*collinear_points[:,2] + intercept, color = 'red')

    # plt.grid()
    # plt.title("z-x plane")
    # plt.xlabel("z (mm)")
    # plt.ylabel("x (mm)")
    #*********************************************************************

    # """RECO VERTEX"""
    # print("---------------------------RECO-----------------------------")
    # reco_vertex = GetRecoVertex(all_collinear_points)
    # print("RECO vertex: ", reco_vertex)

    # """RECO DIRECTION"""
    # all_reco_directions = GetRecoDirections(reco_vertex, all_collinear_points)
    # print("RECO directions: ", all_reco_directions)

    # """ANGLE"""
    # theta = GetRecoAngle(all_reco_directions, true_directions)
    # print("RECO angle: ", theta)

    #************************3D PLOT****************************
    # fig5 = plt.figure()
    # ax = fig5.add_subplot(projection='3d')
    # scalesz = np.max(recodata_all_ev[i].shape) * 12 / 1.6
    # ax.set_xlim([-scalesz, scalesz])
    # ax.set_ylim([-scalesz, scalesz])
    # ax.set_zlim([-scalesz, scalesz])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.scatter3D(centers_all_ev[i][:, 0], centers_all_ev[i][:,1], centers_all_ev[i][:,2], c = y_pred, cmap = 'cividis',s=15)
    # for cluster in all_clusters_in_ev:
    #   single_curve = np.asarray(cluster.LPCs[0])
    #   cluster.FindBreakPoint()
    #   ax.scatter3D(single_curve[:, 0], single_curve[:,1], single_curve[:,2], color = 'red')#allLPCpoints
    # # ax.quiver(0, 0, 0, all_reco_directions[0][0], all_reco_directions[0][1], all_reco_directions[0][2], color='r', arrow_length_ratio=0.1)
    # # ax.quiver(0, 0, 0, true_directions[0][0], true_directions[0][1], true_directions[0][2], color='b', arrow_length_ratio=0.1)
    # # ax.quiver(0, 0, 0, all_reco_directions[1][0], all_reco_directions[1][1], all_reco_directions[1][2], color='g', arrow_length_ratio=0.1)
    # # ax.quiver(0, 0, 0, true_directions[1][0], true_directions[1][1], true_directions[1][2], color='b', arrow_length_ratio=0.1)
    # plt.title(selectedEvents[i])
    # plt.legend()
    #************************************************************
    fig7 = plt.figure()
    plt.hist(all_distancesZY, 100)
    plt.title(selectedEvents[i])

  for all_dist in all_distances_all_evZY:
    for dist in all_dist:
      final_dist.append(dist)
  print("all_dist: ", final_dist)

  fig9 = plt.figure()
  plt.hist(final_dist, 50, histtype='step')
  plt.xlabel("distance lpc point from the closest hough line")
  plt.ylabel("n entries")
  plt.title("ZY plane")

  plt.show()
  
  # fig10 = plt.figure()
  # plt.hist(all_distances_all_evZY, 50)

  