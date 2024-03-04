import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy as scp
import skimage as sk

import clusterclass
from recodisplay import load_pickle
from geom_import import load_geometry
from mctruth import loadPrimariesEdepSim

defs = {}
defs['voxel_size'] = 12
#bandwidth = 30, stepsize = 55
all_parameters = {"epsilon" : 3*defs['voxel_size'], "min_points" : 6, "bandwidth" : 2.5*defs['voxel_size'], "stepsize" : 4.6*defs['voxel_size'], "cuts" : [100,500], "g_cut_perc" : 0.05, "it" : 500, "g_it" : 200}

def EventNumberList(file_list):
  all_events = []
  for file in file_list:
    f = open(file, "r")
    events_string = []
    events = []

    for x in f:
        events_string.append(x)
    
    for i in events_string:
      new_ev = "".join(x for x in i if x.isdecimal())
      events.append(int(new_ev))
    all_events.append(events)
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
def AllCentersAllEvents(events, file_list):
  centers_all_ev = []
  amps_all_ev = []
  recodata_all_ev = []
  tot_events = []
  for j, file in enumerate(file_list):
    for ev in events[j]:
      tot_events.append(ev)
      data = load_pickle(file, ev)
      recodata = data[all_parameters["it"]]
      data_masked = recodata[5:-5,5:-5,5:-5]
      cut = all_parameters["cuts"][0]
      centers, amps = getVoxelsCut(data_masked, geom.fiducial, cut)
      centers = np.transpose(centers)
      recodata_all_ev.append(recodata)
      centers_all_ev.append(centers)
      amps_all_ev.append(amps)
  recodata_all_ev = np.asarray(recodata_all_ev)
  return centers_all_ev, amps_all_ev, recodata_all_ev, tot_events

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

def FindLocalMaxima(accumulator,lpcs):
  #print("len lpcs: ", len(lpcs))
  # if len(lpcs)>=12: 
  #   thr = len(lpcs)*0.3
  #   local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=7, threshold_abs = thr, exclude_border=False)
  # if len(lpcs)<12:
  #   thr = len(lpcs)*0.5
  #   local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=7, threshold_abs = thr, exclude_border=False)
  thr = len(lpcs)*0.35
  local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=7, threshold_abs = thr, exclude_border=False)

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

  if collinear_points1 != []:
    all_collinear_pointsZY.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0 and collinear_points2 != []:
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
    min_dist = np.min(np.asarray(dist_point_to_lines)[:,0])
    
    closest_line = []
    for tuple in dist_point_to_lines: 
      if tuple[0] == min_dist and tuple[0]<45:
        closest_line = (tuple[1],tuple[2])
      if tuple[0] == min_dist:
        all_distancesZX.append(tuple[0])

    if closest_line == rho_thetas_max[0]:
      collinear_points1.append((coord1,coord2,coord3))
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))

  if collinear_points1  != []:
    all_collinear_pointsZX.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0 and collinear_points2 != []:
    all_collinear_pointsZX.append(collinear_points2)
  return all_collinear_pointsZX, all_distancesZX

def ExtractTrueParameters(file_list, events):
  vertices_coord = []
  ev_directions = []
  ev_numbers = []
  for j, fname in enumerate(file_list): 
    for ev in events[j]:
      true_event = loadPrimariesEdepSim(fname, ev)
      ev_numbers.append(true_event.eventID)
      true_vertices = true_event.vertices
      for vertex in true_vertices:
        vertices_coord.append(vertex.position)
        particles = vertex.particles
        particles_directions = []
        for particle in particles:
          momentum = particle.momentum
          particles_directions.append(momentum)
        ev_directions.append(particles_directions)
  return vertices_coord, ev_directions, ev_numbers

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
    #if len(collinear_points) != 0:
    collinear_points = np.asarray(collinear_points)
    slopeZY, interceptZY = Fit(collinear_points[:,2], collinear_points[:,1])
    slopeZX, interceptZX = Fit(collinear_points[:,2], collinear_points[:,0])
    z = collinear_points[0,2]
    point2 = np.asarray((slopeZX*z + interceptZX, slopeZY*z + interceptZY, z))#x,y,z
    #if isinstance(reco_vertex, np.ndarray):
    reco_vertex = np.asarray(reco_vertex)
    direction = (reco_vertex - point2)
    all_reco_directions.append(direction)
    # elif not isinstance(reco_vertex, np.ndarray):
    #   if len(collinear_points)>1:
    #     z1 = collinear_points[1,2]
    #     point1 = np.asarray((slopeZX*z1 + interceptZX, slopeZY*z1 + interceptZY, z1))#x,y,z
    #     direction = (point1 - point2)
    #     all_reco_directions.append(direction)  
  return all_reco_directions 

def GetRecoAngle(reco_directions, true_directions):
  all_angles = []
  min_thetas = []
  if reco_directions != []:
    for r_dir in reco_directions:
      thetas_one_track = []
      for t_dir in true_directions:
        theta = np.arccos(np.dot(r_dir,t_dir)/(np.linalg.norm(r_dir)*np.linalg.norm(t_dir)))
        if np.rad2deg(theta) > 90:
          alpha = - (180 - np.rad2deg(theta))
          thetas_one_track.append(np.deg2rad(alpha))
        elif np.rad2deg(theta) <= 90:
          thetas_one_track.append(theta)
      sel_theta = [theta for theta in thetas_one_track if abs(theta) == (np.min(np.abs(thetas_one_track)))]
      min_thetas.append((np.rad2deg(sel_theta)))
  for theta_arr in min_thetas:
    for theta in theta_arr:
      all_angles.append(theta)
  return all_angles

def VertexCoordinatesHisto(true_vertices, reco_vertices, coord):
  diff_vertices = []
  n = 0
  p = 0
  for i, true_vertex in enumerate(true_vertices):
    p += 1
    if isinstance(reco_vertices[i], tuple):
      n += 1
      diff = (np.asarray(reco_vertices[i][coord]) - np.asarray(true_vertex[coord]))
      diff_vertices.append(diff)
  print("dopo if array: ", n)
  print("prima if array: ", p)
  return diff_vertices

def DistanceRecoTrueVertex(true_vertices, reco_vertices):
  distance_vertices = []
  for i, true_vertex in enumerate(true_vertices):
    if isinstance(reco_vertices[i], tuple):
      dist = np.linalg.norm(np.asarray(reco_vertices[i]) - np.asarray(true_vertex))
      distance_vertices.append(dist)
  return distance_vertices

def gauss(x,amp,mu,sigma):
  return amp*np.exp(-((x-mu)**2)/(2*sigma**2))



if __name__ == '__main__':
  list1 = "./data/data_19-2/id-list/idlist.1.txt"
  list2 = "./data/data_19-2/id-list/idlist.2.txt"
  list3 = "./data/data_19-2/id-list/idlist.3.txt"
  list4 = "./data/data_19-2/id-list/idlist.4.txt"
  list5 = "./data/data_19-2/id-list/idlist.5.txt"
  list6 = "./data/data_19-2/id-list/idlist.6.txt"
  list7 = "./data/data_19-2/id-list/idlist.7.txt"
  list8 = "./data/data_19-2/id-list/idlist.8.txt"
  list9 = "./data/data_19-2/id-list/idlist.9.txt"
  list10 = "./data/data_19-2/id-list/idlist.10.txt"
  id_list = [list1, list2, list3, list4, list5, list6, list7, list8, list9, list10]

  selectedEvents = (EventNumberList(id_list))
  print("selec events: ", selectedEvents)
  #eventNumbers = EventNumberList("./data/data_1-12/idlist_ccqe_mup.txt")
  #selectedEvents = eventNumbers#[eventNumbers[4],eventNumbers[7],eventNumbers[16],eventNumbers[18],eventNumbers[19]]

  geometryPath = "./geometry" #path to GRAIN geometry
  fpkl1 = "./data/data_19-2/pickles/3dreco_1.pkl"
  fpkl2 = "./data/data_19-2/pickles/3dreco_2.pkl"
  fpkl3 = "./data/data_19-2/pickles/3dreco_3.pkl"
  fpkl4 = "./data/data_19-2/pickles/3dreco_4.pkl"
  fpkl5 = "./data/data_19-2/pickles/3dreco_5.pkl"
  fpkl6 = "./data/data_19-2/pickles/3dreco_6.pkl"
  fpkl7 = "./data/data_19-2/pickles/3dreco_7.pkl"
  fpkl8 = "./data/data_19-2/pickles/3dreco_8.pkl"
  fpkl9 = "./data/data_19-2/pickles/3dreco_9.pkl"
  fpkl10 = "./data/data_19-2/pickles/3dreco_10.pkl"
  pickles = [fpkl1, fpkl2, fpkl3, fpkl4, fpkl5, fpkl6, fpkl7, fpkl8, fpkl9, fpkl10]

  #fpkl4 = "./data/data_1-12/3dreco_ccqe_mup.pkl"

  edepsim_file1 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.1.edep-sim.root"
  edepsim_file2 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.2.edep-sim.root"
  edepsim_file3 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.3.edep-sim.root"
  edepsim_file4 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.4.edep-sim.root"
  edepsim_file5 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.5.edep-sim.root"
  edepsim_file6 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.6.edep-sim.root"
  edepsim_file7 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.7.edep-sim.root"
  edepsim_file8 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.8.edep-sim.root"
  edepsim_file9 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.9.edep-sim.root"
  edepsim_file10 = "./data/data_19-2/edepsim/events-in-GRAIN_LAr_lv.10.edep-sim.root"
  edepsim_files = [edepsim_file1, edepsim_file2, edepsim_file3, edepsim_file4, edepsim_file5, edepsim_file6, edepsim_file7, edepsim_file8, edepsim_file9, edepsim_file10]
  #edepsim_file = "./data/data_1-12/events-in-GRAIN_LAr_lv.999.edep-sim.root"
  geom = load_geometry(geometryPath, defs)

  centers_all_ev, amps_all_ev, recodata_all_ev, tot_events = AllCentersAllEvents(selectedEvents, pickles)

  all_max_amps = []
  all_amps = []
  for i, amps in enumerate(amps_all_ev):
    if len(amps) != 0:
      for amp in amps:
  #     fig = plt.figure()
  #     plt.hist(amps, 100)
  #     plt.xlabel("score")
  #     plt.ylabel("n entries")
  #     plt.title(f"Photon score in event {tot_events[i]}")
  # plt.show()
        all_amps.append(amp)
      # max_amp = max(amps)
      # all_max_amps.append(max_amp)
  # print("mean max: ", np.average(all_max_amps))
  
  plt.hist(all_amps, 200)
  plt.xlabel("score")
  plt.ylabel("n entries")
  plt.title("Photon score in all events")
  #plt.show()

  #************************MC TRUTH**************************
  all_true_vertices, all_true_directions, ev_numbers = ExtractTrueParameters(edepsim_files, selectedEvents)
  # print("---------------------MC TRUTH-------------------------")
  # print("TRUE vertex: ", all_true_vertices)
  # print("TRUE directions: ", all_true_directions)
  #**********************************************************

  ### VERTICES
  twoPlanes2Tracks_reco_vertices = []
  twoPlanes2Tracks_true_vertices = []
  twoPlanes2Tracks_true_events = []
  onePlane2Tracks_reco_vertices = []
  onePlane2Tracks_true_vertices = []
  onePlane2Tracks_true_events = []

  ### DIRECTIONS
  all_thetas = []
  all_angles = []
  selected_true_directions = []

  cluster_countsZY = []
  cluster_countsZX = []

  onePlane2Tracks = 0
  onePlane0Tracks = 0
  onePlane1Tracks = 0
  twoPlanes2Tracks = 0
  twoPlanes1Tracks = 0
  twoPlanes0Tracks = 0
  twoZX_zeroZY = 0
  twoZX_oneZY = 0
  twoZY_zeroZX = 0
  twoZY_oneZX = 0
  n_reco = 0
  n_angle = 0
  if_angle = 0

  all_distances_all_evZY = []
  all_distances_all_evZX = []
  final_distZY = []
  final_distZX = []
  final_all_dist = []


  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(tot_events)):
    if len(centers_all_ev[i]) != 0:
      all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    print("*******************NUOVO EVENTO***********************")
    print("evento: ", tot_events[i])

    #**************************ACCUMULATOR***************************
    if len(all_clusters_in_ev) != 0:
      all_lpcs = np.concatenate([cluster.LPCs[0] for cluster in all_clusters_in_ev])
      points = list(zip(all_lpcs[:,0],all_lpcs[:,1],all_lpcs[:,2]))
      pointsZY = list(zip(all_lpcs[:,2],all_lpcs[:,1]))
      pointsZX = list(zip(all_lpcs[:,2],all_lpcs[:,0]))
    
      accumulatorZY, rhos, thetas = HoughTransform(pointsZY)
      accumulatorZX, _, _ = HoughTransform(pointsZX)

      local_max_indicesZY, rho_thetas_maxZY = FindLocalMaxima(accumulatorZY, all_lpcs)
      local_max_indicesZX, rho_thetas_maxZX = FindLocalMaxima(accumulatorZX, all_lpcs)
      # print("indices_local_max (rows, columns) ZY: ", local_max_indicesZY)
      # print("len: ", len(local_max_indicesZY))
      # print("indices_local_max (rows, columns) ZX: ", local_max_indicesZX)
      # print("rhos thetas max ZY: ", rho_thetas_maxZY)
      # print("rhos thetas max ZX: ", rho_thetas_maxZX)


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

      all_collinear_pointsZY, all_distancesZY = FindClosestToLinePointsZY(points, rho_thetas_maxZY)
      all_distances_all_evZY.append(all_distancesZY)
      all_collinear_pointsZX, all_distancesZX = FindClosestToLinePointsZX(points, rho_thetas_maxZX)
      all_distances_all_evZX.append(all_distancesZX)

    cluster_countsZY.append(len(all_collinear_pointsZY))
    cluster_countsZX.append(len(all_collinear_pointsZX))

    """condition for two tracks in both planes"""
    if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 2:
      twoPlanes2Tracks += 1
      #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
        #twoPlanes2Tracks += 1
      twoPlanes2Tracks_true_vertices.append(all_true_vertices[i])
      #twoPlanes2Tracks_true_directions.append(all_true_directions[i])
      twoPlanes2Tracks_true_events.append(ev_numbers[i])
      print("---------------------------RECO-----------------------------")
      reco_vertex = GetRecoVertex(all_collinear_pointsZX)
      print("RECO vertex 2planes2tracks: ", reco_vertex)
      twoPlanes2Tracks_reco_vertices.append(reco_vertex)

    """condition for two tracks in plane zx or zy: true and reco"""
    if len(all_collinear_pointsZX) == 2 and (len(all_collinear_pointsZY) == 0 or len(all_collinear_pointsZY) == 1): 
      #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
        onePlane2Tracks += 1
        onePlane2Tracks_true_vertices.append(all_true_vertices[i])
        #onePlane2Tracks_true_directions.append(all_true_directions[i])
        onePlane2Tracks_true_events.append(ev_numbers[i])
        reco_vertex = GetRecoVertex(all_collinear_pointsZX)
        print("---------------------------RECO-----------------------------")
        print("RECO vertex 1plane2tracks: ", reco_vertex)
        onePlane2Tracks_reco_vertices.append(reco_vertex)
    if len(all_collinear_pointsZY) == 2 and (len(all_collinear_pointsZX) == 0 or len(all_collinear_pointsZX) == 1):
      #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
      onePlane2Tracks += 1
      onePlane2Tracks_true_vertices.append(all_true_vertices[i])
      #onePlane2Tracks_true_directions.append(all_true_directions[i])
      onePlane2Tracks_true_events.append(ev_numbers[i])
      reco_vertex = GetRecoVertex(all_collinear_pointsZY)
      print("---------------------------RECO-----------------------------")
      print("RECO vertex 1plane2tracks: ", reco_vertex)
      onePlane2Tracks_reco_vertices.append(reco_vertex)

    """RECO DIRECTION"""
    all_reco_directions = []
    if len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 1:
      n_reco+=1
      reco_directions = GetRecoDirections(reco_vertex, all_collinear_pointsZY)
      print("RECODIR: ", reco_directions)
      #selected_true_directions.append(all_true_directions[i])
      theta = GetRecoAngle(reco_directions, all_true_directions[i])
      n_angle += 1
      print("RECO angles: ", theta)
      all_thetas.append(theta)
      #all_reco_directions.append(reco_directions)
    elif len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 2:
      reco_directions = GetRecoDirections(reco_vertex, all_collinear_pointsZX)
      #selected_true_directions.append(all_true_directions[i])
      theta = GetRecoAngle(reco_directions, all_true_directions[i])
      n_angle += 1
      print("RECO angles: ", theta)
      all_thetas.append(theta)
      #all_reco_directions.append(reco_directions)
    elif len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 2:
      reco_directions = GetRecoDirections(reco_vertex, all_collinear_pointsZY)
      #selected_true_directions.append(all_true_directions[i])
      theta = GetRecoAngle(reco_directions, all_true_directions[i])
      n_angle += 1
      print("RECO angles: ", theta)
      all_thetas.append(theta)
      #all_reco_directions.append(reco_directions)
    elif len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 1:
      reco_directions = GetRecoDirections(reco_vertex, all_collinear_pointsZY)
      #selected_true_directions.append(all_true_directions[i])
      theta = GetRecoAngle(reco_directions, all_true_directions[i])
      n_angle += 1
      print("RECO angles: ", theta)
      all_thetas.append(theta)
      #all_reco_directions.append(reco_directions)
      print("RECO directions: ", reco_directions)

      # """ANGLE"""
      # # print("test1: ", len(reco_directions))
      # # print("test2: ", len(all_collinear_pointsZY))
      # #if len(reco_directions) != 0:
      # #for i, reco_directions in enumerate(all_reco_directions):
      # theta = GetRecoAngle(reco_directions, all_true_directions[i])
      # n_angle += 1
      # print("RECO angles: ", theta)
      # all_thetas.append(theta)

        #***********************************2D PLOT ZY***************************    
        # fig3 = plt.figure()
        # ax = fig3.add_subplot()
        # for cluster in all_clusters_in_ev:
        #   single_curve = np.asarray(cluster.LPCs[0])
        #   plt.scatter(single_curve[:,2], single_curve[:,1], color = 'red')#allLPCpoints
      
        #i collinear points sono organizzati come x,y,z; li disegno nel piano z-y
        # all_collinear_pointsZY[0] = np.asarray(all_collinear_pointsZY[0])
        # plt.scatter(all_collinear_pointsZY[0][:,2], all_collinear_pointsZY[0][:,1], color = 'green')
        # all_collinear_pointsZY[1] = np.asarray(all_collinear_pointsZY[1])
        # plt.scatter(all_collinear_pointsZY[1][:,2], all_collinear_pointsZY[1][:,1], color = 'blue')
        # plt.ylim(-700,700)
        # plt.xlim(-200,200)
        # plt.gca().set_aspect('equal', adjustable='box')

        # """HOUGH TRANSFORM LINES"""
        # for rho,theta in rho_thetas_maxZY:
        #   y = np.linspace(-400, 400, 10000)
        #   z = -(np.cos(theta)/np.sin(theta))*y + rho/np.sin(theta)
        #   plt.plot(z, y)

        # for collinear_points in all_collinear_pointsZY:
        #   collinear_points = np.asarray(collinear_points)
        #   slope, intercept = Fit(collinear_points[:,2], collinear_points[:,1])
        #   plt.plot(collinear_points[:,2], slope*collinear_points[:,2] + intercept, color = 'red')
    
        # plt.grid()
        # plt.title("z-y plane")
        # plt.xlabel("z (mm)")
        # plt.ylabel("y (mm)")
        #*********************************************************************

        #***********************************2D PLOT ZX***************************
    # if len(all_collinear_pointsZX) == 0 and (len(all_collinear_pointsZY) == 2 or len(all_collinear_pointsZY) == 1):
    #   onePlane0Tracks += 1
    # if len(all_collinear_pointsZY) == 0 and (len(all_collinear_pointsZX) == 2 or len(all_collinear_pointsZX) == 1):
    #   onePlane0Tracks += 1

    if len(all_collinear_pointsZX) == 0 and len(all_collinear_pointsZY) == 0:
      twoPlanes0Tracks += 1
        
    if (len(all_collinear_pointsZX) == 1 and len(all_collinear_pointsZY) == 0) or (len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 0):
      onePlane1Tracks += 1

    if len(all_collinear_pointsZX) == 1 and len(all_collinear_pointsZY) == 1:
      twoPlanes1Tracks += 1

    if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 0:
      twoZX_zeroZY += 1

    if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 1:
      twoZX_oneZY += 1

    if len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 0:
      twoZY_zeroZX += 1

    if len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 1:
      twoZY_oneZX += 1

    #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):

  

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
        #   x = np.linspace(-400, 400, 10000)
        #   z = -(np.cos(theta)/np.sin(theta))*x + rho/np.sin(theta)
        #   plt.plot(z, x)

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
        # plt.title(tot_events[i])
        # plt.legend()
        #************************************************************

        #plt.show()

  print("tot events: ", len(tot_events))
  print("events: ", tot_events)
  #print("1 piano 0 tracce: ", onePlane0Tracks)
  print("2 piani 0 tracce: ", twoPlanes0Tracks)
  print("1 piano 1 traccia (e nell'altro 0): ", onePlane1Tracks)
  print("1 piano 2 tracce (l'altro o 0 o 1): ", onePlane2Tracks)
  print("2 piani 1 traccia: ", twoPlanes1Tracks)
  print("2 piani 2 tracce: ", twoPlanes2Tracks)
  print("2 tracce su ZX e 0 su ZY", twoZX_zeroZY)
  print("2 tracce su ZX e 1 su ZY", twoZX_oneZY)
  print("2 tracce su ZY e 0 su ZX", twoZY_zeroZX)
  print("2 tracce su ZY e 1 su ZX", twoZY_oneZX)
  print("sono dentro all'if: ", n_reco)
  print("NUMERO RECO ANGLE: ", n_angle)

  for theta_arr in all_thetas:
    for angle in theta_arr:
      all_angles.append(angle)
  print("NUMERO TOT angles: ", len(all_angles))

  for i in range(len(onePlane2Tracks_reco_vertices)):
    twoPlanes2Tracks_true_vertices.append(onePlane2Tracks_true_vertices[i])
    twoPlanes2Tracks_reco_vertices.append(onePlane2Tracks_reco_vertices[i])
  print("ev 2 tracce almeno in un piano: ", len(twoPlanes2Tracks_reco_vertices))

  # print("quanti cluster ZX: ", cluster_countsZX)
  # print("quanti cluster ZY: ", cluster_countsZY)

  ####VERTEX COORD########
  coord = ["X","Y","Z"]
  for i, c in enumerate(coord):
    print(f"---RECO VERTEX {c}---")
    diff_vertices = VertexCoordinatesHisto(twoPlanes2Tracks_true_vertices, twoPlanes2Tracks_reco_vertices, i)
    fig = plt.figure()
    plt.xlabel(f"{c} reco - {c} true")
    plt.ylabel("n entries")
    n, bins, patches = plt.hist(diff_vertices, 50, (-100,100), histtype='step')
    
    #fit the histo
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(n))
    popt, pcov = scp.optimize.curve_fit(gauss, x, n, p0 = [10, 0, 5])
    amp, mu, sigma = popt
    amp_err, mu_err, sigma_err = np.diagonal(pcov)
    chi_square, p_value = scp.stats.chisquare(n)
    print("vertex coord chi square: ", chi_square, p_value)
    print(f"{c} mu + mu_err:", popt[1], pcov[1][1])
    print(f"{c} std + std_err: ", popt[2], pcov[2][2])
    plt.plot(x, gauss(x,*popt), color = 'red')
    
    fit_info = f'Entries: {len(diff_vertices)}\nConst: {amp:.2f}±{amp_err:.2f}\nMean: {mu:.2f}±{mu_err:.2f}\nSigma: {sigma:.2f}±{sigma_err:.2f}'
    # plt.text(0.95, 0.95, f"      Vertex {c}     ",
    #      ha='right', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
    plt.text(0.95, 0.95, fit_info,
         ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
    #Mean: {np.mean(diff_vertices):.2f}\nStd Dev: {np.std(diff_vertices):.2f}\n$\chi^2$/ndf: {chi_square:.2f}/{len(bins) - 1}\nProb: {p_value:.4f}\n

    plt.title(f"{c} - {c}mc")
  ############################
  

  ########DISTANCE VERTEX##########
  print("------RECO DISTANCE------")
  fig5 = plt.figure()
  distance_vertices = DistanceRecoTrueVertex(twoPlanes2Tracks_true_vertices, twoPlanes2Tracks_reco_vertices)
  plt.xlabel(f"distance reco - true vertex")
  plt.ylabel("n entries")
  n3, bins3, _ = plt.hist(distance_vertices, 50, (0,200), histtype='step')
  
  #fit the histo
  # xmin3, xmax3 = plt.xlim()
  # x3 = np.linspace(xmin3, xmax3, len(n3))
  # popt3, pcov3 = scp.optimize.curve_fit(gauss, x3, n3, p0 = [10, 20, 10])
  # amp3, mu3, sigma3 = popt3
  # amp3_err, mu3_err, sigma3_err = np.diagonal(pcov3)
  # chi_square3, p_value3 = scp.stats.chisquare(n3)
  # print("distance chi square: ", chi_square3, p_value3)
  # print(f"mu + mu_err:", popt3[1], pcov3[1][1])
  # print(f"std + std_err: ", popt3[2], pcov3[2][2])
  # plt.plot(x3, gauss(x3,*popt), color = 'red')
  
  # fit_info = f'Entries: {len(distance_vertices)}\nMean: {np.mean(distance_vertices):.2f}\nStd Dev: {np.std(distance_vertices):.2f}'#\n$\chi^2$/ndf: {chi_square3:.2f}/{len(bins3) - 1}\nProb: {p_value3:.2f}\nConstant: {amp3:.2f}±{amp3_err:.2f}\nMean: {mu3:.2f}±{mu3_err:.2f}\nSigma: {sigma3:.2f}±{sigma3_err:.2f}'
  # plt.text(0.8035, 0.99, "Vertex distance",
  #       ha='left', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
  # plt.text(0.9, 0.95, fit_info,
        #ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))

  plt.title(f"distance reco - mc")
  ####################################

  ############ANGLE###################
  print("---RECO ANGLE---")
  fig6 = plt.figure()
  n2, bins2, _ = plt.hist(all_angles, 50, (-90,90), histtype='step')
  plt.xlabel("angle between reco and MC track ($^\circ$)")
  plt.ylabel("n entries")
  #fit the histo
  xmin2, xmax2 = plt.xlim()
  x2 = np.linspace(-75, 75, len(n2))
  #y = n
  popt2, pcov2 = scp.optimize.curve_fit(gauss, x2, n2, p0 = [60, 0, 5])
  amp2, mu2, sigma2 = popt2
  amp2_err, mu2_err, sigma2_err = np.diagonal(pcov2)
  chi_square2, p_value2 = scp.stats.chisquare(n2)
  print("angle chi square: ", chi_square2, p_value2)
  print("angle mu + mu_err:", popt2[1], pcov2[1][1])
  print("angle std + std_err: ", popt2[2], pcov2[2][2])
  plt.plot(x2, gauss(x2,*popt2), color = 'red')

  fit_info = f'Entries: {len(all_angles)}\nConst: {amp2:.2f}±{amp2_err:.2f}\nMean: {mu2:.2f}±{mu2_err:.2f}\nSigma: {sigma2:.2f}±{sigma2_err:.2f}'
  # plt.text(0.95, 0.95, "     angle      ",
  #       ha='right', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
  plt.text(0.95, 0.95, fit_info,
        ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
  #Mean: {np.mean(all_angles):.2f}\nStd Dev: {np.std(all_angles):.2f}\n$\chi^2$/ndf: {chi_square2:.2f}/{len(bins2) - 1}\nProb: {p_value2:.4f}

  plt.title("Angle")
  ####################################

  for all_dist in all_distances_all_evZY:
    for dist in all_dist:
      final_distZY.append(dist)

  for all_dist in all_distances_all_evZX:
    for dist in all_dist:
      final_distZX.append(dist)

  fig9 = plt.figure()
  plt.hist(final_distZY, 100, histtype='step')
  plt.xlabel("distance lpc point from the closest hough line (mm)")
  plt.ylabel("n entries")
  plt.title("ZY plane")

  fig10 = plt.figure()
  plt.hist(final_distZX, 100, histtype='step')
  plt.xlabel("distance lpc point from the closest hough line (mm)")
  plt.ylabel("n entries")
  plt.title("ZX plane")

  for dist in final_distZX:
    final_distZY.append(dist)

  fig11 = plt.figure()
  plt.hist(final_distZY, 100, histtype='step')
  plt.xlabel("distance lpc point from the closest hough line (mm)")
  plt.ylabel("n entries")
  plt.title("both planes")

  plt.show()

