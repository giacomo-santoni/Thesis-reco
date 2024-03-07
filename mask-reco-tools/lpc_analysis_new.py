import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy as scp
import skimage as sk
import kneed as kn
from sklearn.neighbors import NearestNeighbors
import sys
from skspatial.objects import Line, Points

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
  # all_eps = []
  # # Calculate NN
  # nearest_neighbors = NearestNeighbors(n_neighbors=6)
  # neighbors = nearest_neighbors.fit(centers)
  # distances, indices = neighbors.kneighbors(centers)
  # distances = np.sort(distances, axis=0)
  # # Get distances
  # distances = distances[:,1]
  # i = np.arange(len(distances))
  # # sns.lineplot(x = i, y = distances)
  # # plt.xlabel("Points")
  # # plt.ylabel("Distance")
  # #plt.show()
  # kneedle = kn.KneeLocator(x = range(1, len(distances)+1), y = distances, S = 1.0, 
  #                   curve = "concave", direction = "increasing", online=True)
  # # get the estimate of knee point
  # epsilon = kneedle.knee_y
  # print("EPSILON: ", epsilon)
  # kneedle.plot_knee()
  # all_eps.append(epsilon)
  # #plt.show()

  db_algo = DBSCAN(eps = 36, min_samples = all_parameters["min_points"])
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
      rho = coord1 * np.cos(theta) + coord2 * np.sin(theta)
      rho_index = np.argmin(np.abs(rhos - rho))
      accumulator[rho_index, theta_index] += 1
  return accumulator, rhos, thetas

def FindLocalMaxima(accumulator,lpcs):
  thr = len(lpcs)*0.35
  local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=7, threshold_abs = thr, exclude_border=False)

  if len(local_max_indices) > 2:
    local_max_indices = local_max_indices[:2]
  
  rho_thetas_max = []
  for i in local_max_indices:
    theta_max = thetas[i[1]]
    #print("THETA MAX: ", theta_max)
    rho_max = rhos[i[0]]
    rho_thetas_max.append((rho_max, theta_max))
  return local_max_indices, rho_thetas_max

def HoughLinesParams(rho_thetas_maxZY, rho_thetas_maxZX):
  all_slopesZY = []
  all_interceptsZY = []
  print("NR RHO THETA ZY: ",rho_thetas_maxZY)
  for rho,theta in rho_thetas_maxZY:
    # theta = rho_thetas_maxZY[1]
    # rho = rho_thetas_maxZY[0]
    if np.sin(theta) != 0:
      mZY = - (np.cos(theta)/np.sin(theta))
      qZY = rho/(np.sin(theta))
      all_slopesZY.append(mZY)
      all_interceptsZY.append(qZY)

  print("NR RHO THETA ZX: ",rho_thetas_maxZX)
  all_slopesZX = []
  all_interceptsZX = []
  for rho,theta in rho_thetas_maxZX:
    # theta = rho_thetas_maxZX[1]
    # rho = rho_thetas_maxZX[0]
    if np.sin(theta) != 0:
      mZX = - (np.cos(theta)/np.sin(theta))
      qZX = rho/(np.sin(theta))
      all_slopesZX.append(mZX)
      all_interceptsZX.append(qZX)
  return all_slopesZY, all_interceptsZY, all_slopesZX, all_interceptsZX

def HoughLines3D(points, all_slopesZY, all_interceptsZY, all_slopesZX, all_interceptsZX):
  collinear_points = [[],[],[],[]]
  accumulator = [0,0,0,0]
  #q11
  Q11 = np.array((all_interceptsZX[0], all_interceptsZY[0], 0))
  P11 = np.array((all_slopesZX[0]*100 + all_interceptsZX[0], all_slopesZY[0]*100 + all_interceptsZY[0], 100))
  if len(all_slopesZY) == 2: 
    #q12
    Q12 = np.array((all_interceptsZX[0], all_interceptsZY[1], 0))
    P12 = np.array((all_slopesZX[0]*100 + all_interceptsZX[0], all_slopesZY[1]*100 + all_interceptsZY[1], 100))
  else: 
    Q12 = np.array((np.nan, np.nan, np.nan))
    P12 = np.array((np.nan, np.nan, np.nan))
  if len(all_slopesZX) == 2:
    #q21
    Q21 = np.array((all_interceptsZX[1], all_interceptsZY[0], 0))
    P21 = np.array((all_slopesZX[1]*100 + all_interceptsZX[1], all_slopesZY[0]*100 + all_interceptsZY[0], 100))
  else: 
    Q21 = np.array((np.nan, np.nan, np.nan))
    P21 = np.array((np.nan, np.nan, np.nan))
  if len(all_slopesZY) == 2 and len(all_slopesZX) == 2:
    #q22
    Q22 = np.array((all_interceptsZX[1], all_interceptsZY[1], 0))
    P22 = np.array((all_slopesZX[1]*100 + all_interceptsZX[1], all_slopesZY[1]*100 + all_interceptsZY[1], 100))
  else: 
    Q22 = np.array((np.nan, np.nan, np.nan))
    P22 = np.array((np.nan, np.nan, np.nan))
  
  hough_lines_P = [P11,P12,P21,P22]
  hough_lines_Q = [Q11,Q12,Q21,Q22]

  for point in points:#points è una lista di punti x,y,z
    point = np.array(point)
    d11 = np.linalg.norm(np.cross((point - P11),(point - Q11)))/np.linalg.norm((P11 - Q11))
    d12 = np.linalg.norm(np.cross((point - P12),(point - Q12)))/np.linalg.norm((P12 - Q12))
    d21 = np.linalg.norm(np.cross((point - P21),(point - Q21)))/np.linalg.norm((P21 - Q21))
    d22 = np.linalg.norm(np.cross((point - P22),(point - Q22)))/np.linalg.norm((P22 - Q22))
    all_distances = [d11, d12, d21, d22]
    min_distance = np.nanmin(all_distances)
    min_distance_idx = np.nanargmin(all_distances)
    #print("all distances: ", all_distances)

    if min_distance > 100: 
      continue
    else:
      collinear_points[min_distance_idx].append(point)
      accumulator[min_distance_idx] += min_distance
    
  for i in range(4):
    collinear_points[i] = np.asarray(collinear_points[i])
    if len(collinear_points[i]):
      accumulator[i]/=len(collinear_points[i])
  #print("LISTE di COLL: ", collinear_points[0].shape)
  return collinear_points, hough_lines_P, hough_lines_Q, accumulator

def ExtractTrueParameters(file_list, events):
  vertices_coord = []
  ev_muon_directions = []
  ev_p_directions = []
  particles_directions = []
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
        muon_directions = []
        p_directions = []
        for particle in particles:
          momentum = particle.momentum
          particles_directions.append(momentum)
        ev_directions.append(particles_directions)
  #         if particle.PDGCode == 13: #muon 
  #           muon_momentum = particle.momentum
  #           muon_directions.append(muon_momentum)
  #         elif particle.PDGCode == 2212: #proton
  #           p_momentum = particle.momentum
  #           p_directions.append(p_momentum)
  #       ev_muon_directions.append(muon_directions)
  #       ev_p_directions.append(p_directions)
  # print("MUON direction len: ", len(ev_muon_directions))
  # print("PROT direction len: ", len(ev_p_directions))
  return vertices_coord, ev_directions, ev_numbers#, ev_p_directions

# def GetRecoVertex(all_collinear_points):
#   all_slopesZY = []
#   all_interceptsZY = []
#   all_slopesZX = []
#   all_interceptsZX = []
#   for collinear_points in all_collinear_points:
#     if len(collinear_points) != 0:
#       #ZY
#       collinear_points = np.asarray(collinear_points)
#       slopeZY, interceptZY = Fit(collinear_points[:,2], collinear_points[:,1])
#       all_slopesZY.append(slopeZY)
#       all_interceptsZY.append(interceptZY)
#       #ZX
#       slopeZX, interceptZX = Fit(collinear_points[:,2], collinear_points[:,0])
#       all_slopesZX.append(slopeZX)
#       all_interceptsZX.append(interceptZX)
    
#   if len(all_interceptsZY)>1 or len(all_interceptsZX)>1:
#     z_vertex1 = (all_interceptsZY[1] - all_interceptsZY[0])/(all_slopesZY[0] - all_slopesZY[1])
#     y_vertex = all_slopesZY[0]*z_vertex1 + all_interceptsZY[0]
#     #print("ZY vertex: ", z_vertex1, y_vertex)

#     z_vertex2 = (all_interceptsZX[1] - all_interceptsZX[0])/(all_slopesZX[0] - all_slopesZX[1])
#     x_vertex = all_slopesZX[0]*z_vertex2 + all_interceptsZX[0]
#     #print("ZX vertex: ", z_vertex2, x_vertex)

#     z_vertex = (z_vertex1 + z_vertex2)/2

#     x_vertex = slopeZX*z_vertex + interceptZX
#     reco_vertex = (x_vertex, y_vertex, z_vertex)
#   else: reco_vertex = "Vertex not found or not present"
#   return reco_vertex

# def GetRecoDirections(all_collinear_points):
#   all_reco_directions = []
#   for collinear_points in all_collinear_points: 
#     collinear_points = np.asarray(collinear_points)
#     slopeZY, _ = Fit(collinear_points[:,2], collinear_points[:,1])
#     slopeZX, _ = Fit(collinear_points[:,2], collinear_points[:,0])
#     reco_direction = (slopeZX, slopeZY, 1)
#     all_reco_directions.append(reco_direction) 
#   return all_reco_directions 

def GetRecoAngle(reco_direction, true_directions):
  # #print("TRUE DIR: ", true_direction[0])
  # all_selected_thetas = []
  # for t_dir in true_directions:#dato che ho solo una traccia vera per ogni evento (visto che considero solo il muone) 
  #   thetas_one_track = []
  #   #cosines_one_track = []
  #   for r_dir in reco_directions:#se ho 2 tracce, calcolo il prodotto scalare tra ognuna e il muone vero, e vedo il coseno massimo
  #     cosine = (np.dot(t_dir, r_dir))/(np.linalg.norm(t_dir)*np.linalg.norm(r_dir))
  #     theta = np.rad2deg(np.arccos(cosine))
  #     print("THETA: ", theta)
  #     if theta > 90:
  #       alpha = - (180 - theta)
  #       thetas_one_track.append(alpha)
  #     elif theta <= 90:
  #       thetas_one_track.append(theta)
  #   sel_theta = [theta for theta in thetas_one_track if abs(theta) == (np.min(np.abs(thetas_one_track)))][0]
  #   all_selected_thetas.append(sel_theta)

  all_angles = []
  min_thetas = []
  # for r_dir in reco_directions:
  #   print("RECO DIRECTIONS: ", r_dir)
  thetas_one_track = []
  for t_dir in true_directions:
    # print("RECO DIRECTIONS: ", reco_direction)
    # print("TRUE DIRECTIONS: ", t_dir)
    cosine = np.dot(reco_direction,t_dir)/(np.linalg.norm(reco_direction)*np.linalg.norm(t_dir))
    theta = np.arccos(cosine)
    if np.rad2deg(theta) > 90:
      alpha = - (180 - np.rad2deg(theta))
      thetas_one_track.append(np.deg2rad(alpha))
    else: 
      thetas_one_track.append(theta)
  sel_theta = np.min(np.abs(np.asarray(thetas_one_track)))
  all_angles.append(np.rad2deg(sel_theta))
  return all_angles

# def is_point_in_array(point, array):
#     return any(np.all(point == i) for i in array)

# # def VertexCoordinatesHisto(true_vertices, reco_vertices, coord):
# #   diff_vertices = []
# #   n = 0
# #   p = 0
# #   for i, true_vertex in enumerate(true_vertices):
# #     p += 1
# #     if isinstance(reco_vertices[i], tuple):
# #       n += 1
# #       diff = (np.asarray(reco_vertices[i][coord]) - np.asarray(true_vertex[coord]))
# #       diff_vertices.append(diff)
# #   print("dopo if array: ", n)
# #   print("prima if array: ", p)
# #   return diff_vertices

# # def DistanceRecoTrueVertex(true_vertices, reco_vertices):
# #   distance_vertices = []
# #   for i, true_vertex in enumerate(true_vertices):
# #     if isinstance(reco_vertices[i], tuple):
# #       dist = np.linalg.norm(np.asarray(reco_vertices[i]) - np.asarray(true_vertex))
# #       distance_vertices.append(dist)
# #   return distance_vertices

# # def gauss(x,amp,mu,sigma):
# #   return amp*np.exp(-((x-mu)**2)/(2*sigma**2))



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

  selectedEvents = EventNumberList(id_list)
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
  print("TOT_EVENTS: ", tot_events)

  # all_max_amps = []
  # all_amps = []
  # for i, amps in enumerate(amps_all_ev):
  #   if len(amps) != 0:
  #     for amp in amps:
  # #     fig = plt.figure()
  # #     plt.hist(amps, 100)
  # #     plt.xlabel("score")
  # #     plt.ylabel("n entries")
  # #     plt.title(f"Photon score in event {tot_events[i]}")
  # # plt.show()
  #       all_amps.append(amp)
  #     # max_amp = max(amps)
  #     # all_max_amps.append(max_amp)
  # # print("mean max: ", np.average(all_max_amps))
  
  # plt.hist(all_amps, 200)
  # plt.xlabel("score")
  # plt.ylabel("n entries")
  # plt.title("Photon score in all events")
  #plt.show()

  #************************MC TRUTH**************************
  all_true_vertices, all_true_directions, ev_numbers = ExtractTrueParameters(edepsim_files, selectedEvents)
  print("EV NUMBERS: ", ev_numbers)
  # print("---------------------MC TRUTH-------------------------")
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
  all_distances = []


  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(tot_events)):
    if len(centers_all_ev[i]) != 0:
      all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    print("*******************NUOVO EVENTO***********************")
    print("evento: ", tot_events[i])
    #print("MUON DIR: ", all_true_muon_directions[i])

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

      #ZY

      #fig3 = plt.figure()

      for k in local_max_indicesZY:
        local_max_values = accumulatorZY[k[0],k[1]]
        # print("max_values: ", local_max_values)
        # print("theta: ", np.rad2deg(thetas[k[1]]))
        # print("rho: ", rhos[k[0]])
      
      # plt.imshow(accumulatorZY, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
      # plt.xlabel('theta')
      # plt.ylabel('rho')
      # plt.colorbar()

      # for s in local_max_indicesZY:
      #   plt.plot(np.rad2deg(thetas[s[1]]),rhos[s[0]], 'ro')

      #ZX
      #fig4 = plt.figure()

      for k in local_max_indicesZX:
        local_max_values = accumulatorZX[k[0],k[1]]
        # print("max_values: ", local_max_values)
        # print("theta: ", np.rad2deg(thetas[k[1]]))
        # print("rho: ", rhos[k[0]])
      
      # plt.imshow(accumulatorZX, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
      # plt.xlabel('theta')
      # plt.ylabel('rho')
      # plt.colorbar()

      # for s in local_max_indicesZX:
      #   plt.plot(np.rad2deg(thetas[s[1]]),rhos[s[0]], 'ro')
      #******************************************************************
      
      all_slopesZY, all_interceptsZY, all_slopesZX, all_interceptsZX = HoughLinesParams(rho_thetas_maxZY, rho_thetas_maxZX)

      # print("slopes ZY: ", all_slopesZY)
      # print("intercepts ZY: ", all_interceptsZY)
      # print("slopes ZX: ", all_slopesZX)
      # print("intercepts ZX: ", all_interceptsZX)

      collinear_points, hough_lines_P, hough_lines_Q, distances = HoughLines3D(points, all_slopesZY, all_interceptsZY, all_slopesZX, all_interceptsZX)
      
      #fig = plt.figure()
      #ax = fig.add_subplot(projection="3d")

      colors = ["red", "blue", "green", "magenta"]
      points = np.asarray(points)
      #ax.scatter(points[:,0], points[:,1],points[:,2],color="orange",alpha=0.4)
      #ax.scatter(centers_all_ev[i][:,0],centers_all_ev[i][:,1],centers_all_ev[i][:,2],color = "black",alpha=0.1)
      for j in range(4):
        #ax.plot([hough_lines_P[j][0], hough_lines_Q[j][0]],[hough_lines_P[j][1],hough_lines_Q[j][1]],zs = [hough_lines_P[j][2],hough_lines_Q[j][2]], color=colors[j])
        if len(collinear_points[j]) > 3:
          #ax.scatter(collinear_points[j][:,0],collinear_points[j][:,1],collinear_points[j][:,2],color=colors[j])

          line_fit = Line.best_fit(Points(collinear_points[j]))
          #ax.quiver(*line_fit.point,*line_fit.direction, length=100, color=colors[j])

          reco_direction = np.asarray((line_fit.direction[0],line_fit.direction[1],line_fit.direction[2]))
          thetas = GetRecoAngle(reco_direction, all_true_directions[i])
          all_thetas.append(thetas)

          #print("POINT FIT: ", line_fit.point)
          distance = np.linalg.norm(np.cross((all_true_vertices[i] - line_fit.point),(all_true_vertices[i] - (line_fit.point + line_fit.direction))))/np.linalg.norm(line_fit.direction)
          #print("DIST: ", distance)
          all_distances.append(distance)

  for theta in all_thetas: 
    all_angles.append(theta[0])
  print("ANGOLI: ", len(all_angles))
  fig5 = plt.figure()
  plt.hist(all_angles,25,histtype="step")
  plt.xlabel("Angle reco - MC track [$^\circ$]")
  plt.ylabel("n entries")
  print("MEAN ANGLE: ", np.average(all_angles))
  print("STD ANGLE: ", np.std(all_angles))

  fig6 = plt.figure()
  plt.hist(all_distances,40,range=(0,400),histtype="step")
  plt.xlabel("Closest distance true vertex - reco track [mm]")
  plt.ylabel("n entries")
  print("MEAN DISTANCE: ", np.average(all_distances))
  print("STD DISTANCE: ", np.std(all_distances))

  plt.show()

    

      # fig2 = plt.figure()
      # ax = fig2.add_subplot()
      # for cluster in all_clusters_in_ev:
      #   single_curve = np.asarray(cluster.LPCs[0])
      #   ax.scatter(single_curve[:,2], single_curve[:,1], color = 'red')#allLPCpoints
      # # #for collinear_points in all_collinear_pt:
      # # all_collinear_points[0] = np.asarray(all_collinear_points[0])
      # # ax.scatter(all_collinear_points[0][:,0], all_collinear_points[0][:,1], color = 'green')
      # # if len(all_collinear_points)>1:
      # #   all_collinear_points[1] = np.asarray(all_collinear_points[1])
      # #   plt.scatter(all_collinear_points[1][:,0], all_collinear_points[1][:,1], color = 'blue')
      # plt.ylim(-700,700)
      # plt.xlim(-200,200)
      # plt.gca().set_aspect('equal', adjustable='box')

      # for rho,theta in rho_thetas_maxZY:
      #   z = np.linspace(-400, 400, 10000)
      #   y = -(np.cos(theta)/np.sin(theta))*z + rho/np.sin(theta)
      #   plt.plot(z, y)
      # plt.show()


  #     all_collinear_pointsZY, all_distancesZY = FindClosestToLinePointsZY(points, rho_thetas_maxZY)
  #     all_distances_all_evZY.append(all_distancesZY)
  #     all_collinear_pointsZX, all_distancesZX = FindClosestToLinePointsZX(points, rho_thetas_maxZX)
  #     all_distances_all_evZX.append(all_distancesZX)
    
  #   print("COLL ZX: ", all_collinear_pointsZX)
  #   print("COLL ZY: ", all_collinear_pointsZY)

  #   cluster_countsZY.append(len(all_collinear_pointsZY))
  #   cluster_countsZX.append(len(all_collinear_pointsZX))


  #   """RECO DIRECTION"""
  #   all_reco_directions = []
  #   # if len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 1:
  #   #   n_reco+=1
  #   #   reco_directions = GetRecoDirections(all_collinear_pointsZY)
  #   #   print("RECODIR: ", reco_directions)
  #   #   #selected_true_directions.append(all_true_directions[i])
  #   #   theta = GetRecoAngle(reco_directions, all_true_muon_directions[i])
  #   #   n_angle += 1
  #   #   print("RECO angles: ", theta)
  #   #   all_thetas.append(theta)
  #   #   #all_reco_directions.append(reco_directions)
  #   # elif len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 2:
  #   #   reco_directions = GetRecoDirections(all_collinear_pointsZX)
  #   #   #selected_true_directions.append(all_true_directions[i])
  #   #   theta = GetRecoAngle(reco_directions, all_true_muon_directions[i])
  #   #   n_angle += 1
  #   #   print("RECO angles: ", theta)
  #   #   all_thetas.append(theta)
  #   #   #all_reco_directions.append(reco_directions)
  #   if len(all_collinear_pointsZY) == 2 and len(all_collinear_pointsZX) == 2:
  #     # common_collinear1 = []
  #     # for point in all_collinear_pointsZY[0]:
  #     #   print("coll points zx[0]: ", all_collinear_pointsZX[0])
  #     #   if point in all_collinear_pointsZX[0]:
  #     #     print("PUNTO: ", point)
  #     #     common_collinear1.append(point)
  #     #   elif point in all_collinear_pointsZX[1]:
  #     #     print("PUNTO 2: ", point)
  #     #     common_collinear1.append(point)
  #     # #print("COMMON 1: ", common_collinear1)

  #     # common_collinear2 = []
  #     # for point in all_collinear_pointsZY[1]:
  #     #   if point in all_collinear_pointsZX[0]:
  #     #     common_collinear2.append(point)
  #     #   elif point in all_collinear_pointsZX[1]:
  #     #     common_collinear2.append(point)

  #     # common_collinear = [common_collinear1, common_collinear2]
  #     # print("COMMON: ", common_collinear)

  #     reco_directions = GetRecoDirections(all_collinear_pointsZX)
  #     #selected_true_directions.append(all_true_directions[i])
  #     theta = GetRecoAngle(reco_directions, all_true_muon_directions[i])
  #     n_angle += 1
  #     print("RECO angles: ", theta)
  #     all_thetas.append(theta)
  #     #all_reco_directions.append(reco_directions)
  #   elif len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 1:
  #     # common_collinear1 = []
  #     # for point in all_collinear_pointsZY[0]:
  #     #   print("coll points zx[0]: ", all_collinear_pointsZX[0])
  #     #   if point in all_collinear_pointsZX[0]:
  #     #     print("PUNTO: ", point)
  #     #     common_collinear1.append(point)
  
  #     # print("COMMON UNO: ", common_collinear1)

  #     reco_directions = GetRecoDirections(all_collinear_pointsZX)
  #     #selected_true_directions.append(all_true_directions[i])
  #     theta = GetRecoAngle(reco_directions, all_true_muon_directions[i])
  #     n_angle += 1
  #     print("RECO angles: ", theta)
  #     all_thetas.append(theta)
  #     #all_reco_directions.append(reco_directions)
  #     #print("RECO directions: ", reco_directions)
  #   #print("TUTTI GLI ANGOLI: ", all_thetas)


  #   """condition for two tracks in both planes"""
  #   if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 2:
  #     twoPlanes2Tracks += 1
  #     #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
  #       #twoPlanes2Tracks += 1
  #     twoPlanes2Tracks_true_vertices.append(all_true_vertices[i])
  #     #twoPlanes2Tracks_true_directions.append(all_true_directions[i])
  #     twoPlanes2Tracks_true_events.append(ev_numbers[i])
  #     print("---------------------------RECO-----------------------------")
  #     reco_vertex = GetRecoVertex(all_collinear_pointsZX)
  #     print("RECO vertex 2planes2tracks: ", reco_vertex)
  #     twoPlanes2Tracks_reco_vertices.append(reco_vertex)

  #   """condition for two tracks in plane zx or zy: true and reco"""
  #   if len(all_collinear_pointsZX) == 2 and (len(all_collinear_pointsZY) == 0 or len(all_collinear_pointsZY) == 1): 
  #     #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
  #       onePlane2Tracks += 1
  #       onePlane2Tracks_true_vertices.append(all_true_vertices[i])
  #       #onePlane2Tracks_true_directions.append(all_true_directions[i])
  #       onePlane2Tracks_true_events.append(ev_numbers[i])
  #       reco_vertex = GetRecoVertex(all_collinear_pointsZX)
  #       print("---------------------------RECO-----------------------------")
  #       print("RECO vertex 1plane2tracks: ", reco_vertex)
  #       onePlane2Tracks_reco_vertices.append(reco_vertex)
  #   if len(all_collinear_pointsZY) == 2 and (len(all_collinear_pointsZX) == 0 or len(all_collinear_pointsZX) == 1):
  #     #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
  #     onePlane2Tracks += 1
  #     onePlane2Tracks_true_vertices.append(all_true_vertices[i])
  #     #onePlane2Tracks_true_directions.append(all_true_directions[i])
  #     onePlane2Tracks_true_events.append(ev_numbers[i])
  #     reco_vertex = GetRecoVertex(all_collinear_pointsZY)
  #     print("---------------------------RECO-----------------------------")
  #     print("RECO vertex 1plane2tracks: ", reco_vertex)
  #     onePlane2Tracks_reco_vertices.append(reco_vertex)


  # print("tot events: ", len(tot_events))
  # print("events: ", tot_events)
  # #print("1 piano 0 tracce: ", onePlane0Tracks)
  # print("2 piani 0 tracce: ", twoPlanes0Tracks)
  # print("1 piano 1 traccia (e nell'altro 0): ", onePlane1Tracks)
  # print("1 piano 2 tracce (l'altro o 0 o 1): ", onePlane2Tracks)
  # print("2 piani 1 traccia: ", twoPlanes1Tracks)
  # print("2 piani 2 tracce: ", twoPlanes2Tracks)
  # print("2 tracce su ZX e 0 su ZY", twoZX_zeroZY)
  # print("2 tracce su ZX e 1 su ZY", twoZX_oneZY)
  # print("2 tracce su ZY e 0 su ZX", twoZY_zeroZX)
  # print("2 tracce su ZY e 1 su ZX", twoZY_oneZX)
  # print("sono dentro all'if: ", n_reco)
  # print("NUMERO RECO ANGLE: ", n_angle)

  # for angle in all_thetas: 
  #   all_angles.append(angle[0])
  # print("TUTTI GLI ANGOLI: ", all_angles)

  # for i in range(len(onePlane2Tracks_reco_vertices)):
  #   twoPlanes2Tracks_true_vertices.append(onePlane2Tracks_true_vertices[i])
  #   twoPlanes2Tracks_reco_vertices.append(onePlane2Tracks_reco_vertices[i])
  # print("ev 2 tracce almeno in un piano: ", len(twoPlanes2Tracks_reco_vertices))

  # # print("quanti cluster ZX: ", cluster_countsZX)
  # # print("quanti cluster ZY: ", cluster_countsZY)

  # # ####VERTEX COORD########
  # # coord = ["X","Y","Z"]
  # # for i, c in enumerate(coord):
  # #   print(f"---RECO VERTEX {c}---")
  # #   diff_vertices = VertexCoordinatesHisto(twoPlanes2Tracks_true_vertices, twoPlanes2Tracks_reco_vertices, i)
  # #   stand_dev = np.std(diff_vertices)
  # #   print(f"{c} std dev: ", stand_dev)
  # #   fig = plt.figure()
  # #   plt.xlabel(f"{c} reco - {c} true")
  # #   plt.ylabel("n entries")
  # #   n, bins, patches = plt.hist(diff_vertices, 20, (-100,100), histtype='step')
  # #   bin_centers = (bins[:-1] + bins[1:]) / 2

  # #   original_stdout = sys.stdout

  # #   with open(f'./{c} coordinate', 'a') as f:
  # #     sys.stdout = f 
  # #     for i, bin in enumerate(bin_centers):
  # #       print(bin, n[i],'\n')
  # #     sys.stdout = original_stdout 
  # #     f.close()
    
  # #   #fit the histo
  # #   xmin, xmax = plt.xlim()
  # #   x = np.linspace(xmin, xmax, len(n))
  # #   x_gauss = np.linspace(xmin, xmax, 50)
  # #   popt, pcov = scp.optimize.curve_fit(gauss, x, n, p0 = [65, 0, 30])
  # #   amp, mu, sigma = popt
  # #   amp_err, mu_err, sigma_err = np.diagonal(pcov)
  # #   chi_square, p_value = scp.stats.chisquare(n)
  # #   print("vertex coord chi square: ", chi_square, p_value)
  # #   print(f"{c} mu + mu_err:", popt[1], pcov[1][1])
  # #   print(f"{c} std + std_err: ", popt[2], pcov[2][2])
  # #   plt.plot(x_gauss, gauss(x_gauss,*popt), color = 'red')
    
  # #   fit_info = f'Entries: {len(diff_vertices)}\nConst: {amp:.2f}±{amp_err:.2f}\nMean: {mu:.2f}±{mu_err:.2f}\nSigma: {sigma:.2f}±{sigma_err:.2f}'
  # #   # plt.text(0.95, 0.95, f"      Vertex {c}     ",
  # #   #      ha='right', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
  # #   plt.text(0.95, 0.95, fit_info,
  # #        ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
  # #   #Mean: {np.mean(diff_vertices):.2f}\nStd Dev: {np.std(diff_vertices):.2f}\n$\chi^2$/ndf: {chi_square:.2f}/{len(bins) - 1}\nProb: {p_value:.4f}\n

  # #   plt.title(f"{c} - {c}mc")
  # # ############################
  

  # # ########DISTANCE VERTEX##########
  # # print("------RECO DISTANCE------")
  # # fig5 = plt.figure()
  # # distance_vertices = DistanceRecoTrueVertex(twoPlanes2Tracks_true_vertices, twoPlanes2Tracks_reco_vertices)
  # # plt.xlabel(f"distance reco - true vertex")
  # # plt.ylabel("n entries")
  # # n3, bins3, _ = plt.hist(distance_vertices, 20, (0,200), histtype='step')
  
  # # #fit the histo
  # # # xmin3, xmax3 = plt.xlim()
  # # # x3 = np.linspace(xmin3, xmax3, len(n3))
  # # # popt3, pcov3 = scp.optimize.curve_fit(gauss, x3, n3, p0 = [, 20, 10])
  # # # amp3, mu3, sigma3 = popt3
  # # # amp3_err, mu3_err, sigma3_err = np.diagonal(pcov3)
  # # # chi_square3, p_value3 = scp.stats.chisquare(n3)
  # # # print("distance chi square: ", chi_square3, p_value3)
  # # # print(f"mu + mu_err:", popt3[1], pcov3[1][1])
  # # # print(f"std + std_err: ", popt3[2], pcov3[2][2])
  # # # plt.plot(x3, gauss(x3,*popt), color = 'red')
  
  # # # fit_info = f'Entries: {len(distance_vertices)}\nMean: {np.mean(distance_vertices):.2f}\nStd Dev: {np.std(distance_vertices):.2f}'#\n$\chi^2$/ndf: {chi_square3:.2f}/{len(bins3) - 1}\nProb: {p_value3:.2f}\nConstant: {amp3:.2f}±{amp3_err:.2f}\nMean: {mu3:.2f}±{mu3_err:.2f}\nSigma: {sigma3:.2f}±{sigma3_err:.2f}'
  # # # plt.text(0.8035, 0.99, "Vertex distance",
  # # #       ha='left', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
  # # # plt.text(0.9, 0.95, fit_info,
  # #       #ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))

  # # plt.title(f"distance reco - mc")
  # # ####################################

  # ############ANGLE###################
  # print("---RECO ANGLE---")
  # #stand_dev_angle = np.std(all_angles)
  # #print("std dev angles: ", stand_dev_angle)
  # fig6 = plt.figure()
  # n2, bins2, _ = plt.hist(all_angles, 50, (-90,90), histtype='step')
  # plt.xlabel("angle between reco and MC track ($^\circ$)")
  # plt.ylabel("n entries")
  # bin_centers2 = (bins2[:-1] + bins2[1:]) / 2

  # original_stdout = sys.stdout

  # with open('./angle_muon', 'a') as f:
  #   sys.stdout = f 
  #   for i, bin in enumerate(bin_centers2):
  #     print(bin, n2[i],'\n')
  #   sys.stdout = original_stdout 
  #   f.close()

  # # #fit the histo
  # # xmin2, xmax2 = plt.xlim()
  # # x2 = np.linspace(-75, 75, len(n2))
  # # x_gauss2 = np.linspace(xmin2, xmax2, 50)
  # # #y = n
  # # popt2, pcov2 = scp.optimize.curve_fit(gauss, x2, n2, p0 = [170, 0, 30])
  # # amp2, mu2, sigma2 = popt2
  # # amp2_err, mu2_err, sigma2_err = np.diagonal(pcov2)
  # # chi_square2, p_value2 = scp.stats.chisquare(n2)
  # # print("angle chi square: ", chi_square2, p_value2)
  # # print("angle mu + mu_err:", popt2[1], pcov2[1][1])
  # # print("angle std + std_err: ", popt2[2], pcov2[2][2])
  # # plt.plot(x_gauss2, gauss(x_gauss2,*popt2), color = 'red')

  # # fit_info = f'Entries: {len(all_angles)}\nConst: {amp2:.2f}±{amp2_err:.2f}\nMean: {mu2:.2f}±{mu2_err:.2f}\nSigma: {sigma2:.2f}±{sigma2_err:.2f}'
  # # # plt.text(0.95, 0.95, "     angle      ",
  # # #       ha='right', va='top', transform=plt.gca().transAxes, weight='bold', bbox=dict(facecolor='white', edgecolor='black'))
  # # plt.text(0.95, 0.95, fit_info,
  # #       ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
  # # #Mean: {np.mean(all_angles):.2f}\nStd Dev: {np.std(all_angles):.2f}\n$\chi^2$/ndf: {chi_square2:.2f}/{len(bins2) - 1}\nProb: {p_value2:.4f}

  # # plt.title("Angle")
  # # ####################################

  # # for all_dist in all_distances_all_evZY:
  # #   for dist in all_dist:
  # #     final_distZY.append(dist)

  # # for all_dist in all_distances_all_evZX:
  # #   for dist in all_dist:
  # #     final_distZX.append(dist)

  # # fig9 = plt.figure()
  # # plt.hist(final_distZY, 100, histtype='step')
  # # plt.xlabel("distance lpc point from the closest hough line (mm)")
  # # plt.ylabel("n entries")
  # # plt.title("ZY plane")

  # # fig10 = plt.figure()
  # # plt.hist(final_distZX, 100, histtype='step')
  # # plt.xlabel("distance lpc point from the closest hough line (mm)")
  # # plt.ylabel("n entries")
  # # plt.title("ZX plane")

  # # for dist in final_distZX:
  # #   final_distZY.append(dist)

  # # fig11 = plt.figure()
  # # ymin1, ymax1 = plt.ylim()
  # # plt.hist(final_distZY, 100, histtype='step')
  # # plt.axvline(45, ymin= ymin1, ymax= ymax1, color = 'red')
  # # plt.xlabel("distance lpc point from the closest hough line (mm)", fontsize = 11)
  # # plt.ylabel("n entries", fontsize = 11)
  # # plt.title("both planes")

  # plt.show()

