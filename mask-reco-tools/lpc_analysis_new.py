import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import scipy as scp
import skimage as sk
from scipy.stats import norm

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
  print("len lpcs: ", len(lpcs))
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

    if closest_line == rho_thetas_max[0]:
      collinear_points1.append((coord1,coord2,coord3))
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))

  all_collinear_pointsZY.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0:
    all_collinear_pointsZY.append(collinear_points2)
  return all_collinear_pointsZY

def FindClosestToLinePointsZX(points, rho_thetas_max):
  all_collinear_pointsZX = []
  collinear_points1 = []
  collinear_points2 = []
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

    if closest_line == rho_thetas_max[0]:
      collinear_points1.append((coord1,coord2,coord3))
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))

  all_collinear_pointsZX.append(collinear_points1)
  if len(rho_thetas_max)>1 and len(collinear_points2) != 0:
    all_collinear_pointsZX.append(collinear_points2)
  return all_collinear_pointsZX

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

def VertexCoordinatesHisto(true_vertices, reco_vertices, coord):
  diff_vertices = []
  fit_pars = []
  for i, true_vertex in enumerate(true_vertices):
    if isinstance(true_vertex, np.ndarray):
      diff = (np.asarray(reco_vertices[i][coord]) - np.asarray(true_vertex[coord]))
      diff_vertices.append(diff)

  return diff_vertices#, fit_pars

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

  #************************MC TRUTH**************************
  all_true_vertices, all_true_directions, ev_numbers = ExtractTrueParameters(edepsim_files, selectedEvents)
  print("---------------------MC TRUTH-------------------------")
  # print("TRUE vertex: ", all_true_vertices)
  # print("TRUE directions: ", len(all_true_directions))
  #**********************************************************

  twoPlanes2Tracks_reco_vertices = []
  twoPlanes2Tracks_true_vertices = []
  twoPlanes2Tracks_true_directions = []
  twoPlanes2Tracks_true_events = []

  onePlane2Tracks_reco_vertices = []
  onePlane2Tracks_true_vertices = []
  onePlane2Tracks_true_directions = []
  onePlane2Tracks_true_events = []

  cluster_countsZY = []
  cluster_countsZX = []

  onePlane2Tracks = 0
  onePlane0Tracks = 0
  onePlane1Tracks = 0
  twoPlanes2Tracks = 0
  twoPlanes1Tracks = 0

  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  # for events_file in selectedEvents:
  #   print("events in file: ", events_file)
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

      all_collinear_pointsZY = FindClosestToLinePointsZY(points, rho_thetas_maxZY)
      all_collinear_pointsZX = FindClosestToLinePointsZX(points, rho_thetas_maxZX)

    cluster_countsZY.append(len(all_collinear_pointsZY))
    cluster_countsZX.append(len(all_collinear_pointsZX))

    """condition for two tracks in both planes"""
    if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 2:
      if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
        twoPlanes2Tracks += 1
        twoPlanes2Tracks_true_vertices.append(all_true_vertices[i])
        twoPlanes2Tracks_true_directions.append(all_true_directions[i])
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
        onePlane2Tracks_true_directions.append(all_true_directions[i])
        onePlane2Tracks_true_events.append(ev_numbers[i])
        reco_vertex = GetRecoVertex(all_collinear_pointsZX)
        print("---------------------------RECO-----------------------------")
        print("RECO vertex 1plane2tracks: ", reco_vertex)
        onePlane2Tracks_reco_vertices.append(reco_vertex)
    if len(all_collinear_pointsZY) == 2 and (len(all_collinear_pointsZX) == 0 or len(all_collinear_pointsZX) == 1):
      #if (all_collinear_pointsZY[0] != [] and all_collinear_pointsZX[0] != []) and (all_collinear_pointsZY[1] != [] and all_collinear_pointsZX[1] != []):
      onePlane2Tracks += 1
      onePlane2Tracks_true_vertices.append(all_true_vertices[i])
      onePlane2Tracks_true_directions.append(all_true_directions[i])
      onePlane2Tracks_true_events.append(ev_numbers[i])
      reco_vertex = GetRecoVertex(all_collinear_pointsZY)
      print("---------------------------RECO-----------------------------")
      print("RECO vertex 1plane2tracks: ", reco_vertex)
      onePlane2Tracks_reco_vertices.append(reco_vertex)


        # """RECO DIRECTION"""
        # all_reco_directions = GetRecoDirections(reco_vertex, all_collinear_pointsZY)
        # print("RECO directions: ", all_reco_directions)

        # """ANGLE"""
        # theta = GetRecoAngle(all_reco_directions, all_true_directions[i])
        # print("RECO angle: ", theta)

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
        
        # if (len(all_collinear_pointsZX) == 1 and len(all_collinear_pointsZY) == 0) or (len(all_collinear_pointsZY) == 1 and len(all_collinear_pointsZX) == 0):
        #   onePlane1Tracks += 1
      
        # if len(all_collinear_pointsZX) == 2 and (len(all_collinear_pointsZY) == 0 or len(all_collinear_pointsZY) == 1):
        #   onePlane2Tracks += 1
        # if len(all_collinear_pointsZY) == 2 and (len(all_collinear_pointsZX) == 0 or len(all_collinear_pointsZX) == 1):
        #   onePlane2Tracks += 1

        # if len(all_collinear_pointsZX) == 1 and len(all_collinear_pointsZY) == 1:
        #   twoPlanes1Tracks += 1

        # if len(all_collinear_pointsZX) == 2 and len(all_collinear_pointsZY) == 2:
        #   twoPlanes2Tracks += 1

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
  print("1 piano 0 tracce: ", onePlane0Tracks)
  print("1 piano 1 traccia: ", onePlane1Tracks)
  print("1 piano 2 tracce: ", onePlane2Tracks)
  print("2 piani 1 traccia: ", twoPlanes1Tracks)
  print("2 piani 2 tracce: ", twoPlanes2Tracks)

  twoPlanes2Tracks_true_vertices.append(onePlane2Tracks_true_vertices)
  twoPlanes2Tracks_reco_vertices.append(onePlane2Tracks_reco_vertices)

  print("quanti cluster ZX: ", cluster_countsZX)
  print("quanti cluster ZY: ", cluster_countsZY)

  coord = ["x","y","z"]
  for i, c in enumerate(coord):
    diff_vertices = VertexCoordinatesHisto(twoPlanes2Tracks_true_vertices, twoPlanes2Tracks_reco_vertices, i)
    fig = plt.figure()
    plt.xlabel(f"{c} reco - {c} true")
    plt.ylabel("counts")
    #plt.hist(diff_vertices, 50, (-60,60), histtype='step')
    n, bins, patches = plt.hist(diff_vertices, 50, (-100,100), histtype='step')
    #fit the histo
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, len(n))
    #y = n
    popt, pcov = scp.optimize.curve_fit(gauss, x, n, p0 = [10, 0, 5])
    #print("gauss: ", max(gauss))
    plt.plot(x, gauss(x,*popt), color = 'red')
    plt.title(f"fit values {popt[1],popt[2]}")
  plt.show()

