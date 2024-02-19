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
  
  #coord1=z, coord2=y
  for coord1,coord2 in points:
    for theta_index, theta in enumerate(thetas):
      rho = coord2 * np.cos(theta) + coord1 * np.sin(theta)
      rho_index = np.argmin(np.abs(rhos - rho))
      accumulator[rho_index, theta_index] += 1
  return accumulator, rhos, thetas

def FindLocalMaxima(accumulator):
  local_max_indices = sk.feature.peak_local_max(accumulator, min_distance=4, threshold_rel = 0.6, exclude_border=False)
  rho_thetas_max = []
  for i in local_max_indices:
    theta_max = thetas[i[1]]
    rho_max = rhos[i[0]]
    rho_thetas_max.append((rho_max, theta_max))
  return local_max_indices, rho_thetas_max

def FindClosestToLinePoints(points, rho_thetas_max):
  all_collinear_points = []
  collinear_points1 = []
  collinear_points2 = []
  points_labels = []
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
      points_labels.append(0)
    elif len(rho_thetas_max)>1 and closest_line == rho_thetas_max[1]:
      collinear_points2.append((coord1,coord2,coord3))
      points_labels.append(1)

  all_collinear_points.append(collinear_points1)
  if len(rho_thetas_max)>1:
    all_collinear_points.append(collinear_points2)
  return all_collinear_points, points_labels

def ExtractTrueParameters(fname, ev):
  true_event = loadPrimariesEdepSim(fname, ev)
  true_vertices = true_event.vertices
  vertices_coord = []
  for vertex in true_vertices:
    vertices_coord.append(vertex.position)
    particles = vertex.particles
    directions = []
    for j, particle in enumerate(particles):
      momentum = particle.momentum
      directions.append(momentum)
  return vertices_coord, directions

def Fit(points1, points2):
  p, res,_,_,_ = np.polyfit(points1, points2, 1, full=True)
  slope = p[0]
  intercept = p[1]
  #plt.plot(points1, slope*points2 + intercept, color='red')
  return slope, intercept, res

def GetRecoVertex(all_collinear_points):
  all_slopesZY = []
  all_interceptsZY = []
  all_slopesZX = []
  all_interceptsZX = []
  for collinear_points in all_collinear_points:
    #ZY
    collinear_points = np.asarray(collinear_points)
    slopeZY, interceptZY, resZY = Fit(collinear_points[:,2], collinear_points[:,1])
    #print("resZY: ", resZY)
    all_slopesZY.append(slopeZY)
    all_interceptsZY.append(interceptZY)
    #ZX
    slopeZX, interceptZX, resZX = Fit(collinear_points[:,2], collinear_points[:,0])
    #print("resZX: ", resZX)
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
  return reco_vertex

def GetRecoDirections(reco_vertex, all_collinear_points):
  all_reco_directions = []
  for collinear_points in all_collinear_points: 
    collinear_points = np.asarray(collinear_points)
    slopeZY, interceptZY, _ = Fit(collinear_points[:,2], collinear_points[:,1])
    slopeZX, interceptZX, _ = Fit(collinear_points[:,2], collinear_points[:,0])
    z = collinear_points[0,0]
    point2 = np.asarray((slopeZX*z + interceptZX, slopeZY*z + interceptZY, z))#x,y,z
    reco_vertex = np.asarray(reco_vertex)

    direction = (point2 - reco_vertex)
    all_reco_directions.append(direction)  
  return all_reco_directions 


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

  edepsim_file = "./data/data_1-12/events-in-GRAIN_LAr_lv.999.edep-sim.root"
  geom = load_geometry(geometryPath, defs)

  centers_all_ev, amps_all_ev, recodata_all_ev = AllCentersAllEvents(selectedEvents, fpkl4, fpkl2, geom.fiducial, applyGradient=False)

  """LOOP SU TUTTI GLI EVENTI SELEZIONATI"""
  for i in range(len(selectedEvents)):
    all_clusters_in_ev, y_pred = Clustering(centers_all_ev[i], amps_all_ev[i])

    print("evento: ", selectedEvents[i])

    #************************MC TRUTH**************************
    vertices, directions = ExtractTrueParameters(edepsim_file, selectedEvents[i])
    print("TRUE vertex: ", vertices)
    print("TRUE directions: ", directions)
    #**********************************************************

    #************************3D PLOT****************************
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
    #************************************************************

    #**************************ACCUMULATOR***************************
    all_lpcs = np.concatenate([cluster.LPCs[0] for cluster in all_clusters_in_ev])
    points = list(zip(all_lpcs[:,0],all_lpcs[:,1],all_lpcs[:,2]))
    pointsZY = list(zip(all_lpcs[:,2],all_lpcs[:,1]))
    
    accumulator, rhos, thetas = HoughTransform(pointsZY)

    local_max_indices, rho_thetas_max = FindLocalMaxima(accumulator)
    # print("indices_local_max (rows, columns): ", local_max_indices)
    # print("rhos thetas max: ", rho_thetas_max)

    for i in local_max_indices:
      local_max_values = accumulator[i[0],i[1]]
      # print("max_values: ", local_max_values)
      # print("theta: ", np.rad2deg(thetas[i[1]]))
      # print("rho: ", rhos[i[0]])

    fig3 = plt.figure()
    plt.imshow(accumulator, cmap='cividis', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect = 'auto')
    plt.xlabel('theta')
    plt.ylabel('rho')
    plt.colorbar()

    for i in local_max_indices:
      plt.plot(np.rad2deg(thetas[i[1]]),rhos[i[0]], 'ro')
    #*********************************************************************
    
    #***********************************2D PLOT***************************
    all_collinear_points, points_labels = FindClosestToLinePoints(points, rho_thetas_max)

    fig2 = plt.figure()
    ax = fig2.add_subplot()
    for cluster in all_clusters_in_ev:
      single_curve = np.asarray(cluster.LPCs[0])
      ax.scatter(single_curve[:,2], single_curve[:,1], color = 'red')#allLPCpoints

    # all_points = []
    # for collinear_points in all_collinear_points:
    #   for points in collinear_points:
    #     all_points.append(points)
    # all_points = np.asarray(all_points)
    # ax.scatter(all_points[:,2], all_points[:,1], c=points_labels, cmap='cividis')
    
    #i collinear points sono organizzati come x,y,z; li disegno nel piano z-y
    all_collinear_points[0] = np.asarray(all_collinear_points[0])
    ax.scatter(all_collinear_points[0][:,2], all_collinear_points[0][:,1], color = 'green')
    if len(all_collinear_points)>1:
      all_collinear_points[1] = np.asarray(all_collinear_points[1])
      plt.scatter(all_collinear_points[1][:,2], all_collinear_points[1][:,1], color = 'blue')
    plt.ylim(-700,700)
    plt.xlim(-200,200)
    plt.gca().set_aspect('equal', adjustable='box')

    """HOUGH TRANSFORM LINES"""
    # for rho,theta in rho_thetas_max:
    #   z = np.linspace(-400, 400, 10000)
    #   y = -(np.cos(theta)/np.sin(theta))*z + rho/np.sin(theta)
    #   plt.plot(y, z)

    for collinear_points in all_collinear_points:
      slope, intercept, res = Fit(collinear_points[:,2], collinear_points[:,1])
      plt.plot(collinear_points[:,2], slope*collinear_points[:,2] + intercept, color = 'red')
    
    """RECO VERTEX"""
    reco_vertex = GetRecoVertex(all_collinear_points)
    print("RECO vertex: ", reco_vertex)

    """RECO DIRECTION"""
    all_reco_directions = GetRecoDirections(reco_vertex, all_collinear_points)
    print("RECO directions: ", all_reco_directions)

    theta = np.arccos(np.dot(all_reco_directions[0],directions[1])/(np.linalg.norm(all_reco_directions[0])*np.linalg.norm(directions[1])))
    
    print("angle: ", np.rad2deg(theta))#since the outcome of arccos is in radians

    # p, res,_,_,_ = np.polyfit(all_collinear_points[0][:,2], all_collinear_points[0][:,1], 1, full=True)
    # yfit = np.polyval(p,all_collinear_points[0][:,2])
    # residual = np.sqrt(np.sum((all_collinear_points[0][:,1]-yfit)**2)/len(all_collinear_points[0][:,1]))
    # print("resi: ", residual)
    # plt.plot(all_collinear_points[0][:,2], all_collinear_points[0][:,1]-yfit)


    # from matplotlib.patches import Circle

    # center = (0, 0)
    # radius = 220
    # circle = Circle(center, radius,facecolor = None, edgecolor = 'blue',alpha=0.1)
    # ax.add_patch(circle)
    plt.grid()
    plt.title("z-y plane")
    plt.xlabel("z (mm)")
    plt.ylabel("y (mm)")
    #*********************************************************************

    plt.show()