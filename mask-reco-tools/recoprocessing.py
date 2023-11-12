import numpy as np 
from skimage import exposure, metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, energy_distance



def compute_pca(data,geotree, cut, plot=False):
  xform = geotree.gxform()[0:3,0:3]
  scale = geotree.voxel_size
  pca=PCA(n_components=3)
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
  #print(xyz.shape)
  x, y, z = np.vsplit(xyz, 3)
  pca.fit(np.transpose(xyz))
  if plot == True:
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
  c = pca.mean_
  w = pca.explained_variance_
  #distance = np.amax(np.linalg.norm(np.transpose(xyz) - c, axis=1))
  maxdist=[]
  for i, v in enumerate(pca.components_):
    point = xyz.T - c
    dist = []
    for p in point:
      proj = np.dot(v, p) / (np.linalg.norm(v) ** 2) * v
      distance = np.linalg.norm(proj)
      dist.append(distance)
    md = np.amax(np.asarray(dist))
    dirp= v * md
    maxdist.append(md)
    if plot == True:
      ax.quiver(*c,*dirp, color='xkcd:pink', arrow_length_ratio=0.1, linewidth=2 )
  if plot==True:
    ax.scatter(x,y,z, alpha=0.2)
    plt.axis('equal')
  return (c, pca.components_[0], maxdist )


def mask_voxels(data):
  shape = data.shape
  ellipse_mask = np.zeros(shape = shape[:-1])
  hx = shape[0] / 2.
  hy = shape[1] / 2.
  hz = shape[2] / 2.
  for ix, iy in np.ndindex(ellipse_mask.shape):
    y = iy - hy + 0.5
    x = ix - hx + 0.5
    if y**2 / (hy - 5)**2 + x**2 / (hx-5)**2 <= 1:  
      ellipse_mask[ix, iy] = 1
  mask = np.broadcast_to(ellipse_mask[..., None], shape=shape).copy() 
  print("volume = ", np.sum(mask))
  data *= mask
  return data, mask


def equalizeHistogram(data):
  flatdata = data.flatten()
  #print("flat")
  mask = np.ones(shape=data.shape).flatten()
  hist, bin_edges = np.histogram(flatdata,bins=256)   
  #print("hist")
  bin_max = np.partition(hist, kth=-2)[-2]
  mm = np.where(hist == bin_max )[0][0]
  threshold = bin_edges[mm]
  flatdata[flatdata<threshold] = 0
  mask[flatdata<threshold] = 0
  dataEqualized = exposure.equalize_hist(flatdata, mask = mask, nbins=1024)     
  return dataEqualized

def compute_ssim(data, truth):
  truth = np.float32(truth)
  ssim = metrics.structural_similarity(data , truth)
  return ssim


   
if __name__ == '__main__':
  data = np.random.normal(10, 2, 1000)
  equalized = equalizeHistogram(data)
  plt.figure()
  plt.hist(data, bins= 100, label ='data')
  plt.hist(equalized, bins=100,  label ='equalized data')
  plt.legend()
  plt.show()




