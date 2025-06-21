import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from PIL import Image
import random
import time
import tqdm
import math
import os

from pylibraft.common import device_ndarray
from scipy.sparse import coo_matrix, csr_matrix, issparse, save_npz
from umap.umap_ import find_ab_params, simplicial_set_embedding, fuzzy_simplicial_set

from cuvs.neighbors import cagra

from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

# Wrapper function for parallel processing implementation (jank fix)
def cluster_stability(args):
    return fowlkes_mallows_score(*args)

def protein_norm(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

def preprocess(adata,modality):
  adata.var_names_make_unique()
  if modality in ['rna','atac']:
    sc.pp.filter_genes(adata,min_cells=10)
    sc.pp.log1p(adata)

    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X

  elif modality=='protein':
    adata.X = np.apply_along_axis(protein_norm, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X)))
    return adata.X     

  elif modality=='metabolite':
    sc.pp.log1p(adata)
    if scipy.sparse.issparse(adata.X):
      return adata.X.A
    else:
      return adata.X

# New (fast) connectivity graph based on Aagam's code
def get_connectivity_matrix(Y, intermediate_graph_degree = 128, graph_degree = 64, batch_size = 65000, itopk_size = 64, n_neighbors = 15, set_op_mix_ratio = 1.0, local_connectivity = 1.0):

  build_kwargs = {"graph_degree": graph_degree,
                  "intermediate_graph_degree" : intermediate_graph_degree}
  search_kwargs = {"itopk_size": itopk_size}
  connectivity_args = {'set_op_mix_ratio': set_op_mix_ratio,
                      'local_connectivity': local_connectivity}
  
  k = n_neighbors
  build_params = cagra.IndexParams(metric="sqeuclidean", build_algo="nn_descent", **build_kwargs)
  start = time.time()
  t0 = time.time()
  print("Building index....", end = "")
  index = cagra.build(build_params, Y)
  print(f' done: {(time.time() - t0)/60:.2f} min')  
  
  n_samples = Y.shape[0]

  print("Initializing neighbors and distances array...............")
  all_neighbors = np.zeros((n_samples, k), dtype=np.int32)
  all_distances = np.zeros((n_samples, k), dtype=np.float32)

  batchsize = batch_size
  n_batches = math.ceil(n_samples / batchsize)
  print(f"Starting processing of {n_batches} batches")

  for batch in tqdm.tqdm(range(n_batches)):
    start_idx = batch * batchsize
    stop_idx = min((batch + 1) * batchsize, n_samples)
    batch_Y = device_ndarray(Y[start_idx:stop_idx, :])

    search_params = cagra.SearchParams(**search_kwargs)
    distances, neighbors = cagra.search(
            search_params, index, batch_Y, k
    )
    all_neighbors[start_idx:stop_idx, :] = neighbors.copy_to_host()
    all_distances[start_idx:stop_idx, :] = distances.copy_to_host()

  print(f"KNN computation completed in {time.time() - t0:.2f} seconds")
  all_distances = np.sqrt(all_distances)
  t0 - time.time()
  print(f'Calculating connectivities.....', end = '')
  
  X = coo_matrix(([], ([], [])), shape = (all_neighbors.shape[0], 1))
  connectivites, _sigmas, _rhos = fuzzy_simplicial_set(
    X = X,
    n_neighbors = k,
    random_state = 100,
    metric = 'euclidean',
    knn_indices = all_neighbors,
    knn_dists = all_distances,
    verbose = True,
    **connectivity_args
  )
  print(f' done: {(time.time() - t0)/60:.2f} min')
  print(f'---done: {(time.time() - start)/60:.2f} min---')
  return connectivites.tocoo()

def calculate_affinity(X1, sig=30, sparse = False, neighbors = 100):
  if not sparse:
    dist1 = pairwise_distances(X1)
    a1 = np.exp(-1*(dist1**2)/(2*(sig**2)))
    return a1
  else:
    dist1 = kneighbors_graph(X1, n_neighbors = neighbors, mode='distance')
    dist1.data = np.exp(-1*(dist1.data**2)/(2*(sig**2)))
    dist1.eliminate_zeros()
    return dist1

def slice_sparse_coo_tensor(t, keep_indices):
    t = t.coalesce()
    indices = t.indices()
    values = t.values()
    
    row_mask = torch.isin(indices[0], keep_indices)
    col_mask = torch.isin(indices[1], keep_indices)
    
    mask = row_mask & col_mask

    new_indices = indices[:,mask]
    new_values = values[mask]
    
    # Original adjacency matrix has indices in order from 0-n
    # With the dataloader we shuffled them
    # Now we map new indices to be in the "correct" order
    lookup_table = torch.empty(indices.max() + 1)
    for i, idx1 in enumerate(keep_indices):
        lookup_table[idx1] = i
    remapped_indices = lookup_table[new_indices]
    
    out = torch.sparse_coo_tensor(remapped_indices, new_values, size = (len(keep_indices), len(keep_indices)))
    return out

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)



def cmap_tab30(x):
    n_base = 20
    n_max = 30
    brightness = 0.7
    brightness = (brightness,) * 3 + (1.0,)
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) * (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x-n_base) * brightness
        + isin_beyond * (0.0, 0.0, 0.0, 1.0))
    return color


def cmap_tab70(x):
    cmap_base = cmap_tab30
    brightness = 0.5
    brightness = np.array([brightness] * 3 + [1.0])
    color = [
        cmap_base(x),  # same as base colormap
        1 - (1 - cmap_base(x-20)) * brightness,  # brighter
        cmap_base(x-20) * brightness,  # darker
        1 - (1 - cmap_base(x-40)) * brightness**2,  # even brighter
        cmap_base(x-40) * brightness**2,  # even darker
        [0.0, 0.0, 0.0, 1.0],  # black
        ]
    x = x[..., np.newaxis]
    isin = [
        (x < 30),
        (x >= 30) * (x < 40),
        (x >= 40) * (x < 50),
        (x >= 50) * (x < 60),
        (x >= 60) * (x < 70),
        (x >= 70)]
    color_out = np.sum(
            [isi * col for isi, col in zip(isin, color)],
            axis=0)
    return color_out


def plot(clusters,locs):
  locs['2'] = locs['2'].astype('int')
  locs['3'] = locs['3'].astype('int')
  im1 = np.empty((locs['2'].max()+1, locs['3'].max()+1))
  im1[:] = np.nan
  im1[locs['2'],locs['3']] = clusters
  im2 = cmap_tab70(im1.astype('int'))
  im2[np.isnan(im1)] = 1
  im3 = Image.fromarray((im2 * 255).astype(np.uint8))
  return im3

def plot_on_histology(clusters, locs, im, scale, s=10):
  locs = locs*scale
  locs = locs.round().astype('int')
  im = im[(locs['4'].min()-10):(locs['4'].max()+10),(locs['5'].min()-10):(locs['5'].max()+10)]
  locs = locs-locs.min()+10
  cmap1 = mcolors.ListedColormap([cmap_tab70(np.array(i)) for i in range(len(np.unique(clusters)))])
  plt.imshow(im, alpha=0.7); 
  plot = plt.scatter(x=locs['5'], y=locs['4'], c = clusters, cmap=cmap1, s=s); 
  plt.axis('off'); 
  return plot

def set_random_seed(seed=100):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)

def get_train_test_validation_split(df, group_keys, sample_key, test_size = 0.2, validation_size = 0.25, random_state = 100):
  idx_all = np.array([i for i in range(len(df))])
  
  # Group dataframe by group_keys, then random sample by sample keys
  g = df.groupby(group_keys, observed = False)[sample_key].agg(['unique'])
  # Get the initial train/test split
  g['train_test'] = g['unique'].apply(lambda x: train_test_split(x, test_size = test_size, random_state = random_state))
  # Split the train data into train/validation
  g['train_validation'] = g['train_test'].apply(lambda x: train_test_split(x[0], test_size = validation_size, random_state=random_state))
  # Get the batches as numpy arrays (Earlier functions made pandas columns with lists of batches)
  test_batches = np.concat([x[1] for x in g['train_test'].values])
  train_batches = np.concat([x[0] for x in g['train_validation'].values])
  validation_batches = np.concat([x[1] for x in g['train_validation'].values])
  
  # Get numeric indices for train/test/validation data based on batches found above
  out = {
    'train': idx_all[df[sample_key].isin(train_batches)],
    'test': idx_all[df[sample_key].isin(test_batches)],
    'validation': idx_all[df[sample_key].isin(validation_batches)]
  }

  return out
