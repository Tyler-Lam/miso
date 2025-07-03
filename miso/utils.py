import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import random
import time
import tqdm
import math
import os
import sys

try:
    from cuml.preprocessing import StandardScaler
    from cuml.decomposition import IncrementalPCA, PCA
    import cupy as cp
    print('cuml was imported for scaling/PCA')
except:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import IncrementalPCA, PCA
    print('sklearn was imported for scaling/PCA\n\n!!! this will be much faster with cuml !!!')

    
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

  for batch in tqdm.tqdm(range(n_batches), desc = "Calculing knn"):
    start_idx = batch * batchsize
    stop_idx = min((batch + 1) * batchsize, n_samples)
    batch_Y = device_ndarray(Y[start_idx:stop_idx, :])

    search_params = cagra.SearchParams(**search_kwargs)
    distances, neighbors = cagra.search(
            search_params, index, batch_Y, k
    )
    all_neighbors[start_idx:stop_idx, :] = neighbors.copy_to_host()
    all_distances[start_idx:stop_idx, :] = distances.copy_to_host()

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
  return connectivites.tocoo()

# Old pairwise adjacency matrix calculation
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
  
# Vibe coded fast replacement for torch.isin() but kinda cool implementation:
# *Sorts the indices to keep
# *Finds where to insert tensor idx into sorted keep indices (capped at len(keep_idxs)-1)
# *If the tensor index equals the sorted keep index at the insertion index, then the tensor index is in the keep index
# *Example:
#   idx = [0,1,5,3,2,1], keep_idx = [3, 0, 1]
# *By inspection, we can see the mask to keep only indices in keep_idx should be [1,1,0,1,0,1]
#   sorted_keep = [0, 1, 3]
#   idxs = [0, 1, 2, 2, 2, 1]  <-- where to insert idx into sorted_keep (after clamping)
#   keep_sorted[idxs] = [0, 1, 3, 3, 3, 1]
#   (keep_sorted[idxs] == idx) = [1,1,0,1,0,1]  <-- final mask for indices
def isin_fast(idx, keep_idx):
    keep_sorted, _ = torch.sort(keep_idx)
    idxs = torch.searchsorted(keep_sorted, idx)
    idxs = torch.clamp(idxs, max=keep_sorted.size(0) - 1)
    return keep_sorted[idxs] == idx
  
def slice_sparse_coo_tensor(t, keep_indices):
    if not t.is_coalesced():
        t = t.coalesce()
    indices = t.indices()
    values = t.values()
    
    row_mask = isin_fast(indices[0], keep_indices)
    col_mask = isin_fast(indices[1], keep_indices)
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
  
def sc_loss(A,Y):
    row = A.coalesce().indices()[0]
    col = A.coalesce().indices()[1]
    rows1 = Y[row]
    rows2 = Y[col]
    dist = torch.norm(rows1 - rows2, dim=1)
    return (dist*A.coalesce().values()).mean()

def get_interaction_matrix(emb1, emb2):
  interaction = emb1[:,:,None] * emb2[:,None,:]
  interaction = interaction.reshape(interaction.shape[0], -1)
  interaction = torch.matmul(interaction, torch.pca_lowrank(interaction, q = min(emb1.shape[1], emb2.shape[1]))[2])
  interaction = StandardScaler().fit_transform(interaction.cpu().detach().numpy())
  return interaction

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


import gc
import glob


def compute_pcs_online(dir_npy, npca=128, batch_size=2**14):
    flist = glob.glob(dir_npy + '*')
    flist = [f for f in flist if 'index' not in f]

    print(f'{len(flist)} files will be processing...')
    
    if 'cupy' in sys.modules.keys():
        is_cuml = True
    else:
        is_cuml = False
    
    # Scale the input features
    scaler = StandardScaler()
    
    if is_cuml: 
        for f in tqdm.tqdm(flist, desc = f"Incremental scalar fit per slide using cuml"):
            partial_x = cp.asarray(np.nan_to_num(np.load(f).astype(np.float32)))        
            scaler.partial_fit(partial_x)
            del partial_x
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
    
    
    else:
        for f in tqdm.tqdm(flist, desc = f"Incremental scalar fit per slide using sklearn (slow)"):
            partial_x = np.nan_to_num(np.load(f).astype(np.float32))
            scaler.partial_fit(partial_x)
            del partial_x
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    
    # Get PCA
    ipca = IncrementalPCA(n_components = npca)
    
    
    print("Calculating PCs")
    if is_cuml:
    
        for f in tqdm.tqdm(flist, desc = f'Fitting PCA using cuml'):
    
            # scale using pre-fitted scaler then input PCA function
            scaled_x = scaler.transform(cp.asarray(np.nan_to_num(np.load(f).astype(np.float32))))
            n = scaled_x.shape[0]
            if n > batch_size:
                n_batches = math.ceil(n / batch_size)
                for j in range(n_batches):
                    partial_size = min(batch_size, n - batch_size * j)
                    partial_x = scaled_x[batch_size*j:batch_size*j+partial_size]
                
                    ipca.partial_fit(partial_x)
                
                    del partial_x
                    cp.get_default_memory_pool().free_all_blocks()

            else:
                ipca.partial_fit(scaled_x)

            del scaled_x
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
        for f in tqdm.tqdm(flist, desc = f'Transforming PCA using cuml'):

            # scale using pre-fitted scaler then input PCA function
            scaled_x = scaler.transform(cp.asarray(np.nan_to_num(np.load(f).astype(np.float32))))
           
            n = scaled_x.shape[0]
            pcs = cp.zeros((n, npca))
            
            if n > batch_size:
                n_batches = math.ceil(n / batch_size)
                for j in range(n_batches):
                    partial_size = min(batch_size, n - batch_size * j)
                    partial_x = scaled_x[batch_size*j:batch_size*j+partial_size]
                
                    pcs[batch_size*j: batch_size * j + partial_size] = ipca.transform(partial_x) 
                
                    del partial_x
                    cp.get_default_memory_pool().free_all_blocks()

            else:
                pcs = ipca.transform(scaled_x)

            # save pcs
            savefn = dir_npy + '/' + f.split('/')[-1].replace('featpercell',f'PC{npca}')
            cp.save(savefn, pcs)
            del pcs, scaled_x
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    else:   
        for f in tqdm.tqdm(flist, desc = f'Fitting PCA using sklearn (slow)'):
            partial_x = np.nan_to_num(np.load(f).astype(np.float32))
            # scale using pre-fitted scaler then input PCA function
            ipca.partial_fit(scaler.transform(partial_x))
            del partial_x
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        for f in tqdm.tqdm(flist, desc = f'Transforming PCA using sklearn (slow)'):
            partial_x = np.nan_to_num(np.load(f).astype(np.float32))
            pcs = ipca.transform(scaler.transform(partial_x))
    
            # save pcs
            savefn = dir_npy + '/' + f.split('/')[-1].replace('featpercell',f'PC{npca}')
            np.save(savefn, pcs)
            del partial_x, pcs
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

    print('done!')

    
