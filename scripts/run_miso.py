from miso.utils import *
from miso import Miso
from miso.nets import MisoDataSet
import pandas as pd
import numpy as np
import scanpy as sc
import os
import time
import matplotlib.pyplot as plt
import tqdm
import torch
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir_out', default = '/common/lamt2/miso_rapids/miso/data/outputs', type = str, help = "Directory for miso output")
parser.add_argument('-a', '--anndata', default = '/common/lamt2/miso_rapids/miso/adata_full.h5ad', type = str, help = "Input anndata file path")
parser.add_argument('-m', '--modality', default = ['X_scVI', 'X_agg_uniform_r88', 'X_virchow_weighted'], nargs = '+', type = str, help = "Keys in anndata.obsm for input modalities")
parser.add_argument('-t', '--trained', default = [1, 1, 0], nargs = '+', type = int, help = "Indicates which input modalities are final embeddings (already trained) (1 = True, 0 = False)")
parser.add_argument('-n', '--n_clusters', default = None, type = int, help = "Number of clusters for KMeans (default to None to calculate best using FMI stability)")
parser.add_argument('-l', '--learning_rate', default = 0.01, type = float, help = "Learning rate for training model")
parser.add_argument('-p', '--patience', default = 10, type = int, help = "Patience for early stopping")
parser.add_argument('-e', '--epochs', default = 1000, type = int, help = "Number of epochs for training")
parser.add_argument('-s', '--slide', default = 'all', type = str, help = 'Slicing for anndata slide')
parser.add_argument('-n_neighbors', default = 15, type = int, help = "Number of nearest neighbors for knn")
parser.add_argument('--batch_size', default = 2**18, type = int, help = "Batch size for training")
parser.add_argument('--train_on_full_dataset', action = 'store_true', help = "Don't split and train on full dataset")
parser.add_argument('--test_size', default = 0.2, type = float, help = "Fraction of data for test split")
parser.add_argument('--validation_size', default = 0.25, type = float, help = "Fraction of (total) data for validation split")
parser.add_argument('--split_by_batch', action = 'store_true', help = "Split train/test by batch instead of by cell")
parser.add_argument('--delta', default = 0.005, type = float, help = "Min Delta for loss improvement for early stopping")
parser.add_argument('--n_min', default = 10, type = int, help = "Min clusters for auto-clustering")
parser.add_argument('--n_max', default = 30, type = int, help = "Max clusters for auto-clustering")
parser.add_argument('--n_iter', default = 10, type = int, help = "Iterations to try for auto-clustering")

args, unknown = parser.parse_known_args()
args = vars(args)

# Unpack arguments
dir_out = args['dir_out']
f_anndata = args['anndata']
"""
Explaination of default modalities:
    X_scVI = scvi latent rep
    X_agg_uniform_r88 = aggregated spatial rep from Rick's code (uniform = uniform weights, r88 = 88 um radius)
    X_virchow = foundation model rep
"""
modalities = args['modality']
final_embedding = [bool(x) for x in args['trained']]
n_clusters = args['n_clusters']
n_neighbors = args['n_neighbors']
epochs = args['epochs']
cluster_args = {
    'n_min': args['n_min'],
    'n_max': args['n_max'],
    'n_iter': args['n_iter'],
    'save_dir': f'{dir_out}/auto_cluster_stability.png'
}
batch_size = args['batch_size']
test_size = args['test_size']
validation_size = args['validation_size']
learning_rate = args['learning_rate']
split_by_batch = args['split_by_batch']

assert (len(modalities) == len(final_embedding)), "Modality and trained input args must be same length"

# Make output directory if does not exist
from pathlib import Path
if not os.path.exists(dir_out):
    Path(dir_out).mkdir(parents = True, exist_ok=True)
    
import psutil
print(f"CPU Memory: {psutil.virtual_memory().used >> 30:.2f}/{psutil.virtual_memory().available >> 30:.2f} GB used/available")

# Use GPU if available. Otherwise use cpu
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. GPU:", torch.cuda.get_device_name())
    print(f"{torch.cuda.device_count()} devices available")
    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"Free GPU memory: {free_memory >> 30 :.2f} GB")
    print(f"Total GPU memory: {total_memory >> 30:.2f} GB")
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")

seed = 100
set_random_seed(seed, device = device)

# Read in the anndata
t0 = time.time()
print("Reading anndata ..... ", end = '')
adata = sc.read_h5ad(f'{f_anndata}')
print(f' done: {(time.time() - t0)/60:.2f} min')
# Associate xenium slide name to h&e slide name
anno_dict = {
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060075__Region_1__20250213__202651': 'ag_hpv_01',
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060077__Region_1__20250213__202651': 'ag_hpv_02',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060354__Region_1__20250224__233922': 'ag_hpv_04',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060367__Region_1__20250224__233922': 'ag_hpv_03',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0059911__Region_1__20250304__005817': 'ag_hpv_06',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0060395__Region_1__20250304__005817': 'ag_hpv_05',
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060364__Region_1__20250305__223715': 'ag_hpv_08',
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060366__Region_1__20250305__223715': 'ag_hpv_07',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060488__Region_1__20250312__004017': 'ag_hpv_09',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060493__Region_1__20250312__004017': 'ag_hpv_10'
}
if args['slide'] != 'all':
    adata.obs['slide_idx'] = adata.obs['slide_idx'].map(anno_dict)
    adata = adata[adata.obs['slide_idx'] == args['slide']]
    
# Get the train/test/split
external_index = None
if split_by_batch:
    try:
        external_index = get_train_test_validation_split(adata.obs, 'slide', 'batch', test_size = test_size, validation_size = validation_size, random_state = seed)
    except Exception as e:
        print(f"Error during train/test split: {e}. Defaulting to splitting by cells")

# Batch size to run cagra knn graph
connectivity_args = {
    'batch_size': batch_size,
    'n_neighbors': n_neighbors
}

# Arguments for early stopping
early_stopping_args = {
    'patience': args['patience'], # stop training if score doesn't improve after n epochs
    'delta': args['delta']      # minimum score improvement to restart early stopping counter
}

# Make the input datasets:
datasets = []
for m,t in zip(modalities, final_embedding):
    datasets.append(MisoDataSet(
        m,
        adata.obsm[m],
        pcs = None,
        adj = None,
        device = device,
        is_final_embedding = t,
        epochs = epochs,
        batch_size = batch_size,
        random_state = seed,
        learning_rate = learning_rate,
        connectivity_args = connectivity_args,
        early_stopping_args = early_stopping_args,
    ))
    
# Initialize the miso model
model = Miso(
    datasets,
    device = device,
    external_indexing = external_index,
    split_data = (not args['train_on_full_dataset']),
    test_size = test_size,
    validation_size = validation_size,
    random_state = seed
)

# Save pca and adjacency matrix
for d in model.datasets:
    if model.datasets[d].adj is not None:
        torch.save(model.datasets[d].adj, f'{dir_out}/adj_{d}.pt')
    if model.datasets[d].pcs is not None:
        np.save(f'{dir_out}/pca_{d}.npy', model.datasets[d].pcs.numpy())

print("\n\n----Training model----")

# Train the untrained modalities
model.train()
# Save the exact training loss and trained models for each modality
model.save_loss(dir_out)
for d in model.datasets:
    if model.datasets[d].mlp is not None:
        torch.save(model.datasets[d].mlp.state_dict(), f'{dir_out}/model_{d}.pt')
print('\n---done training models---')
# Calculate the embeddings for clustering
model.get_embeddings()
# Save the embeddings
np.save(f'{dir_out}/X_miso.npy', model.emb)

print("\n\n----Clustering embeddings using kmeans----")
if n_clusters is None:
    # Perform clustering based on FMI stability
    # Check between 10 and 30 clusters (inclusive) performing 10 iterations of each. Save output stability plot as a png
    model.auto_cluster(**cluster_args)
else:
    model.cluster(n_clusters = n_clusters)
    
# Save the clusters as a .pkl file
adata.obs['miso'] = model.clusters.astype(str)
adata.obs[['miso']].to_pickle(f'{dir_out}/niches.pkl')

print("\n\n---Clustering embeddings using leiden---")
# Manual leiden clustering different resolutions:
for r in [0.01, 0.015, 0.02, 0.05, 0.1, 0.5]:
    res = str(r).replace('.', 'p')
    clusters = get_leiden_clusters(model.emb.astype(np.float32), connectivity_args = connectivity_args, resolution = r, random_state = seed)
    adata.obs[f'miso_leiden{res}'] = clusters
    adata.obs[[f'miso_leiden{res}']].to_pickle(f'{dir_out}/niches_leiden{res}.pkl')

print('\n\n---done running miso---\n\n')