from miso.utils import *
from miso import Miso
import pandas as pd
import numpy as np
import scanpy as sc
import os
import time
import matplotlib.pyplot as plt

import torch
import random
seed=100
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir_out', default = '/common/lamt2/miso_rapids/miso/data/outputs', type = str, help = "Directory for miso output")
parser.add_argument('-a', '--anndata', default = '/common/lamt2/miso_rapids/miso/adata_full.h5ad', type = str, help = "Input anndata file path")
parser.add_argument('-m', '--modality', default = ['X_scVI', 'X_agg_uniform_r88', 'X_virchow_weighted'], nargs = '+', type = str, help = "Keys in anndata.obsm for input modalities")
parser.add_argument('-t', '--trained', default = [1, 1, 0], nargs = '+', type = int, help = "Indicates which input modalities are final embeddings (already trained) (1 = True, 0 = False)")
parser.add_argument('-n', '--n_clusters', default = None, type = int, help = "Number of clusters for KMeans (default to None to calculate best using FMI stability)")
parser.add_argument('-l', '--learning_rate', default = 0.1, type = float, help = "Learning rate for training model")
parser.add_argument('-p', '--patience', default = 10, type = int, help = "Patience for early stopping")
parser.add_argument('--train_on_full_dataset', action = 'store_true', help = "Don't split and train on full dataset")
parser.add_argument('--test_size', default = 0.2, type = float, help = "Fraction of data for test split")
parser.add_argument('--validation_size', default = 0.25, type = float, help = "Fraction of (total) data for validation split")
parser.add_argument('--split_by_batch', action = 'store_true', help = "Split train/test by batch instead of by cell")
parser.add_argument('--delta', default = 0.005, type = float, help = "Min Delta for loss improvement for early stopping")
parser.add_argument('--n_min', default = 10, type = int, help = "Min clusters for auto-clustering")
parser.add_argument('--n_max', default = 30, type = int, help = "Max clusters for auto-clustering")
parser.add_argument('--n_iter', default = 10, type = int, help = "Iterations to try for auto-clustering")

args = vars(parser.parse_args())

# Explaination of default modalities
#    X_scVI = scvi latent rep
#    X_agg_uniform_r88 = aggregated spatial rep from Rick's code (uniform = uniform weights, r88 = 88 um radius)
#    X_virchow = foundation model rep

dir_out = args['dir_out']
f_anndata = args['anndata']
modalities = args['modality']
final_embedding = [bool(x) for x in args['trained']]
n_clusters = args['n_clusters']
cluster_args = {
    'n_min': args['n_min'],
    'n_max': args['n_max'],
    'n_iter': args['n_iter'],
    'save_dir': f'{dir_out}/auto_cluster_stability.png'
}

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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA is available. GPU:", torch.cuda.get_device_name())
    print(f"{torch.cuda.device_count()} devices available")
    free_memory, total_memory = torch.cuda.mem_get_info()
    print(f"Free GPU memory: {free_memory >> 30 :.2f} GB")
    print(f"Total GPU memory: {total_memory >> 30:.2f} GB")
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")

# Read in the anndata
t0 = time.time()
print("Reading anndata ..... ", end = '')
adata = sc.read_h5ad(f'{f_anndata}')
print(f' done: {(time.time() - t0)/60:.2f} min')

# Get the train/test/split
external_index = None
if split_by_batch:
    try:
        external_index = get_train_test_validation_split(adata.obs, 'slide', 'batch', test_size = test_size, validation_size = validation_size, random_state = seed)
    except Exception as e:
        print(f"Error during train/test split: {e}. Defaulting to splitting by cells")

# Batch size to run cagra knn graph
connectivity_args = {'batch_size': 2**18}

# Arguments for early stopping
early_stopping_args = {
    'patience': args['patience'], # stop training if score doesn't improve after n epochs
    'delta': args['delta']      # minimum score improvement to restart early stopping counter
}

# Initialize the miso model
model = Miso(
    [adata.obsm[m] for m in modalities],
    is_final_embedding=final_embedding, 
    device = device,
    batch_size = 2**18, # Batch size for training
    epochs = 1000,
    split_data = (not args['train_on_full_dataset']),
    test_size = test_size,
    val_size = validation_size,
    random_state = seed,
    learning_rate = learning_rate,
    connectivity_args = connectivity_args, 
    external_indexing = external_index,
    early_stopping_args = early_stopping_args
)

print("Training model")
# Train the untrained modalities
model.train()
# Save the exact training loss and trained models for each modality
model.save_loss(dir_out)
for i in range(len(model.mlps)):
    torch.save(model.mlps[i].state_dict(), f'{dir_out}/model_modality_{i}.pt')

# Calculate the embeddings for clustering
model.get_embeddings()

# Save the embeddings
np.save(f'{dir_out}/X_miso.npy', model.emb)

print("Clustering embeddings")
if n_clusters is None:
    # Perform clustering based on FMI stability
    # Check between 10 and 30 clusters (inclusive) performing 10 iterations of each. Save output stability plot as a png
    model.auto_cluster(**cluster_args)
else:
    model.cluster(n_clusters = n_clusters)
    
# Save the clusters as a .pkl file
adata.obs['miso'] = model.clusters.astype(str)
adata.obs[['miso']].to_pickle(f'{dir_out}/niches.pkl')

print('---done---')