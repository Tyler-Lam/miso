from . nets import *
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from . utils import get_connectivity_matrix, slice_sparse_coo_tensor, cluster_stability, get_interaction_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import IncrementalPCA, PCA
import time
from collections import defaultdict

from itertools import permutations, combinations
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import math

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

class Miso(nn.Module):
    def __init__(self, datasets, ind_views='all', combs='all', device='cpu', nembedding = 32, random_state = None, external_indexing = None, split_data = True, test_size = 0.2, validation_size = 0.25):

        super(Miso, self).__init__()
        
        self.datasets = {d.name: d for d in datasets}
        start = time.time()

        print("\n\n----Initializing Miso model----")
        # Get min # of embedding nodes for untrained modalities
        self.nembedding = nembedding
        for d in self.datasets:
            if self.datasets[d].is_final_embedding:
                self.nembedding = min(self.nembedding, self.datasets[d].features_raw.shape[1])
            # Set consistent random seed for all modalities if provided
            if random_state is not None:
                self.datasets[d].random_state = random_state
        t0 = time.time()
        for d in self.datasets:
            print(f"\nPreprocessing modality {d}")
            self.datasets[d].preprocess()
            print(f'Done preprocessing modality {d}')
        print(f'\n---done preprocessing all datasets: {(time.time() - t0)/60:.2f} min---')
        self.device = device
        self.num_views = len(datasets)
        self.test_size = test_size
        self.validation_size = validation_size
        self.external_indexing = external_indexing
        self.random_state = random_state

        # Get consistent train/test splitting indices for all datasets
        if not split_data:
            for d in self.datasets:
                self.datasets[d].split_data = False
        else:
            self.external_indexing = {'train': [], 'test': [], 'validation': []}
            try:
                self.external_indexing['train'] = external_indexing['train']
                self.external_indexing['test'] = external_indexing['test']
                self.external_indexing['validation'] = external_indexing['validation']
            except:
                train_idx, test_idx = train_test_split(list(range(datasets[0].features.shape[0])), test_size = self.test_size, random_state = self.random_state)
                train_idx, validation_idx = train_test_split(train_idx, test_size = self.validation_size, random_state = self.random_state)

                self.external_indexing['train'] = train_idx
                self.external_indexing['test'] = test_idx
                self.external_indexing['validation'] = validation_idx
            for d in self.datasets:
                self.datasets[d].external_indexing = self.external_indexing

        if ind_views=='all':
            self.ind_views = list(self.datasets.keys())
        else:
            self.ind_views = ind_views
        if combs=='all':
            if len(self.datasets) > 1:
                self.combinations = list(combinations(list(self.datasets.keys()),2))
            else:
                self.combinations = None
        else:
            self.combinations = combs

    def train(self):
        for d in self.datasets:
            t0 = time.time()
            print(f'\nTraining modality {d}')
            if self.datasets[d].is_final_embedding:
                print(f'   modality already has embedding, skipping')
                continue
            self.datasets[d].nembedding = self.nembedding
            self.datasets[d].make_dataloaders()
            self.datasets[d].train()
            print(f'Done training modality {d}: {(time.time() - t0)/60:.2f} min')
        
    def save_loss(self, out_dir = ''):
        for d in self.datasets:
            if self.datasets[d].is_final_embedding:
                continue
            self.datasets[d].save_loss(out_dir)

    def get_embeddings(self):
        
        Y = [self.datasets[d].get_embedding() for d in self.ind_views]

        if self.combinations is not None:
            interactions = [get_interaction_matrix(self.datasets[i].get_embedding(), self.datasets[j].get_embedding()) for i,j in self.combinations]
            interactions = np.concatenate(interactions,1)
            Y = np.concatenate(Y, 1)
            emb = np.concatenate((Y, interactions), 1)
        else:
            Y = np.concatenate(Y, 1)
            emb = Y
        self.emb = emb
        return self.emb

    def cluster(self, n_clusters=10, random_state = 100):
        clusters = KMeans(n_clusters, random_state = random_state).fit_predict(self.emb)
        self.clusters = clusters
        return clusters

    # Function to find best clustering. Based on cellcharter approach with FMI stability
    def auto_cluster(self, n_min = 5, n_max = 20, n_iter = 10, random_state = 100, save_dir = None):

        # Do n_iter random clusterings for each number of clusters
        clusterings = defaultdict(list)
        for n in tqdm(range(n_min, n_max+1), desc = f"Performing clustering {n_iter} times per n_cluster"):
            for i in range(n_iter):
                clusterings[n].append(self.cluster(n_clusters = n, random_state = random_state + i))
        
        # Calculate pairwise cluster scores (parallelized) (poorly)
        t0 = time.time()
        print('Calculating FMI between n-1,n and n,n+1')
        res = []
        pairs =  [(clusterings[n][i],clusterings[n+1][j]) for i,j in permutations(range(n_iter), 2) for n in range(n_min, n_max)]
        pairs_idx =  [(n,n+1,i,j) for i,j in permutations(range(n_iter), 2) for n in range(n_min, n_max)]
        with ProcessPoolExecutor() as executor:
            results = executor.map(cluster_stability, pairs)
            
        # Make dataframe to simplify calculations (probably bad way to do but easier for me to plot with)
        n1 = []
        n2 = []
        i = []
        j = []
        score = []
        for x, res in zip(pairs_idx, results):
            n1.append(x[0])
            n2.append(x[1])
            i.append(x[2])
            j.append(x[3])
            score.append(res)
            
            n2.append(x[0])
            n1.append(x[1])
            j.append(x[2])
            i.append(x[3])
            score.append(res)
        df = pd.DataFrame({'n1': n1, 'n2': n2, 'i': i, 'j': j, 'score': score})
        
        # Plot cluster stability
        means = df[~(df['n1'] == df['n2'])].groupby(['n1'])['score'].mean()
        best_n = n_min + means.argmax()
        stds = df[~(df['n1'] == df['n2'])].groupby(['n1'])['score'].std()
        f, ax = plt.subplots()
        ax.errorbar(x = means.index.get_level_values('n1'), y = means, yerr = stds)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Cluster stability score")
        ax.set_title(f"Maximum at {best_n}")
        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches = 'tight')
        plt.show()
        plt.close()
            
        print(f'Best cluster found at n = {best_n} in {(time.time() - t0)/60:.2f} min. Picking best cluster for final clustering')
        
        # Find cluster with highest average FMI with remaining clusters for best_n
        pairs =  [(clusterings[best_n][i],clusterings[best_n][j]) for i,j in combinations(range(n_iter), 2)]
        pairs_idx =  [(i,j) for i,j in combinations(range(n_iter), 2)]
        
        scores_internal = np.zeros((n_iter, n_iter))
        with ProcessPoolExecutor() as executor:
            results = executor.map(cluster_stability, pairs)
            
        for (i,j), score in zip(pairs_idx, results):
            scores_internal[i][j] = score
            scores_internal[j][i] = score
        means = np.mean(scores_internal, axis = 0)
        best_i = np.argmax(means)
        self.clusters = clusterings[best_n][best_i]
        print('Finished auto-clustering')
        return self.clusters