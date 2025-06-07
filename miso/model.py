from . nets import *
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from . utils import calculate_affinity, get_connectivity_matrix, slice_sparse_coo_tensor
import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import euclidean_distances
from scanpy.external.tl import phenograph
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse import kron
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import PCA
from PIL import Image
import scipy
import time

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


class Miso(nn.Module):
    def __init__(self, features, ind_views='all', combs='all', is_final_embedding = None, device='cpu', npca = 128, nembedding = 32, batch_size = 32, epochs = 100, connectivity_args = {}):
        super(Miso, self).__init__()
        
        self.device = device
        self.num_views = len(features)
        # List of booleans to check if we need to train on a given feature or if we already have the final embedding
        self.is_final_embedding = is_final_embedding if is_final_embedding is not None else [False for _ in len(features)]
        self.features = [torch.Tensor(i) for i in features]  # List of all input modalities
        features = [StandardScaler().fit_transform(i) for i in features]
        self.features_to_train = [self.features[i] for i in range(len(self.features)) if self.is_final_embedding[i] == False] # List of all untrained input modalities
        self.trained_features = [self.features[i] for i in range(len(self.features)) if self.is_final_embedding[i] == True] # List of all trained input modalities

        self.npca = npca # number of components for initial feature pca
        self.nembedding = min([nembedding] + [feat.shape[1] for feat in self.trained_features])  # number of components for final embedding (and interaction matrix pca). Can't be larger than smallest dim. embedding from pre-trained features
        self.epochs = epochs
        self.history = {} # To track training, validation, and test loss across modalities
        
        t0 = time.time()
        print("Calculating PCs")        
        pcs = [PCA(self.npca).fit_transform(i) if i.shape[1] > self.npca else i for i in self.features_to_train]
        print(f'---done: {(time.time()-t0)/60:.2f} min---')
        self.pcs = [torch.Tensor(i) for i in pcs] 
        
        # Adjacency matrix and pca only needed for untrained features
        t0 = time.time()
        print("Calculating adjacency matrices")
        adj = [get_connectivity_matrix(i.numpy(), **connectivity_args) for i in self.pcs]
        
        # Make dataset of feature + index to track adjacency matrix through batches
        self.dataloaders = [DataLoader(TensorDataset(i, torch.arange(len(i))), batch_size = batch_size, shuffle = True) for i in self.pcs]

        adj = [coo_matrix(i) for i in adj]
        indices = [torch.LongTensor(np.vstack((i.row, i.col))) for i in adj]
        values = [torch.FloatTensor(i.data) for i in adj]
        shape = [torch.Size(i.shape) for i in adj]
        self.adj = [torch.sparse_coo_tensor(indices[i], values[i], shape[i]) for i in range(len(adj))]

        if ind_views=='all':
            self.ind_views = list(range(len(self.features)))
        else:
            self.ind_views = ind_views   
        if combs=='all':
            self.combinations = list(combinations(list(range(len(self.features))),2))
        else:
            self.combinations = combs        

    def train(self):
        self.mlps = [MLP(input_shape = self.pcs[i].shape[1], output_shape = self.nembedding).to(self.device) for i in range(len(self.pcs))]

        def sc_loss(A,Y):
            row = A.coalesce().indices()[0]
            col = A.coalesce().indices()[1]
            rows1 = Y[row]
            rows2 = Y[col]
            dist = torch.norm(rows1 - rows2, dim=1)
            return (dist*A.coalesce().values()).mean()

        for i in range(len(self.dataloaders)):
            self.history[i] = {
                'training_loss': [],
                'validation_loss': [],
                'test_loss': []
            }
            self.mlps[i].train()
            loader = self.dataloaders[i]
            optimizer = optim.Adam(self.mlps[i].parameters(), lr=1e-3)
            training_loss = [] # Track average loss per epoch
            for epoch in tqdm(range(self.epochs), desc = f'Training network for modality {i+1}'):
                epoch_loss = 0.0
                for batch in tqdm(loader, desc = f'Current batch training', leave = False):
                    optimizer.zero_grad()
                    
                    x = batch[0].to(self.device)
                    x_hat = self.mlps[i](x)
                    Y1 = self.mlps[i].get_embeddings(x)
                    
                    loss1 = nn.MSELoss()(x, x_hat)
                    adj_batch =  slice_sparse_coo_tensor(self.adj[i], batch[1])
                    loss2 = sc_loss(adj_batch.to(self.device), Y1)
                    loss = loss1 + loss2

                    epoch_loss += loss * x.shape[0]

                    loss.backward()
                    optimizer.step()
                    
                training_loss.append(epoch_loss.cpu().detach().numpy() / len(loader.dataset))
            self.history[i]['training_loss'] = training_loss
            #np.save(f'/common/lamt2/miso/loss_{i}.npy', np.array(training_loss))

        """
        for i in range(len(self.pcs)):
            training_loss = []
            self.mlps[i].train()
            optimizer = optim.Adam(self.mlps[i].parameters(), lr=1e-3)          
            for epoch in tqdm(range(100), desc='Training network for modality ' + str(i+1)):
                optimizer.zero_grad()
                x_hat = self.mlps[i](self.pcs[i])
                Y1 = self.mlps[i].get_embeddings(self.pcs[i])
                loss1 = nn.MSELoss()(self.pcs[i],x_hat)
                loss2 = sc_loss(self.adj[i], Y1)
                loss=loss1+loss2
                training_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            np.save(f'/common/lamt2/miso/loss_{i}.npy', np.array(training_loss))
        """
    def get_embeddings(self):
        [self.mlps[i].eval() for i in range(len(self.pcs))]
        Y = [self.mlps[i].get_embeddings(self.pcs[i]) for i in range(len(self.pcs))] + self.trained_features
        if self.combinations is not None:
            interactions = [Y[i][:, :, None]*Y[j][:, None, :] for i,j in self.combinations]
            interactions = [i.reshape(i.shape[0],-1) for i in interactions]
            interactions = [torch.matmul(i,torch.pca_lowrank(i,q=self.nembedding)[2]) for i in interactions]
        Y = [Y[i] for i in self.ind_views]
        Y = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in Y]
        Y = np.concatenate(Y,1)
        if self.combinations is not None:
            interactions = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in interactions]
            interactions = np.concatenate(interactions,1)
            emb = np.concatenate((Y,interactions),1)
        else:
            emb = Y
        self.emb = emb

    def cluster(self, n_clusters=10):
      clusters = KMeans(n_clusters, random_state = 100).fit_predict(self.emb)
      self.clusters = clusters
      return clusters
    
