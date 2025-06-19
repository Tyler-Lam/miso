from . nets import *
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from . utils import get_connectivity_matrix, slice_sparse_coo_tensor, cluster_stability
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.decomposition import IncrementalPCA
import time
from collections import defaultdict

from itertools import permutations, combinations
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

class Miso(nn.Module):
    def __init__(self, features, ind_views='all', combs='all', is_final_embedding = None, device='cpu', npca = 128, nembedding = 32, batch_size = 2**17, epochs = 100, learning_rate = 0.05, connectivity_args = {}, test_size = 0.2, val_size = 0.25, external_indexing = None, early_stopping_args = {}, parallel = False):
        """
        Parameters
        ------------------
        *features
            List of feature matrices for each modality
        *is_final_embedding
            List of booleans indicating if features for each respective modality are the final embeddings or not
        *npca
            Number of components for initial PCA
        *nembedding
            Number of nodes for embedding layer
        *batch_size
            Batch size for training
        *epochs
            Number of epochs for training
        *connectivity_args
            Keyword arguments for adjacency matrix calculations. See utils.get_connectivity_matrix
        *test_size, val_size
            Fractions for random train/test/validation splitting. Validation total fraction = (1 - test_size) * val_size
        *external_indexing
            Use predetermined labels for datasplitting. Must be dictionary with 'train', 'test', and 'validation' for each respective indexing
        *early_stopping_args
            Keyword arguments for EarlyStopping class (currently only learning_rate and delta)
        *parallel
            Use DistributedDataParallel (requires device == 'cuda') to split data efficiently (implementation in progress)
        """
        super(Miso, self).__init__()
        
        start = time.time()
        print("Initializing Miso model")
        self.device = device
        self.num_views = len(features)
        # List of booleans to check if we need to train on a given feature or if we already have the final embedding
        self.is_final_embedding = is_final_embedding if is_final_embedding is not None else [False for _ in len(features)]
        
        self.features = [torch.from_numpy(StandardScaler().fit_transform(i)) for i in features]  # List of all input modalities
        self.features_to_train = [self.features[i] for i in range(len(self.features)) if self.is_final_embedding[i] == False] # List of all untrained input modalities
        self.trained_features = [self.features[i] for i in range(len(self.features)) if self.is_final_embedding[i] == True] # List of all trained input modalities
        self.early_stopping_args = early_stopping_args
        self.npca = npca # number of components for initial feature pca
        self.nembedding = min([nembedding] + [feat.shape[1] for feat in self.trained_features])  # number of components for final embedding (and interaction matrix pca). Can't be larger than smallest dim. embedding from pre-trained features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.parallel = False
        self.history = {} # To track training, validation, and test loss across modalities
        
        t0 = time.time()
        print("Calculating PCs")
        #import psutil
        #print(f"Memory: {psutil.virtual_memory().used >> 30:.2f}/{psutil.virtual_memory().available >> 30:.2f} GB used/available")        
        pcs = [IncrementalPCA(self.npca, batch_size=2**18).fit_transform(i) if i.shape[1] > self.npca else i for i in self.features_to_train]
        print(f'---done: {(time.time()-t0)/60:.2f} min---')
        self.pcs = [torch.Tensor(i) for i in pcs] 
        
        # Adjacency matrix and pca only needed for untrained features
        t0 = time.time()
        print("Calculating adjacency matrices")
        adj = [get_connectivity_matrix(i.numpy(), **connectivity_args) for i in self.pcs]
        # Convert scipy coo matrix to torch sparse_coo_tensor
        indices = [torch.LongTensor(np.vstack((i.row, i.col))) for i in adj]
        values = [torch.FloatTensor(i.data) for i in adj]
        shape = [torch.Size(i.shape) for i in adj]
        self.adj = [torch.sparse_coo_tensor(indices[i], values[i], shape[i]) for i in range(len(adj))]
        # Make dataset of feature + index to track adjacency matrix through batches
        self.dataloaders = [DataLoader(TensorDataset(i, torch.arange(len(i))), batch_size = min(len(i), batch_size), shuffle = True) for i in self.pcs]
        
        # Get indices for train/test/validation splitting
        train_idx = []
        validation_idx = []
        test_idx = []
        
        if external_indexing is None:
            train_idx, test_idx = train_test_split(list(range(self.pcs[0].shape[0])), test_size = test_size, random_state = 100)
            train_idx, validation_idx = train_test_split(train_idx, test_size = val_size, random_state = 100)
        elif 'train' in external_indexing and 'test' in external_indexing and 'validation' in external_indexing:
            train_idx = external_indexing['train']
            validation_idx = external_indexing['validation']
            test_idx = external_indexing['test']
        else:
            print('External indexing requires "train", "test", and "validation" keys given in dictionary. Defaulting to random 60/20/20 train/validation/test split')
            train_idx, test_idx = train_test_split(list(range(self.pcs[0].shape[0])), test_size = test_size, random_state = 100)
            train_idx, validation_idx = train_test_split(train_idx, test_size = val_size, random_state = 100)

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.validation_idx = validation_idx

        self.train_loaders = [DataLoader(TensorDataset(i[train_idx], torch.IntTensor(train_idx)), batch_size = min(len(train_idx), batch_size), shuffle = True) for i in self.pcs]
        self.val_loaders = [DataLoader(TensorDataset(i[validation_idx], torch.IntTensor(validation_idx)), batch_size = min(len(validation_idx), batch_size), shuffle = True) for i in self.pcs]
        self.test_loaders = [DataLoader(TensorDataset(i[test_idx], torch.IntTensor(test_idx)), batch_size = min(len(test_idx), batch_size), shuffle = True) for i in self.pcs]

        if ind_views=='all':
            self.ind_views = list(range(len(self.features)))
        else:
            self.ind_views = ind_views   
        if combs=='all':
            if len(self.features) > 1:
                self.combinations = list(combinations(list(range(len(self.features))),2))
            else:
                self.combinations = None
        else:
            self.combinations = combs
        print(f'..... done initializing model: {(time.time() - start)/60:.2f} min')        

    def train(self):
        self.mlps = [MLP(input_shape = self.pcs[i].shape[1], output_shape = self.nembedding).to(self.device) for i in range(len(self.pcs))]
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            if self.parallel:
                self.mlps = [MisoDataParallel(m) for m in self.mlps]
        def sc_loss(A,Y):
            row = A.coalesce().indices()[0]
            col = A.coalesce().indices()[1]
            rows1 = Y[row]
            rows2 = Y[col]
            dist = torch.norm(rows1 - rows2, dim=1)
            return (dist*A.coalesce().values()).mean()

        for i in range(len(self.features_to_train)):
            self.history[i] = {
                'training_loss': [],
                'validation_loss': [],
            }
            early_stopping = EarlyStopping(**self.early_stopping_args)
            train_loader = self.train_loaders[i]
            val_loader = self.val_loaders[i]
            optimizer = optim.Adam(self.mlps[i].parameters(), lr=self.learning_rate)
            training_loss = [] # Track average loss per epoch
            validation_loss = []
            for epoch in (pbar := tqdm(range(self.epochs))):
                # First run on training set
                pbar.set_description(f'Processing Modality {i}, current score {early_stopping.best_score if early_stopping.best_score is not None else 0:.3f}, current count {early_stopping.counter}')
                
                self.mlps[i].train()
                epoch_train_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    x = batch[0].to(self.device)

                    x_hat = self.mlps[i](x)
                    Y1 = self.mlps[i].get_embeddings(x)
                    
                    loss1 = nn.MSELoss()(x, x_hat)
                    adj_batch =  slice_sparse_coo_tensor(self.adj[i], batch[1])
                    loss2 = sc_loss(adj_batch.to(self.device), Y1)
                    loss = loss1 + loss2

                    epoch_train_loss += loss * x.shape[0]

                    loss.backward()
                    optimizer.step()
                    
                training_loss.append(epoch_train_loss.cpu().detach().numpy() / len(train_loader.dataset))
                
                # Now run on validation set
                self.mlps[i].eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        
                        x = batch[0].to(self.device)

                        x_hat = self.mlps[i](x)
                        Y1 = self.mlps[i].get_embeddings(x)
                        
                        loss1 = nn.MSELoss()(x, x_hat)
                        adj_batch =  slice_sparse_coo_tensor(self.adj[i], batch[1])
                        loss2 = sc_loss(adj_batch.to(self.device), Y1)
                        loss = loss1 + loss2

                        epoch_val_loss += loss * x.shape[0]
                
                validation_loss.append(epoch_val_loss.cpu().detach().numpy() / len(val_loader.dataset))
                early_stopping(validation_loss[-1], self.mlps[i])
                if early_stopping.early_stop:
                    print(f"\nEarly stopping after {epoch} epochs, current loss = {validation_loss[-1]:.4f}, best loss = {-1*early_stopping.best_score:.4f}")
                    break
                        
            self.history[i]['training_loss'] = training_loss
            self.history[i]['validation_loss'] = validation_loss

    def save_loss(self, out_dir = ''):
        for i in range(len(self.features_to_train)):
            np.save(f'{out_dir}/training_loss_modality_{i}.npy', self.history[i]['training_loss'])
            np.save(f'{out_dir}/validation_loss_modality_{i}.npy', self.history[i]['validation_loss'])

            plt.plot(self.history[i]['training_loss'], label = 'Training')
            plt.plot(self.history[i]['validation_loss'], label = 'Validation')
            plt.xlabel("Epoch")
            plt.ylabel("Loss (Reconstruction + Spectral)")
            plt.legend()
            plt.savefig(f'{out_dir}/modality_{i}_loss_vs_epoch.png')
            plt.close()

    def get_embeddings(self):
        [self.mlps[i].eval() for i in range(len(self.pcs))]
        Y = [self.mlps[i].to('cpu').get_embeddings(self.pcs[i]) for i in range(len(self.pcs))] + self.trained_features
        Y = [Y[i] for i in self.ind_views]
        Y = [torch.from_numpy(StandardScaler().fit_transform(i.cpu().detach().numpy())) for i in Y]

        if self.combinations is not None:
            interactions = [Y[i][:, :, None]*Y[j][:, None, :] for i,j in self.combinations]
            interactions = [i.reshape(i.shape[0],-1) for i in interactions]
            interactions = [torch.matmul(i,torch.pca_lowrank(i,q=self.nembedding)[2]) for i in interactions]
            interactions = [StandardScaler().fit_transform(i.cpu().detach().numpy()) for i in interactions]
            interactions = np.concatenate(interactions,1)
            Y = np.concatenate(Y, 1)
            emb = np.concatenate((Y, interactions), 1)
        else:
            Y = np.concatenate(Y, 1)
            emb = Y
        self.emb = emb

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
            
        # Make dataframe to simplify calculations (probably bad to do but easier for me to plot with)
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