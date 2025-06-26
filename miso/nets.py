import torch.nn as nn
import torch
from torch import optim
from . utils import get_connectivity_matrix, slice_sparse_coo_tensor, sc_loss

from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.parametrizations import orthogonal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA, PCA
import scipy.sparse as sp
from itertools import combinations
from collections import defaultdict

import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
        
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer1 = nn.Linear(kwargs["input_shape"], kwargs['output_shape'])
        self.layer2 = orthogonal(nn.Linear(kwargs['output_shape'], kwargs['output_shape']) )
        self.layer3 = nn.Linear(kwargs['output_shape'],kwargs["input_shape"])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def get_embeddings(self,x):
        x = self.layer2(self.layer1(x))
        return x

# Wrapper class for DataParallel so we can access the MLP function (currently for get_embeddings())
class MisoDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
class EarlyStopping:
    def __init__(self, patience = 10, delta = 0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.epoch = -1
    
    def __call__(self, val_loss, model):
        score = -val_loss
        self.epoch += 1
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = self.epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = self.epoch
            self.counter = 0
            
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
# Input dataset class for miso
class MisoDataSet:
    def __init__(self, name, features, pcs = None, adj = None, is_final_embedding = False, npca = 128, nembedding = 32, batch_size = 2**18, device = 'cpu', epochs = 1000, learning_rate = 0.05, connectivity_args = {}, split_data = True, test_size = 0.2, val_size = 0.2, external_indexing = None, early_stopping_args = {}, random_state = 100):
        """
        Parameters
        ------------------
        *name
            Name for dataset
        *features
            List of feature matrices for each modality
        *pca
            Matrix of pca, if already computed. Can be numpy array or torch tensor
        *adj
            Adjacency matrix (cagra fuzzy knn), if already computer. Can be scipy sparse matrix or torch sparse tensor
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
        """
    
        self.name = name
        self.batch_size = batch_size
        self.features_raw = features
        self.pcs = pcs
        self.adj = adj
        self.emb = None
        self.is_final_embedding = is_final_embedding
        self.npca = npca
        self.nembedding = nembedding
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.connectivity_args = connectivity_args
        self.split_data = split_data
        self.test_size = test_size
        self.validation_size = val_size
        self.external_indexing = external_indexing
        self.early_stopping_args = early_stopping_args
        self.random_state = random_state
        self.mlp = None
        self.history = None
        
    def preprocess(self):
        
        # Scale the input features
        scaler = StandardScaler()
        n = self.features_raw.shape[0]
        n_batches = math.ceil(n / self.batch_size)
        for j in tqdm(range(n_batches), desc = f"Incremental scalar fit for modality {self.name}"):
            partial_size = min(self.batch_size, n - self.batch_size * j)
            partial_x = self.features_raw[self.batch_size*j:self.batch_size*j+partial_size]
            scaler.partial_fit(partial_x)
        self.features = torch.from_numpy(scaler.transform(self.features_raw))
        
        if self.is_final_embedding:
            self.emb = self.features
            return

        # Get PCA
        if self.pcs is None:
            print("Calculating PCs")
            pcs = np.zeros((self.features.shape[0], self.npca))
            if self.batch_size < self.features.shape[0]:
                ipca = IncrementalPCA(n_components = self.npca)
                n = self.features.shape[0]
                n_batches = math.ceil(n / self.batch_size)
                for j in tqdm(range(n_batches), desc = f'Fitting PCA for modality {self.name}'):
                    partial_size = min(self.batch_size, n - self.batch_size * j)
                    partial_x = self.features[self.batch_size * j: self.batch_size * j + partial_size]
                    ipca.partial_fit(partial_x)
                for j in tqdm(range(n_batches), desc = f'Transforming PCA for modality {self.name}'):
                    partial_size = min(self.batch_size, n - self.batch_size * j)
                    partial_x = self.features[self.batch_size * j: self.batch_size * j + partial_size]
                    pcs[self.batch_size*j: self.batch_size * j + partial_size] = ipca.transform(partial_x)
            else:
                pcs = PCA(self.npca).fit_transform(self.features)
            
            self.pcs = torch.Tensor(pcs)
            
        elif isinstance(self.pcs, np.ndarray):
            self.pcs = torch.Tensor(self.pcs)
        
        # Get adj matrix
        if self.adj is None:
            print ("Calculating adjacency matrix")
            adj = get_connectivity_matrix(self.pcs.numpy(), **self.connectivity_args)
            indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
            values = torch.FloatTensor(adj.data)
            shape = torch.Size(adj.shape)
            self.adj = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        elif isinstance(self.adj, sp.spmatrix):
            indices = torch.LongTensor(np.vstack((self.adj.row, self.adj.col)))
            values = torch.FloatTensor(self.adj.data)
            shape = torch.Size(self.adj.shape)
            self.adj = torch.sparse_coo_tensor(indices, values, shape).coalesce()
            
    def make_dataloaders(self):
        self.dataloaders = {'train': None, 'test': None, 'val': None}
        self.adj_per_batch = {'train': [], 'test': [], 'val': []}
        if not self.split_data:
            self.dataloaders['train'] = DataLoader(TensorDataset(self.pcs, torch.IntTensor(range(self.pcs.shape[0]))), batch_size = self.batch_size, shuffle = True)
        else:
            train_idx = []
            test_idx = []
            validation_idx = []
            if self.external_indexing is None:
                train_idx, test_idx = train_test_split(list(range(self.features.shape[0])), test_size = self.test_size, random_state = self.random_state)
                train_idx, validation_idx = train_test_split(train_idx, test_size = self.validation_size, random_state = self.random_state)
            elif 'train' in self.external_indexing and 'test' in self.external_indexing and 'validation' in self.external_indexing:
                train_idx = self.external_indexing['train']
                validation_idx = self.external_indexing['validation']
                test_idx = self.external_indexing['test']
            else:
                print('External indexing requires "train", "test", and "validation" keys given in dictionary. Defaulting to random 60/20/20 train/validation/test split')
                train_idx, test_idx = train_test_split(list(range(self.features.shape[0])), test_size = self.test_size, random_state = self.random_state)
                train_idx, validation_idx = train_test_split(train_idx, test_size = self.validation_size, random_state = self.random_state)

            self.dataloaders['train'] = DataLoader(TensorDataset(self.pcs[train_idx], torch.IntTensor(train_idx)), batch_size = self.batch_size, shuffle = True)
            self.dataloaders['test'] = DataLoader(TensorDataset(self.pcs[test_idx], torch.IntTensor(test_idx)), batch_size = self.batch_size, shuffle = True)
            self.dataloaders['val'] = DataLoader(TensorDataset(self.pcs[validation_idx], torch.IntTensor(validation_idx)), batch_size = self.batch_size, shuffle = True)

    def train(self):
        self.mlp = MLP(input_shape = self.npca, output_shape = self.nembedding).to(self.device)

        self.history = {
            'training_loss': [],
            'validation_loss': [],
            'reduce_lr': defaultdict(int),
            'best_epoch': -1,
            'best_loss': -1,
        }
        
        early_stopping = EarlyStopping(**self.early_stopping_args)
        optimizer = optim.Adam(self.mlp.parameters(), lr = self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor = 0.25,
            patience = self.early_stopping_args['patience']//2 if 'patience' in self.early_stopping_args else 5,
            threshold = self.early_stopping_args['delta'] if 'delta' in self.early_stopping_args else 0.005,
            threshold_mode = 'abs',
            min_lr = 0.0001,
        )
        
        current_lr = scheduler.get_last_lr()[0]
        self.history['reduce_lr'][0] = current_lr

        training_loss = []
        validation_loss = []
        torch.manual_seed(self.random_state) # Set seed here for consistent batching among modalities
        for epoch in (pbar := tqdm(range(self.epochs))):
            pbar.set_description(f'Processing {self.name}, best score {early_stopping.best_score if early_stopping.best_score is not None else 0:.4f}, early stopping count {early_stopping.counter}')
            
            # Training
            self.mlp.train()
            epoch_train_loss = 0.0
            for n, batch in enumerate(self.dataloaders['train']):
                optimizer.zero_grad()
                x = batch[0].to(self.device)
                
                x_hat = self.mlp(x)
                Y1 = self.mlp.get_embeddings(x)
                loss1 = nn.MSELoss()(x, x_hat)
                adj_batch = slice_sparse_coo_tensor(self.adj, batch[1])
                loss2 = sc_loss(adj_batch.to(self.device), Y1)
                loss = loss1 + loss2

                epoch_train_loss += loss.item() * x.shape[0]
                loss.backward()
                optimizer.step()
                
            training_loss.append(epoch_train_loss / len(self.dataloaders['train'].dataset))
            
            # If using eval set, run eval to calculate loss
            if self.dataloaders['val'] is not None:
                self.mlp.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for n, batch in enumerate(self.dataloaders['val']):
                        x = batch[0].to(self.device)
                        
                        x_hat = self.mlp(x)
                        Y1 = self.mlp.get_embeddings(x)
                        
                        loss1 = nn.MSELoss()(x, x_hat)
                        adj_batch = slice_sparse_coo_tensor(self.adj, batch[1])
                        loss2 = sc_loss(adj_batch.to(self.device), Y1)
                        loss = loss1 + loss2
                        
                        epoch_val_loss += loss.item() * x.shape[0]
                        
                validation_loss.append(epoch_val_loss / len(self.dataloaders['val'].dataset))
                scheduler.step(validation_loss[-1])
                early_stopping(validation_loss[-1], self.mlp)
                
            # Otherwise use test data for early stopping
            else:
                scheduler.step(training_loss[-1])
                early_stopping(training_loss[-1], self.mlp)
                
            if current_lr != scheduler.get_last_lr()[0]:
                tqdm.write(f"Learning rate reduced from {current_lr:.5f} to {scheduler.get_last_lr()[0]:.5f} after {epoch} epochs")
                current_lr = scheduler.get_last_lr()[0]
                self.history['reduce_lr'][epoch] = current_lr
                
            if early_stopping.early_stop:
                tqdm.write(f"Early stopping after {epoch} epochs, best loss = {-1*early_stopping.best_score:.4f}")
                break
            
        early_stopping.load_best_model(self.mlp)
        self.history['training_loss'] = training_loss
        self.history['validation_loss'] = validation_loss
        self.history['best_epoch'] = early_stopping.best_epoch
        self.history['best_loss'] = -1 * early_stopping.best_score
        
    def save_loss(self, out_dir = ''):
        if self.history is None:
            print(f"cannot save loss information for {self.name}. Run model.train() first")
            return
        np.save(f'{out_dir}/training_loss_{self.name}.npy', self.history['training_loss'])
        np.save(f'{out_dir}/validation_loss_{self.name}.npy', self.history['validation_loss'])

        plt.plot(self.history['training_loss'], label = 'Training')
        plt.plot(self.history['validation_loss'], label = 'Validation')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Reconstruction + Spectral)")
        for n, epoch in enumerate(list(sorted(self.history['reduce_lr'].keys()))[1:]):
            if n == 0:
                plt.axvline(x = epoch, linestyle = '--', linewidth = 0.5, color = 'black', label = "Reduced learning rate")
            else:
                plt.axvline(x = epoch, linestyle = '--', linewidth = 0.5, color = 'black')

        plt.annotate(
            "Best epoch",
            xy=(self.history['best_epoch'], self.history['best_loss']),
            xytext=(self.history['best_epoch'], self.history['best_loss'] + 0.5),
            arrowprops=dict(facecolor='black', headwidth = 4, headlength = 5, width = 1),
            horizontalalignment='center',
            verticalalignment='bottom'
        )
        
        plt.legend(loc = 'upper right')
        plt.savefig(f'{out_dir}/modality_{self.name}_loss_vs_epoch.png')
        plt.close()
        
    def get_embedding(self):
        if self.emb is not None:
            return self.emb
        
        # If we haven't gotten the embeddings, calculate and scale/center
        emb = self.mlp.to('cpu').get_embeddings(self.pcs).cpu().detach().numpy()
        scaler = StandardScaler()
        n = emb.shape[0]
        n_batches = math.ceil(n / self.batch_size)
        for j in range(n_batches):
            partial_size = min(self.batch_size, n - self.batch_size * j)
            partial_x = emb[self.batch_size*j:self.batch_size*j+partial_size]
            scaler.partial_fit(partial_x)
        self.emb = torch.from_numpy(scaler.transform(emb))
        return self.emb