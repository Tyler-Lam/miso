import torch.nn as nn
import torch
from torch.nn.utils.parametrizations import orthogonal

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(in_features=kwargs["input_shape"], out_features=128)
        self.decoder = nn.Linear(in_features=128, out_features=kwargs["input_shape"])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.decoder(x)
        x = self.relu(x)
        return x

    def get_embeddings(self,x):
        x = self.encoder(x)
        x = self.relu(x)
        return x
        
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