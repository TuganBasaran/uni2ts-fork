import torch 
from torch import nn 

class PredictionHead(nn.Module): 
    def __init__(self, seq_len, embedding_dim, hidden_dim, forecast_horizon): 
        super.__init__(self)
        
        self.layer_1 = nn.Linear(seq_len * embedding_dim, 128)
        self.layer_2 = nn.Linear(128, forecast_horizon)
        
    def forward(self, embedding): 
        embedding = embedding.flatten()
        x = self.layer_1(embedding)
        x = nn.ReLU(x)
        x = self.layer_2(x)
        return x 
        