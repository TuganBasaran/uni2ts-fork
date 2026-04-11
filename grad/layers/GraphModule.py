import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GraphModule(nn.Module): 
    def __init__(self, input_dim: int, hidden_dim: int, device="cpu"): 
        super().__init__()
        
        # GCN neden kullandım? 
        # Her inference'da farklı bir graph yapısı verdiğim için static yapısından etkilenmiyorum 
        self.conv_1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv_2 = pyg_nn.GCNConv(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim) # Graph Norm? 
        
    def forward(self, x, edge_index, edge_weight=None): 
        x = self.conv_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv_2(x, edge_index, edge_weight)
        x = self.norm(x)
        
        return x 
        
        
        