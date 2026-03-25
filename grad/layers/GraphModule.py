import torch
import torch.nn as t_nn
import torch_geometric.nn as nn # type: ignore


class GraphModule(nn.Module): 
    def __init__(self, input_dim, hidden_dim, device= "cpu"): 
        super().__init__()
        
        self.dtype = torch.float32 if device == "mps" else torch.long

        # GCN neden kullandım? 
        # Her inference'da farklı bir graph yapısı verdiğim için static yapısından etkilenmiyorum 
        self.conv_1 = nn.GCNConv(input_dim, hidden_dim, dtype= self.dtype)
        self.conv_2 = nn.GCNConv(hidden_dim, input_dim, dtype= self.dtype)
        self.norm = nn.LayerNorm(input_dim, dtype= self.dtype)
        
    def forward(self, x, edge_index, edge_weight): 
        x = self.conv_1(x, edge_index, edge_weight)
        x = t_nn.ReLU(x)
        x = self.conv_2(x, edge_index, edge_weight)
        x = self.norm(x)
        
        return x 
        
        
        