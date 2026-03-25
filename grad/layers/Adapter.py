import torch 
from torch import nn 

class Adapter(nn.Module): 
    def __init__(self, embedding_dim, hidden_dim, device= "cpu"): 
        super.__init__()
        
        self.dtype = torch.float32 if device == "cpu" else torch.long
        
        
        self.seq = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim, dtype= self.dtype), 
            nn.ReLU(), # Gausian Relu -> GELU?? 
            nn.Linear(hidden_dim, embedding_dim, dtype= self.dtype),
            nn.ReLU() # Buradaki activation layer'a ihtiyaç var mı? 
        )
        
        
    def forward(self, e_i, h_i): 
        """
        This method combines the embeddings from the frozen backbone and graph module 
        and allows us to adapt the new enriched embedding. 

        Args:
            e_i (torch.tensor): Embedding from the frozen backbone
            h_i (_type_): Embedding from the GraphModule
        """
        
        x = torch.cat([e_i, h_i], dim= -1) # e_i + h_i = e_i * 2 
        x = self.seq(x)
        return e_i + x 
        
        
        
        
            
        