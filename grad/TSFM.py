import torch 
from torch import nn 
from EmbeddingGenerator import EmbeddingGenerator
from layers.SpatioGraph import SpatioGraph
from layers.GraphModule import GraphModule
from layers.Adapter import Adapter
from layers.PredictionHead import PredictionHead



class TSFM(nn.Module): 
    def __init__(self, 
                 data_path,
                 batch_size, 
                 context_length, 
                 patch_len,
                 index_col="date",
                 corr_period= 30,
                 device= "cpu",
                 forecast_horizon= 1 
                 ):
        
        super().__init__() 
        
        self.data_path = data_path
        self.index_col = index_col
        self.batch_size = batch_size
        self.corr_period = corr_period
        self.context_length = context_length
        self.patch_len = patch_len 
        self.seq_len = context_length // patch_len
        self.forecast_horizon= forecast_horizon
        
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available(): 
            self.device == "mps"
            torch.set_default_dtype(torch.float32)
        else:
            self.device = "cpu"
        
        self.embedding_generator = EmbeddingGenerator(
            data_path= data_path, 
            batch_size= batch_size,
            context_length= context_length,
            patch_len= patch_len
        )
        
        # Embedding Shape: (32, 384)
        
        self.spatio_graph = SpatioGraph(
            data_path= self.data_path,
            index_col= self.index_col,
            corr_period= self.corr_period,
            device= self.device
        )
        
        self.graph_module = GraphModule(
            input_dim= (self.seq_len, 384),
            hidden_dim= 128,
            device = self.device
        )
        
        self.adapter = Adapter(
            input_dim = (self.seq_len, 384),
            hidden_dim= 128, 
            device= self.device
        )
        
        self.PredictionHead = PredictionHead(
            seq_len= self.seq_len,
            embedding_dim= 384, 
            hidden_dim= 128,
            forecast_horizon= self.forecast_horizon 
        )

