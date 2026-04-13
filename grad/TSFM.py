import torch
from layers.Adapter import Adapter
from layers.GraphModule import GraphModule
from layers.PredictionHead import PredictionHead
from layers.SpatioGraph import SpatioGraph
from torch import nn


class TSFM(nn.Module):
    def __init__(
        self,
        data_path,
        context_length,
        patch_len,
        index_col="date",
        corr_period=30,
        threshold=0.3,
        device="cpu",
        forecast_horizon=1,
    ):

        super().__init__()

        self.data_path = data_path
        self.index_col = index_col
        self.corr_period = corr_period
        self.device = device
        self.context_length = context_length
        self.patch_len = patch_len
        self.seq_len = context_length // patch_len
        self.forecast_horizon = forecast_horizon

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            torch.set_default_dtype(torch.long)
        elif device == "mps" and torch.backends.mps.is_available():
            self.device == "mps"
            torch.set_default_dtype(torch.float32)
        else:
            self.device = "cpu"
            torch.set_default_dtype(torch.long)

        # Bunları parametre olarak vermeyi tercih ettim. Generalization bakış açısından bakarak bunu yapıyorum
        # İleriki zamana göre bu kısmı aktif etmek gerekebilir.
        # ------------------------------------------------

        # self.embedding_generator = EmbeddingGenerator(
        #     data_path= data_path,
        #     batch_size= batch_size,
        #     context_length= context_length,
        #     patch_len= patch_len
        # )

        # Embedding Shape: (32, 384)

        # Embedding Shape: (32, 384)

        self.spatio_graph = SpatioGraph(
            data_path=self.data_path,
            index_col=self.index_col,
            corr_period=self.corr_period,
        )

        flat_dim = self.seq_len * 384

        self.graph_module = GraphModule(
            input_dim=flat_dim, hidden_dim=128
        ).to(self.device)

        self.adapter = Adapter(
            embedding_dim=flat_dim, hidden_dim=128
        ).to(self.device)

        self.PredictionHead = PredictionHead(
            input_dim=flat_dim,
            hidden_dim=128,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        self.graph_dict = self.spatio_graph.get_pyg_edges(threshold=threshold)

    def forward(self, e_i, edge_index, edge_weight):
        """

        Forwards the embeddings with correlation edge_index's and edge weights

        Args:
            embedding (torch.tensor):
            This is the embedding that is going to be passed.
            The shapes of the layers can be adjusted to the embedding's shape.

            For example:
            Moirai Embedding Shape: [32, 384]
        """

        h_i = self.graph_module(e_i, edge_index, edge_weight)

        e_i_2 = self.adapter(e_i, h_i)  # Toplama işlemi kendi içerisinde kullanıyor

        pred = self.PredictionHead(e_i_2)

        return pred
