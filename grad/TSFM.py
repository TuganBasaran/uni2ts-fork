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

        self.graph_module = GraphModule(
            input_dim=10,
            hidden_dim=256,  # input dim = column sayısı
        ).to(self.device)

        self.adapter = Adapter(embedding_dim=12288, hidden_dim=256).to(self.device)

        self.PredictionHead = PredictionHead(
            input_dim=512,  # 256 from Graph + 256 from Moirai
            hidden_dim=128,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        self.graph_dict = self.spatio_graph.get_pyg_edges(threshold=threshold)

    def forward(self, e_i, x, edge_index, edge_weight, target_idx, batch_ptr):
        """
        Forwards the embeddings with correlation edge_index's and edge weights.
        """
        # GraphModule tüm node'ları (11 adet) işler
        h_i_all = self.graph_module(x, edge_index, edge_weight)

        # PyG batch içinden sadece hedefin (target) indekslerini buluyoruz
        target_indices = batch_ptr[:-1] + target_idx.flatten()

        # Sadece hedefe ait olan h_i satırlarını çekiyoruz
        h_i_target = h_i_all[target_indices]

        # Moirai embedding'i 256'ya sıkıştırıyoruz
        e_i_projected = self.adapter(e_i)

        # Grafiğin 256'lık çıktısı ve Moirai'nin 256'lık çıktısını birleştiriyoruz
        x_concat = torch.cat([e_i_projected, h_i_target], dim=-1)

        pred = self.PredictionHead(x_concat)

        return pred
