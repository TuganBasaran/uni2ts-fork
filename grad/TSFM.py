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
        num_columns=10,
        embedding_dim=12288,
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
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        torch.set_default_dtype(torch.float32)

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
            input_dim=5,
            hidden_dim=256,  # input dim = column sayısı
        ).to(self.device)

        self.adapter = Adapter(embedding_dim=embedding_dim, graph_dim=256).to(
            self.device
        )

        self.PredictionHead = PredictionHead(
            input_dim=256,
            hidden_dim=128,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        self.graph_dict = self.spatio_graph.get_pyg_edges(threshold=threshold)

    def forward(self, e_i, node_feature, edge_index, edge_weight, batch_ptr, target_idx):
        # 1. GNN
        h_i = self.graph_module(node_feature, edge_index, edge_weight)

        # 2. Target node'u ayıkla
        batch_size = batch_ptr.size(0) - 1
        target_nodes = []
        for i in range(batch_size):
            start = batch_ptr[i]
            local_target = target_idx[i].item()
            target_nodes.append(h_i[start + local_target])
        target_h_i = torch.stack(target_nodes)

        # 3. Adapter: fusion + residual
        enriched = self.adapter(e_i, target_h_i)

        # 4. Prediction
        pred = self.PredictionHead(enriched)
        return pred
