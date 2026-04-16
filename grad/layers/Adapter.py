import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, embedding_dim, graph_dim, bottleneck_dim=64):
        super().__init__()

        # Moirai embedding'i kademeli düşür
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Linear(512, graph_dim),
            nn.GELU(),
        )

        # Her iki branch'i normalize et
        self.norm_temporal = nn.LayerNorm(graph_dim)
        self.norm_spatial = nn.LayerNorm(graph_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, graph_dim),
        )

        # Learnable gate — iki branch'in katkısını dengeler
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, e_i, h_i):
        e_projected = self.projection(e_i)

        # Normalize — ikisini aynı scale'e getir
        e_normed = self.norm_temporal(e_projected)
        h_normed = self.norm_spatial(h_i)

        # Fusion
        combined = torch.cat([e_normed, h_normed], dim=-1)
        adapted = self.fusion(combined)

        # Gated residual — gate 0'a yakınsa GNN baskın, 1'e yakınsa Adapter baskın
        return self.gate * e_normed + (1 - self.gate) * adapted
