import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, device="cpu"):
        super().__init__()

        self.dtype = torch.float32

        self.seq = nn.Sequential(
            nn.LazyLinear(hidden_dim, dtype=self.dtype),
            nn.LayerNorm(hidden_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, e_i):
        """
        This method projects the embeddings from the frozen backbone down to a balanced dimension.

        Args:
            e_i (torch.tensor): Embedding from the frozen backbone
        """
        return self.seq(e_i)
