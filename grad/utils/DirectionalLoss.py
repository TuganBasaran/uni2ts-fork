import torch
from torch import nn


class DirectionalLoss(nn.Module):
    def __init__(self, direction_weight=1.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        magnitude_loss = self.mse(y_pred, y_true)

        # Yön cezası: tahmin ve gerçek farklı yöndeyse cezalandır
        direction_penalty = torch.mean(torch.clamp(-torch.sign(y_true) * y_pred, min=0))

        return magnitude_loss + self.direction_weight * direction_penalty
