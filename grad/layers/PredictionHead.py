from torch import nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_horizon):
        super().__init__()

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, forecast_horizon)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x
