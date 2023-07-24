import torch
from torch import nn


class DeepNeRF(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.input1 = nn.Linear(6, hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input2 = nn.Linear(6, hidden_dim)
        self.mlp2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, points_and_view_directions):
        h1 = self.mlp1(self.input1(points_and_view_directions))
        h = torch.cat([h1, self.input2(points_and_view_directions)], dim=-1)
        h = self.mlp2(h)
        density_and_color = torch.sigmoid(h)
        return density_and_color


if __name__ == "__main__":
    model = DeepNeRF()
    print(model)
