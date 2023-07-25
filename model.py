import torch
from torch import nn


class MildenhallNeRF(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        include_view_direction=True,
        num_density_layers=2,
        num_color_layers=2,
    ):
        super().__init__()
        input_dim = 6 if include_view_direction else 3
        density_layers = []
        density_layers.append(nn.Linear(3, hidden_dim))
        density_layers.append(nn.ReLU())
        for i in range(num_density_layers - 2):
            density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU())
        density_layers.append(nn.Linear(hidden_dim, hidden_dim // 4))
        self.density_mlp = nn.Sequential(*density_layers)
        color_layers = []
        color_layers.append(nn.Linear(input_dim + hidden_dim // 4 - 1, hidden_dim))
        color_layers.append(nn.ReLU())
        for i in range(num_color_layers - 2):
            color_layers.append(nn.Linear(hidden_dim, hidden_dim))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(hidden_dim, 3))
        self.color_mlp = nn.Sequential(*color_layers)

    def forward(self, points_and_view_directions):
        h = self.density_mlp(points_and_view_directions[:, :3])
        density = torch.sigmoid(h[:, 0]).unsqueeze(-1)
        h = torch.nn.functional.relu(h[:, 1:])
        h = torch.cat([points_and_view_directions, h], dim=-1)
        h = self.color_mlp(h)
        color = torch.sigmoid(h)
        density_and_color = torch.cat([density, color], dim=-1)
        return density_and_color


class DeepNeRF(nn.Module):
    def __init__(self, hidden_dim=256, include_view_direction=True):
        super().__init__()
        input_dim = 6 if include_view_direction else 3
        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input2 = nn.Linear(input_dim, hidden_dim)
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
