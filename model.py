import torch
from torch import nn


class MildenhallNeRF(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        include_view_direction=True,
        num_density_layers=1,
        num_color_layers=2,
        device="cpu",
    ):
        super().__init__()
        point_levels = 16
        point_features = 2
        self.point_encoder = PointEncoder3(
            bb_min=torch.tensor([-5, -5, -5], dtype=torch.float32, device=device),
            bb_max=torch.tensor([5, 5, 5], dtype=torch.float32, device=device),
            number_of_levels=point_levels,
            max_entries_per_level=2**14,
            feature_dim=point_features,
            device=device)
        point_dim = point_levels * point_features
        view_dim = 3 if include_view_direction else 0
        density_layers = []
        density_layers.append(nn.Linear(point_dim, hidden_dim))
        density_layers.append(nn.ReLU())
        for i in range(num_density_layers - 1):
            density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU())
        density_layers.append(nn.Linear(hidden_dim, hidden_dim // 4))
        self.density_mlp = nn.Sequential(*density_layers)
        color_layers = []
        color_layers.append(nn.Linear(view_dim + hidden_dim // 4, hidden_dim))
        color_layers.append(nn.ReLU())
        for i in range(num_color_layers - 1):
            color_layers.append(nn.Linear(hidden_dim, hidden_dim))
            color_layers.append(nn.ReLU())
        color_layers.append(nn.Linear(hidden_dim, 3))
        self.color_mlp = nn.Sequential(*color_layers)

    def forward(self, points_and_view_directions):
        points = self.point_encoder(points_and_view_directions[:, :3])
        view_directions = points_and_view_directions[:, 3:]
        h = self.density_mlp(points)
        density = torch.sigmoid(h[:, 0]).unsqueeze(-1)
        h_rest = torch.nn.functional.relu(h[:, 1:])
        h = torch.cat([density, h_rest, view_directions], dim=-1)
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


class PointEncoder3(nn.Module):
    def __init__(self, bb_min, bb_max, number_of_levels=16, max_entries_per_level=2**14, feature_dim=2, coarsest_resolution=16, finest_resolution=512, device="cpu"):
        super().__init__()
        self.number_of_levels = number_of_levels
        self.max_entries_per_level = max_entries_per_level
        self.feature_dim = feature_dim
        embeddings = []
        for i in range(number_of_levels):
            embedding = nn.Embedding(max_entries_per_level, feature_dim)
            nn.init.uniform_(embedding.weight.data, -1e-4, 1e-4)
            embeddings.append(embedding)
        self.layer_embeddings = nn.ModuleList(embeddings)
        n_min = coarsest_resolution
        n_max = finest_resolution
        self.b = torch.exp((torch.log(torch.tensor(n_max)) - torch.log(torch.tensor(n_min))) / (torch.tensor(number_of_levels) - 1)).tolist()
        self.n = [int(n_min * self.b**i) for i in range(number_of_levels)]
        self.primes = torch.tensor([1, 2_654_435_761, 805_459_861], dtype=torch.long, device=device)
        # self.register_buffer("primes", self.primes)
        self.vertex_offsets = []
        for i in range(0, 8):
            vertex_offsets = []
            for j in range(0, 3):
                vertex_offsets.append((i >> (2-j)) & 1)
            self.vertex_offsets.append(vertex_offsets)
        self.vertex_offsets = torch.tensor(self.vertex_offsets, dtype=torch.long, device=device)
        # self.register_buffer("vertex_offsets", self.vertex_offsets)
        self.bb_size = bb_max - bb_min
        self.bb_min = bb_min

    def forward(self, x):
        """
        * `x` shape: [batch_size, 3]
        * output shape: [batch_size, number_of_levels * feature_dim]
        """
        batch_size = x.shape[0]
        scaled_x = (x - self.bb_min) / self.bb_size
        clamped_x = scaled_x.clamp(0, 1 - 1e-6)
        output = []
        for l in range(self.number_of_levels):
            n_l = self.n[l]
            x_l = clamped_x * n_l
            x_l_floor = torch.floor(x_l).long()
            vertex_offsets = self.vertex_offsets.unsqueeze(0).expand(batch_size, 8, 3)
            vertices = x_l_floor.unsqueeze(1).expand(batch_size, 8, 3) + vertex_offsets
            if (n_l + 1)**3 <= self.max_entries_per_level:
                vertex_indices = vertices[:, :, 0] * (n_l + 1)**2 + vertices[:, :, 1] * (n_l + 1) + vertices[:, :, 2]
            else:
                primed = vertices * self.primes.unsqueeze(0).unsqueeze(0).expand(batch_size, 8, 3)
                xor = primed[:, :, 0] ^ primed[:, :, 1] ^ primed[:, :, 2]
                vertex_indices = torch.remainder(xor, self.max_entries_per_level)
            w_l = x_l - x_l_floor
            vertex_weights3 = \
                vertex_offsets * w_l.unsqueeze(1).expand(batch_size, 8, 3) + \
                (1 - vertex_offsets) * (1 - w_l.unsqueeze(1).expand(batch_size, 8, 3))
            vertex_weights = vertex_weights3.prod(dim=2)
            vertex_features = self.layer_embeddings[l](vertex_indices)
            layer_output = (vertex_features * vertex_weights.unsqueeze(2).expand(batch_size, 8, self.feature_dim)).sum(dim=1)
            output.append(layer_output)
        output = torch.cat(output, dim=1)
        return output


if __name__ == "__main__":

    device = "cpu"
    test_points = torch.rand(2, 3, device=device) * 120 - 60
    bb_min = torch.tensor([-50, -50, -50], dtype=torch.float32, device=device)
    bb_max = torch.tensor([50, 50, 50], dtype=torch.float32, device=device)
    encoder = PointEncoder3(bb_min=bb_min, bb_max=bb_max, device=device)
    test_outputs = encoder(test_points)
    print(test_outputs.shape)

    # model = DeepNeRF()
    # print(model)
