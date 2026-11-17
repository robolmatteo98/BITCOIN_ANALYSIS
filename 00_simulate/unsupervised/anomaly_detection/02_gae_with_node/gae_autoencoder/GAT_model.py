"""Modello GAE con encoder GAT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.attention_source = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.attention_target = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        H = X @ self.W
        source_scores = H @ self.attention_source
        target_scores = H @ self.attention_target
        scores = F.leaky_relu(
            target_scores.unsqueeze(1) + source_scores.unsqueeze(0),
            negative_slope=0.2,
        )
        scores = scores.masked_fill(P <= 0, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        return attention @ H + self.bias


class GATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.layer1 = GATLayer(input_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, latent_dim)

    def forward(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        H = F.elu(self.layer1(X, P))
        return F.elu(self.layer2(H, P))


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.W @ Z.T


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def forward(self, X, P):
        Z = self.encoder(X, P)
        return Z, self.decoder(Z)
