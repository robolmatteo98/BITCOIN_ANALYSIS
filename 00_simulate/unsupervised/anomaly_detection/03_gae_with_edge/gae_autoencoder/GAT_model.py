"""Modello GAE con encoder GAT edge-aware."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, edge_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        self.We = nn.Parameter(torch.randn(edge_dim, output_dim) * 0.1)
        self.attention_source = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.attention_target = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.attention_edge = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, X: torch.Tensor, P: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        H = X @ self.W
        edge_messages = torch.einsum("ije,eh->ijh", edge_features, self.We)
        source_scores = H @ self.attention_source
        target_scores = H @ self.attention_target
        edge_scores = torch.einsum("ijh,h->ij", edge_messages, self.attention_edge)

        # P[dest, source]: ogni nodo assegna attenzione ai propri vicini.
        scores = F.leaky_relu(
            target_scores.unsqueeze(1) + source_scores.unsqueeze(0) + edge_scores,
            negative_slope=0.2,
        )
        scores = scores.masked_fill(P <= 0, float("-inf"))
        attention = torch.softmax(scores, dim=1)
        messages = H.unsqueeze(0) + edge_messages
        return torch.sum(attention.unsqueeze(2) * messages, dim=1) + self.bias


class EdgeAwareGATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, edge_dim: int):
        super().__init__()
        self.layer1 = EdgeAwareGATLayer(input_dim, hidden_dim, edge_dim)
        self.layer2 = EdgeAwareGATLayer(hidden_dim, latent_dim, edge_dim)

    def forward(self, X, P, edge_features):
        H = F.elu(self.layer1(X, P, edge_features))
        return F.elu(self.layer2(H, P, edge_features))


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

    def forward(self, Z):
        return Z @ self.W @ Z.T


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, edge_dim: int):
        super().__init__()
        self.encoder = EdgeAwareGATEncoder(input_dim, hidden_dim, latent_dim, edge_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def forward(self, X, P, edge_features):
        Z = self.encoder(X, P, edge_features)
        return Z, self.decoder(Z)
