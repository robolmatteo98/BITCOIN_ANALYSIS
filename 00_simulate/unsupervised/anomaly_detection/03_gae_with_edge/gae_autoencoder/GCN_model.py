"""Modello GAE con encoder GCN edge-aware."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, edge_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1) # casuale piccolo
        self.We1 = nn.Parameter(torch.randn(edge_dim, hidden_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.We2 = nn.Parameter(torch.randn(edge_dim, latent_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

    @staticmethod
    def aggregate_edges(P, edge_features, weights):
        # P[dest, source], edge_features[source, dest, feature].
        return torch.einsum("ij,ije,eh->ih", P, edge_features, weights)

    def forward(self, X, P, edge_features):
        # Messaggio dai nodi e messaggio dalle feature degli archi.
        edge_message_1 = self.aggregate_edges(P, edge_features, self.We1)
        H1 = F.relu(P @ X @ self.W1 + edge_message_1 + self.b1)
        # Il secondo livello lavora sulla rappresentazione intermedia H1.
        edge_message_2 = self.aggregate_edges(P, edge_features, self.We2)
        return F.relu(P @ H1 @ self.W2 + edge_message_2 + self.b2)


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

    def forward(self, Z):
        return Z @ self.W @ Z.T


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, edge_dim: int):
        super().__init__()
        self.encoder = EdgeAwareGCNEncoder(input_dim, hidden_dim, latent_dim, edge_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def forward(self, X, P, edge_features):
        # Encoder edge-aware seguito dal decoder bilineare.
        Z = self.encoder(X, P, edge_features)
        return Z, self.decoder(Z)
