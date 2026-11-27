"""Encoder GCN e decoder bilineare per imparare embedding dei wallet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, X, P):
        H1 = F.relu(_propagate(P, X) @ self.W1 + self.b1)
        return F.relu(_propagate(P, H1) @ self.W2 + self.b2)


def _propagate(P, values):
    """Moltiplica con una matrice dense o sparse senza duplicare la logica GCN."""
    if P.is_sparse:
        return torch.sparse.mm(P, values)
    return P @ values


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

    def forward(self, Z, edge_index):
        src, dst = edge_index
        return torch.sum((Z[src] @ self.W) * Z[dst], dim=1)


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def encode(self, X, P):
        return self.encoder(X, P)

    def score(self, Z, edge_index):
        return self.decoder(Z, edge_index)
