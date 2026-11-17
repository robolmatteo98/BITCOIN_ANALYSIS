"""Modello GAE con encoder GCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1) # casuale piccolo
        self.b1 = nn.Parameter(torch.zeros(hidden_dim)) # tutti a zero
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        # Primo livello GCN: aggrega i nodi vicini e applica la trasformazione W1.
        H1 = F.relu(P @ X @ self.W1 + self.b1)
        # Secondo livello GCN: usa la rappresentazione intermedia H1.
        return F.relu(P @ H1 @ self.W2 + self.b2)


class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.1)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.W @ Z.T


class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def forward(self, X, P):
        # Gestita automaticamente dalla libreria: esegue encoder e decoder.
        Z = self.encoder(X, P)
        return Z, self.decoder(Z)
