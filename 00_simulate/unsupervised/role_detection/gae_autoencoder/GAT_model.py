"""Encoder Graph Attention e decoder bilineare per la role detection v1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .GCN_model import BilinearDecoder


class GATEncoder(nn.Module):
    """GAT a una testa, con attenzione calcolata solo sugli archi esistenti."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.a1_src = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.a1_dst = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 0.1)
        self.a2_src = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self.a2_dst = nn.Parameter(torch.randn(latent_dim) * 0.1)
        self._cached_p_data_ptr = None
        self._cached_edges = None

    @staticmethod
    def _attention_layer(
        X: torch.Tensor,
        edge_index: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        source_attention: torch.Tensor,
        target_attention: torch.Tensor,
    ) -> torch.Tensor:
        """Propaga feature con softmax raggruppato per nodo destinatario."""
        num_nodes = X.shape[0]
        target, source = edge_index
        transformed = X @ weights
        logits = F.leaky_relu(
            transformed[source] @ source_attention
            + transformed[target] @ target_attention,
            negative_slope=0.2,
        )

        max_per_target = torch.full(
            (num_nodes,), -torch.inf, dtype=logits.dtype, device=logits.device
        )
        max_per_target.scatter_reduce_(
            0, target, logits, reduce="amax", include_self=True
        )
        exponentials = torch.exp(logits - max_per_target[target])
        normalizers = torch.zeros(
            num_nodes, dtype=logits.dtype, device=logits.device
        )
        normalizers.scatter_add_(0, target, exponentials)
        coefficients = exponentials / normalizers[target].clamp_min(1e-12)

        output = torch.zeros_like(transformed)
        output.index_add_(
            0, target, coefficients.unsqueeze(1) * transformed[source]
        )
        return output

    def forward(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        # P resta invariata durante il training; riutilizziamo gli archi tra
        # i due layer e tra le epoche, ricalcolandoli solo per un nuovo grafo.
        p_data_ptr = P.data_ptr()
        if self._cached_p_data_ptr != p_data_ptr:
            self._cached_edges = (P > 0).nonzero(as_tuple=True)
            self._cached_p_data_ptr = p_data_ptr

        hidden = self._attention_layer(
            X, self._cached_edges, self.W1, self.a1_src, self.a1_dst
        )
        hidden = F.elu(hidden)
        return F.elu(
            self._attention_layer(
                hidden, self._cached_edges, self.W2, self.a2_src, self.a2_dst
            )
        )


class GraphAttentionAutoencoder(nn.Module):
    """Autoencoder GAT con lo stesso decoder bilineare della GCN."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BilinearDecoder(latent_dim)

    def encode(self, X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        return self.encoder(X, P)

    def score(self, Z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(Z, edge_index)
