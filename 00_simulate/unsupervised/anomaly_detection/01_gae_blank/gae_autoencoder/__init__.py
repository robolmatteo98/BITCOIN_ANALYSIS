"""Componenti del graph autoencoder senza edge features."""

from .GCN_model import BilinearDecoder, GCNEncoder, GraphAutoencoder

__all__ = ["BilinearDecoder", "GCNEncoder", "GraphAutoencoder"]
