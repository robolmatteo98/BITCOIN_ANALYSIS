"""Compatibilità: il modello GCN edge-aware è definito in GCN_model.py."""

from .GCN_model import BilinearDecoder, EdgeAwareGCNEncoder, GraphAutoencoder

__all__ = ["BilinearDecoder", "EdgeAwareGCNEncoder", "GraphAutoencoder"]
