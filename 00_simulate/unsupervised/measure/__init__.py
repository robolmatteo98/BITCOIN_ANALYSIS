"""Metriche per valutare embedding e raggruppamenti non supervisionati."""

from .silhouette import silhouette_score

__all__ = ["silhouette_score"]
