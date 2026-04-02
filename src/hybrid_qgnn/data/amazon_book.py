"""Amazon-Book benchmark (same on-disk layout as other LightGCN-style folders)."""

from __future__ import annotations

from hybrid_qgnn.data.interactions import load_lightgcn_interaction_dir


def load_amazon_book_dir(data_dir: str = "dataset/amazon-book"):
    """Backward-compatible alias for :func:`load_lightgcn_interaction_dir`."""
    return load_lightgcn_interaction_dir(data_dir)
