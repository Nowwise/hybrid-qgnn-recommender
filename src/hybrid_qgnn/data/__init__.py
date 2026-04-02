from hybrid_qgnn.data.graph import (
    BPRTripletDataset,
    PairDataset,
    build_norm_adj_from_train_pairs,
    make_bpr_loader,
    make_loaders,
    make_small_implicit_split,
)
from hybrid_qgnn.data.interactions import load_lightgcn_interaction_dir


def load_amazon_book_dir(data_dir: str = "dataset/amazon-book"):
    """Backward-compatible alias for :func:`load_lightgcn_interaction_dir`."""
    return load_lightgcn_interaction_dir(data_dir)


__all__ = [
    "load_amazon_book_dir",
    "load_lightgcn_interaction_dir",
    "make_small_implicit_split",
    "PairDataset",
    "BPRTripletDataset",
    "make_loaders",
    "make_bpr_loader",
    "build_norm_adj_from_train_pairs",
]
