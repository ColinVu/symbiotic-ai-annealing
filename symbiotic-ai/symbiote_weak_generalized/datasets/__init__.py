"""Dataset management utilities."""

from .scanner import scan_dataset, build_image_to_label_mapping, load_all_cached_embeddings
from .splitter import stratified_split
from .embedding_dataset import EmbeddingDataset

__all__ = [
    'scan_dataset',
    'build_image_to_label_mapping',
    'load_all_cached_embeddings',
    'stratified_split',
    'EmbeddingDataset'
]
