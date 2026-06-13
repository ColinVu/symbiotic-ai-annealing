"""Embedding generation and caching utilities."""

from .clip_embedder import embed_image, embed_frame, embed_image_for_inference
from .cache_manager import (
    get_cache_path,
    load_from_cache,
    save_to_cache,
    save_frame_to_cache
)

__all__ = [
    'embed_image',
    'embed_frame',
    'embed_image_for_inference',
    'get_cache_path',
    'load_from_cache',
    'save_to_cache',
    'save_frame_to_cache'
]
