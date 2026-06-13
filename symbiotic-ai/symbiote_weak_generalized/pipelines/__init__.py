"""High-level training pipelines."""

from .video_training import (
    run_video_training,
    run_multi_video_training,
    run_multi_video_training_from_cache,
    run_incremental_training,
)
# from .image_training import run_training  # Legacy - has import issues
from .video_inference import run_video_inference

__all__ = [
    "run_video_training",
    "run_multi_video_training",
    "run_multi_video_training_from_cache",
    "run_incremental_training",
    # "run_training",  # Legacy
    "run_video_inference",
]
