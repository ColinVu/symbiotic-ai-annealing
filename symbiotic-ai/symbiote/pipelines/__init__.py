"""High-level training pipelines."""

from .video_training import run_video_training
from .image_training import run_training
from .video_inference import run_video_inference

__all__ = ['run_video_training', 'run_training', 'run_video_inference']
