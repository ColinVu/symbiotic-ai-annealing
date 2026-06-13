"""Video to state processing pipeline with training support."""

from .process_video import process_video
from .annotate_video import annotate_video
from .train_model import train_hmm, load_training_data
from .generate_embeddings import generate_embeddings_for_videos
from .hmm_model import HandStateHMM, STATE_SYMBOLS, STATE_LABELS
from .eaf_parser import parse_eaf, eaf_to_state_sequence

__all__ = [
    "process_video",
    "annotate_video",
    "train_hmm",
    "load_training_data",
    "generate_embeddings_for_videos",
    "HandStateHMM",
    "STATE_SYMBOLS",
    "STATE_LABELS",
    "parse_eaf",
    "eaf_to_state_sequence",
]

