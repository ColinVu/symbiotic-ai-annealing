"""State detection for hand tracking in videos.

This module provides a framework for detecting hand states (PICK, CARRY_WITH, PLACE, CARRY_EMPTY)
from video frames. The current implementation is a placeholder that returns all frames as CARRY_WITH.

Future implementation will analyze CLIP embeddings to classify hand states.
"""

import enum
from typing import List
import numpy as np
import pandas as pd


class HandState(enum.Enum):
    """Hand states for object manipulation."""
    PICK = "PICK"
    CARRY_WITH = "CARRY_WITH"
    PLACE = "PLACE"
    CARRY_EMPTY = "CARRY_EMPTY"


def detect_states_from_video(
    video_path: str,
    embeddings: List[np.ndarray],
    frame_numbers: List[int],
    fps: float
) -> pd.DataFrame:
    """
    Detect hand states for each frame in video.
    
    PLACEHOLDER IMPLEMENTATION: Currently returns all frames as CARRY_WITH state.
    This pass-through behavior means all frames are accepted for training.
    
    Future implementation will analyze CLIP embeddings and video frames to classify
    hand states, enabling filtering of training data to only frames where the hand
    is carrying an object (CARRY_WITH state).
    
    Args:
        video_path: Path to video file (currently unused in placeholder)
        embeddings: CLIP embeddings for each frame (currently unused in placeholder)
        frame_numbers: Frame numbers corresponding to embeddings
        fps: Video frames per second for timestamp calculation
    
    Returns:
        DataFrame with columns:
        - timestamp_start: Start time in seconds
        - timestamp_end: End time in seconds  
        - state: HandState value as string
        
    Example output (placeholder):
        timestamp_start  timestamp_end  state
        0.0              10.0           CARRY_WITH
    """
    # PLACEHOLDER: Return single CARRY_WITH state for entire video
    # This means ALL frames will be accepted (no filtering)
    
    if len(frame_numbers) == 0:
        # Empty video, return empty DataFrame with correct columns
        return pd.DataFrame(columns=['timestamp_start', 'timestamp_end', 'state'])
    
    # Calculate time range based on first and last frame
    start_time = frame_numbers[0] / fps if fps > 0 else 0.0
    end_time = frame_numbers[-1] / fps if fps > 0 else 0.0
    
    # Return single row: entire video is CARRY_WITH state
    return pd.DataFrame([{
        'timestamp_start': start_time,
        'timestamp_end': end_time,
        'state': HandState.CARRY_WITH.value
    }])


__all__ = ['HandState', 'detect_states_from_video']
