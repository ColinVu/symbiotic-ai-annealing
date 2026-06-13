"""Configuration for HTK HMM state detection."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HTKConfig:
    """HTK HMM configuration."""

    # Feature extraction (must match FeatureExtractor.FEATURE_DIM)
    # Layout: 14 motion/orientation/obj + 12 HSV color + 3 ARUCO (signed, pick, place)
    feature_dim: int = 29
    frame_skip: int = 4

    # HMM architecture
    num_states: int = 4  # PICK, CARRY_WITH, PLACE, CARRY_EMPTY
    num_mixtures: int = 3  # Gaussian mixtures per state
    num_training_iterations: int = 10  # Baum-Welch iterations

    # State cycle transition probabilities (initial)
    transition_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ARUCO configuration (DICT_5X5_1000 supports IDs 0-999 using 5x5 markers)
    aruco_dict_type: str = "DICT_5X5_1000"
    aruco_distance_decay: float = 5.0  # Controls weight decay with distance

    # HTK sample period in 100ns units
    # For video at 30fps with frame_skip=4, effective rate ~7.5fps => ~133ms/frame
    # We use 100000 (=10ms) as a conventional HTK base unit
    sample_period: int = 100000

    def __post_init__(self):
        if not self.transition_probs:
            self.transition_probs = {
                "PICK": {"PICK": 0.6, "CARRY_WITH": 0.4},
                "CARRY_WITH": {"CARRY_WITH": 0.6, "PLACE": 0.4},
                "PLACE": {"PLACE": 0.6, "CARRY_EMPTY": 0.4},
                "CARRY_EMPTY": {"CARRY_EMPTY": 0.6, "PICK": 0.3, "EXIT": 0.1},
            }


# Valid state cycle order
STATE_CYCLE = ["PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"]

DEFAULT_HTK_CONFIG = HTKConfig()

__all__ = ["HTKConfig", "DEFAULT_HTK_CONFIG", "STATE_CYCLE"]
