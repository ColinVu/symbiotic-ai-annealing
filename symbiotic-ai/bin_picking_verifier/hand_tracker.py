"""MediaPipe hand tracking for interaction point extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import VerifierConfig

logger = logging.getLogger(__name__)


def compute_palm_normal(landmarks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Palm normal via cross product in MediaPipe normalized 3D space.

    Vector A: wrist (L0) -> thumb tip (L4)
    Vector B: wrist (L0) -> middle tip (L12)
    n = A x B (right-hand rule). n.z indicates palm orientation vs camera.
    
    Returns: (palm_normal, vec_a, vec_b)
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z], dtype=np.float64)
    thumb = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z], dtype=np.float64)
    middle = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z], dtype=np.float64)
    vec_a = thumb - wrist
    vec_b = middle - wrist
    palm_normal = np.cross(vec_a, vec_b)
    return palm_normal, vec_a, vec_b


def is_palm_facing_down(palm_normal_z: float, cfg: VerifierConfig) -> bool:
    """True when cross-product z component indicates palm aimed toward bins."""
    return cfg.palm_down_normal_z_sign * palm_normal_z > cfg.palm_down_min_abs_z


class HandTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def get_hand_landmarks(self, frame_bgr_1080p: np.ndarray) -> Optional[Dict[str, Any]]:
        """Landmarks + palm normal. Selects RIGHTMOST hand (highest avg x)."""
        h, w = frame_bgr_1080p.shape[:2]
        try:
            rgb = cv2.cvtColor(frame_bgr_1080p, cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb)
        except Exception as e:
            logger.debug("MediaPipe processing failed: %s", e)
            return None

        if not result.multi_hand_landmarks:
            return None

        best_hand = None
        best_avg_x = -1.0

        for hand_landmarks in result.multi_hand_landmarks:
            avg_x = sum(lm.x for lm in hand_landmarks.landmark) / len(hand_landmarks.landmark)
            if avg_x > best_avg_x:
                best_avg_x = avg_x
                best_hand = hand_landmarks

        if best_hand is None:
            return None

        landmarks = best_hand.landmark

        lm_list: List[Tuple[float, float, float]] = []
        for lm in landmarks:
            lm_list.append((float(lm.x * w), float(lm.y * h), float(lm.z)))

        wrist = landmarks[self._mp_hands.HandLandmark.WRIST]
        thumb = landmarks[self._mp_hands.HandLandmark.THUMB_TIP]
        index = landmarks[self._mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle = landmarks[self._mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        wrist_px = (float(wrist.x * w), float(wrist.y * h), float(wrist.z))
        thumb_tip = (float(thumb.x * w), float(thumb.y * h), float(thumb.z))
        index_tip = (float(index.x * w), float(index.y * h), float(index.z))
        middle_tip = (float(middle.x * w), float(middle.y * h), float(middle.z))
        
        tx, ty = thumb_tip[0], thumb_tip[1]
        ix, iy = index_tip[0], index_tip[1]
        interaction_point = ((tx + ix) * 0.5, (ty + iy) * 0.5)

        avg_z = 0.5 * (thumb_tip[2] + index_tip[2])

        palm_normal, vec_a, vec_b = compute_palm_normal(landmarks)
        palm_normal_z = float(palm_normal[2])

        return {
            "landmarks": lm_list,
            "interaction_point": interaction_point,
            "index_tip": index_tip,
            "thumb_tip": thumb_tip,
            "middle_tip": middle_tip,
            "wrist": wrist_px,
            "avg_z": avg_z,
            "palm_normal": palm_normal.tolist(),
            "palm_normal_z": palm_normal_z,
            "cross_vec_a": vec_a.tolist(),
            "cross_vec_b": vec_b.tolist(),
        }

    def get_interaction_point(self, frame_bgr_1080p: np.ndarray) -> Optional[Tuple[float, float]]:
        """Return midpoint between thumb_tip and index_tip in absolute pixels."""
        data = self.get_hand_landmarks(frame_bgr_1080p)
        if data is None:
            return None
        return data["interaction_point"]
