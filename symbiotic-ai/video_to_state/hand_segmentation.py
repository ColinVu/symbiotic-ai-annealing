from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


HandBoundingBox = Tuple[int, int, int, int]  # left, top, right, bottom


def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def _hand_center(landmarks: np.ndarray, image_shape: Tuple[int, int, int]) -> Tuple[float, float]:
    if len(landmarks) < 21:
        return (float("inf"), float("inf"))

    height, width = image_shape[0], image_shape[1]
    base_points = [landmarks[0]] + [landmarks[idx * 4 + 1] for idx in range(5)]

    total_x = sum(point[0] * width for point in base_points)
    total_y = sum(point[1] * height for point in base_points)
    count = len(base_points)

    return total_x / count, total_y / count


def _hand_bounding_box(landmarks: np.ndarray, image_shape: Tuple[int, int, int]) -> HandBoundingBox:
    height, width = image_shape[0], image_shape[1]

    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height

    left = int(np.min(xs) - 20)
    right = int(np.max(xs) + 20)
    top = int(np.min(ys) - 20)
    bottom = int(np.max(ys) + 20)

    left = _clamp(left, 0, width - 1)
    right = _clamp(right, 0, width - 1)
    top = _clamp(top, 0, height - 1)
    bottom = _clamp(bottom, 0, height - 1)

    if left >= right or top >= bottom:
        # fall back to full frame if the box collapses
        return 0, 0, width - 1, height - 1

    return left, top, right, bottom


def _compute_hand_orientation(landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Computes hand orientation using cross product of vectors from palm to fingers.

    MediaPipe landmark indices:
    - Palm (wrist): 0
    - Thumb tip: 4
    - Middle finger tip: 12

    Returns a 3D direction vector (normalized), or None if landmarks are invalid.
    """
    if len(landmarks) < 13:
        return None

    palm = landmarks[0]
    thumb_tip = landmarks[4]
    middle_tip = landmarks[12]

    thumb_vec = thumb_tip - palm
    middle_vec = middle_tip - palm

    thumb_norm = np.linalg.norm(thumb_vec)
    middle_norm = np.linalg.norm(middle_vec)

    if thumb_norm < 1e-6 or middle_norm < 1e-6:
        return None

    thumb_vec /= thumb_norm
    middle_vec /= middle_norm

    orientation = np.cross(middle_vec, thumb_vec)
    orientation_norm = np.linalg.norm(orientation)

    if orientation_norm < 1e-6:
        return None

    return orientation / orientation_norm


@dataclass
class HandCrop:
    image: np.ndarray
    bounding_box: HandBoundingBox
    landmarks: np.ndarray
    orientation: Optional[np.ndarray] = None  # 3D direction vector from cross product


class HandSegmenter:
    """Extracts the most prominent hand from an image using MediaPipe."""

    def __init__(
        self,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.3,
        max_num_hands: int = 2,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_hands=max_num_hands,
        )

    def __enter__(self) -> "HandSegmenter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._hands:
            self._hands.close()

    def extract_hand(self, image_rgb: np.ndarray) -> Optional[HandCrop]:
        """Returns the cropped hand image if one is detected."""
        results = self._hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_points = [
            np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark[:21]])
            for hand_landmarks in results.multi_hand_landmarks
        ]

        if len(hand_points) == 1:
            selected = hand_points[0]
        else:
            centers = [_hand_center(points, image_rgb.shape) for points in hand_points]
            # select the rightmost hand (higher x coordinate)
            right_index = int(np.argmax([center[0] for center in centers]))
            selected = hand_points[right_index]

        left, top, right, bottom = _hand_bounding_box(selected, image_rgb.shape)
        crop = image_rgb[top:bottom, left:right]

        if crop.size == 0:
            return None

        orientation = _compute_hand_orientation(selected)

        return HandCrop(
            image=crop,
            bounding_box=(left, top, right, bottom),
            landmarks=selected,
            orientation=orientation,
        )


def visualize_bounding_box(image_rgb: np.ndarray, bbox: HandBoundingBox) -> np.ndarray:
    """Utility for debugging; returns an RGB image with the bbox drawn."""
    debug_image = image_rgb.copy()
    left, top, right, bottom = bbox
    cv2.rectangle(debug_image, (left, top), (right, bottom), (255, 0, 0), 2)
    return debug_image


