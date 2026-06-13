"""Extract 29-D feature vectors from video frames for HTK HMM state detection.

Feature vector layout (29D):
    [0-1]   hand_center (x_norm, y_norm)
    [2-3]   velocity    (vx, vy)
    [4-5]   acceleration (ax, ay)
    [6-9]   bounding box (width_norm, height_norm, delta_width, delta_height)
    [10-12] hand orientation cross product (ox, oy, oz) normalised
    [13]    object confidence (from CLIP classifier)
    [14-25] hand HSV color: 8-bin hue + 4-bin saturation histograms (L1-normalised)
    [26]    ARUCO signed bin context (pick minus place), scaled
    [27]    ARUCO pick-proximity channel, scaled
    [28]    ARUCO place-proximity channel, scaled
"""

import os
import sys
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from ..lib.hand_detection import hand_pos, hand_bounding_box

from .aruco_detection import ArucoDetector
from .config import DEFAULT_HTK_CONFIG


class FeatureExtractor:
    """Extract HMM feature vectors from video frames.

    Reuses the existing ``hand_detection`` library for hand position and
    bounding box, and the existing CLIP pipeline for object confidence.
    """

    FEATURE_DIM = DEFAULT_HTK_CONFIG.feature_dim
    # Stable indices for downstream heuristics (two_stage fallback, docs)
    IDX_COLOR_START = 14
    IDX_ARUCO_SIGNED = 26
    IDX_ARUCO_PICK = 27
    IDX_ARUCO_PLACE = 28

    def __init__(
        self,
        aruco_detector: ArucoDetector,
        clip_model=None,
        clip_processor=None,
        recognizer=None,
        orientation_weight: float = 8.0,
        orientation_smoothing: int = 5,
        orientation_change_threshold: float = 0.3,
        orientation_up_threshold: float = 0.35,
        orientation_down_threshold: float = 0.2,
        aruco_weight: float = 10.0,
        aruco_deadband: float = 0.05,
        aruco_persistence_frames: int = 0,
        aruco_smoothing_window: int = 1,
        feature_mask: Optional[List[int]] = None,
        use_object_confidence: bool = False,
    ):
        """Initialise.

        Args:
            aruco_detector: An initialised ``ArucoDetector`` instance.
            clip_model: A loaded CLIP ``AutoModel`` (optional -- needed for
                object-confidence features).
            clip_processor: A loaded CLIP ``AutoProcessor``.
            recognizer: An ``ObjectRecognizer`` instance.  If provided it is
                used for object-confidence instead of the raw CLIP model.
        """
        self.aruco_detector = aruco_detector
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.recognizer = recognizer
        # Explicit feature weighting knobs.
        # Orientation is strongly amplified to separate
        # (PICK/PLACE) vs (CARRY_WITH/CARRY_EMPTY).
        self.orientation_weight = float(orientation_weight)
        # Orientation temporal stabilization knobs (aligned with cponly behavior).
        self.orientation_smoothing = max(1, int(orientation_smoothing))
        self.orientation_change_threshold = float(orientation_change_threshold)
        self.orientation_up_threshold = float(orientation_up_threshold)
        self.orientation_down_threshold = float(orientation_down_threshold)
        # ARUCO context is strongly amplified to separate PICK vs PLACE.
        self.aruco_weight = float(aruco_weight)
        # Ignore tiny ARUCO fluctuations so non-pick/place periods aren't biased.
        self.aruco_deadband = float(aruco_deadband)
        # Old pipeline compatibility: persist latest non-zero ARUCO value.
        self.aruco_persistence_frames = max(0, int(aruco_persistence_frames))
        # Optional temporal smoothing for ARUCO channels.
        self.aruco_smoothing_window = max(1, int(aruco_smoothing_window))
        # Optional feature mask: keep selected dims, zero the rest.
        if feature_mask is None:
            self.feature_mask = None
        else:
            self.feature_mask = sorted(set(int(i) for i in feature_mask if 0 <= int(i) < self.FEATURE_DIM))
        # Object confidence is disabled by default (currently constant/noisy).
        self.use_object_confidence = bool(use_object_confidence)

        # Temporal state for velocity / acceleration across frames
        self._prev_hand_pos: Optional[np.ndarray] = None
        self._prev_velocity: Optional[np.ndarray] = None
        self._prev_bbox_size: Optional[np.ndarray] = None
        self._prev_frame_time: Optional[float] = None
        self._orientation_window: deque[np.ndarray] = deque(maxlen=self.orientation_smoothing)
        self._prev_orientation: Optional[np.ndarray] = None
        self._orientation_up: bool = False
        self._aruco_last_nonzero_frame: Optional[int] = None
        self._aruco_last_nonzero_triple: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._aruco_window: Deque[Tuple[float, float, float]] = deque(maxlen=self.aruco_smoothing_window)
        self._feature_step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset temporal state (call between videos)."""
        self._prev_hand_pos = None
        self._prev_velocity = None
        self._prev_bbox_size = None
        self._prev_frame_time = None
        self._orientation_window = deque(maxlen=self.orientation_smoothing)
        self._prev_orientation = None
        self._orientation_up = False
        self._aruco_last_nonzero_frame = None
        self._aruco_last_nonzero_triple = (0.0, 0.0, 0.0)
        self._aruco_window = deque(maxlen=self.aruco_smoothing_window)
        self._feature_step_count = 0

    def extract_frame_features(
        self,
        frame_rgb: np.ndarray,
        hand_landmarks: List[List[float]],
        segmented_hand: Optional[np.ndarray],
        frame_time: float,
    ) -> np.ndarray:
        """Extract a feature vector from a single frame.

        Args:
            frame_rgb: Full RGB frame.
            hand_landmarks: 21x3 list of MediaPipe hand landmarks
                (normalised ``[x, y, z]``).
            segmented_hand: Cropped hand image (for CLIP confidence / color).
                Can be ``None`` if no recogniser is available.
            frame_time: Timestamp of the current frame in seconds.

        Returns:
            ``np.ndarray`` of shape ``(FEATURE_DIM,)``.
        """
        # 1. Hand centre position (normalised 0-1) --------------------- 2D
        hand_center = self._compute_hand_center(hand_landmarks, frame_rgb.shape)
        hand_center_norm = np.array([
            hand_center[0] / frame_rgb.shape[1],
            hand_center[1] / frame_rgb.shape[0],
        ])

        # 2-3. Velocity & acceleration --------------------------------- 4D
        velocity, acceleration = self._compute_motion(
            hand_center_norm, frame_time
        )

        # 4. Bounding box features ------------------------------------- 4D
        bbox_features = self._compute_bbox_features(hand_landmarks, frame_rgb.shape)

        # 5. Hand orientation cross product ----------------------------- 3D
        orientation_raw = self._compute_hand_orientation(hand_landmarks)
        orientation = self._stabilize_orientation(orientation_raw)
        orientation = orientation * self.orientation_weight

        # 6. Object confidence ------------------------------------------ 1D
        obj_conf = self._compute_object_confidence(segmented_hand) if self.use_object_confidence else 0.0

        # 7. Compact HSV color on hand crop ---------------------------- 12D
        color_hs = self._compute_hand_color_hs_histogram(segmented_hand)

        # 8. ARUCO pick / place / signed (persistence + smoothing) ----- 3D
        pick_raw, place_raw, signed_raw = self.aruco_detector.compute_bin_context_channels(
            frame_rgb, hand_center
        )
        self._feature_step_count += 1
        frame_idx = self._feature_step_count
        aruco_active = (
            max(float(pick_raw), float(place_raw), abs(float(signed_raw))) >= self.aruco_deadband
        )
        if aruco_active:
            self._aruco_last_nonzero_frame = frame_idx
            self._aruco_last_nonzero_triple = (float(pick_raw), float(place_raw), float(signed_raw))
            filled_pick, filled_place, filled_signed = self._aruco_last_nonzero_triple
        elif (
            self.aruco_persistence_frames > 0
            and self._aruco_last_nonzero_frame is not None
            and (frame_idx - self._aruco_last_nonzero_frame) <= self.aruco_persistence_frames
        ):
            filled_pick, filled_place, filled_signed = self._aruco_last_nonzero_triple
        else:
            filled_pick = filled_place = filled_signed = 0.0

        self._aruco_window.append((filled_pick, filled_place, filled_signed))
        if self._aruco_window:
            arr = np.array(self._aruco_window, dtype=np.float64)
            smoothed_pick = float(np.mean(arr[:, 0]))
            smoothed_place = float(np.mean(arr[:, 1]))
            smoothed_signed = float(np.mean(arr[:, 2]))
        else:
            smoothed_pick = smoothed_place = smoothed_signed = 0.0

        if smoothed_pick < self.aruco_deadband:
            smoothed_pick = 0.0
        if smoothed_place < self.aruco_deadband:
            smoothed_place = 0.0
        if abs(smoothed_signed) < self.aruco_deadband:
            smoothed_signed = 0.0

        aruco_signed = float(
            np.clip(smoothed_signed * self.aruco_weight, -self.aruco_weight, self.aruco_weight)
        )
        aruco_pick = float(np.clip(smoothed_pick * self.aruco_weight, 0.0, self.aruco_weight))
        aruco_place = float(np.clip(smoothed_place * self.aruco_weight, 0.0, self.aruco_weight))

        features = np.concatenate([
            hand_center_norm,       # 2
            velocity,               # 2
            acceleration,           # 2
            bbox_features,          # 4
            orientation,            # 3
            [obj_conf],             # 1
            color_hs,               # 12
            [aruco_signed, aruco_pick, aruco_place],  # 3
        ])
        if self.feature_mask is not None:
            masked = np.zeros_like(features)
            masked[self.feature_mask] = features[self.feature_mask]
            features = masked
        return features

    def extract_video_features(
        self,
        video_path: str,
        frame_skip: int = 4,
        blur_threshold: float = 100.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[int], float]:
        """Extract features for an entire video.

        Returns:
            ``(features, frame_numbers, fps)`` where *features* has shape
            ``(n_frames, FEATURE_DIM)``.
        """
        from ..preprocessing.blur_detection import is_blurry

        self.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Guard against bogus FPS (0 or negative) from some codecs/cameras.
        if not fps or fps <= 0:
            fps = 30.0
            if verbose:
                print("  [FeatureExtractor] WARNING: invalid FPS from video; defaulting to 30.0")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            print(f"\n[FeatureExtractor] Processing {video_path}")
            print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
            print(f"  Frame skip: {frame_skip}")

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.3,
            max_num_hands=2,
        )

        all_features: List[np.ndarray] = []
        frame_numbers: List[int] = []
        frame_count = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_time = frame_count / fps if fps > 0 else 0.0

            # Downscale 4K (or larger) frames to 1080p so MediaPipe detects
            # the hand reliably regardless of source camera resolution.
            h_orig, w_orig = frame_rgb.shape[:2]
            if w_orig > 1920 or h_orig > 1080:
                scale = min(1920 / w_orig, 1080 / h_orig)
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (int(w_orig * scale), int(h_orig * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            # Detect hand landmarks
            results = hands.process(frame_rgb)
            if not results.multi_hand_landmarks:
                continue

            # Get right-hand landmarks (leftmost in mirrored view)
            hand_points_list = []
            for hand_landmarks_obj in results.multi_hand_landmarks:
                pts = [
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks_obj.landmark[:21]
                ]
                hand_points_list.append(pts)

            if len(hand_points_list) > 1:
                positions = [
                    hand_pos(hp, frame_rgb) for hp in hand_points_list
                ]
                # Keep the same selection rule as video_to_state_cponly:
                # use the rightmost detected hand when two hands appear.
                idx = max(range(len(positions)), key=lambda i: positions[i][0])
                hand_landmarks = hand_points_list[idx]
            else:
                hand_landmarks = hand_points_list[0]

            # Segment hand (for CLIP / blur / color)
            bbox, bbox_size = hand_bounding_box(hand_landmarks, frame_rgb)
            top, bottom = max(0, bbox[0][1]), min(frame_rgb.shape[0], bbox[2][1])
            left, right = max(0, bbox[0][0]), min(frame_rgb.shape[1], bbox[2][0])
            if top >= bottom or left >= right:
                continue
            segmented = frame_rgb[top:bottom, left:right]

            if segmented.size == 0:
                continue
            if is_blurry(segmented, blur_threshold):
                continue

            features = self.extract_frame_features(
                frame_rgb, hand_landmarks, segmented, frame_time
            )
            all_features.append(features)
            frame_numbers.append(frame_count)

        cap.release()
        hands.close()

        if verbose:
            print(f"  Extracted {len(all_features)} feature vectors")

        if len(all_features) == 0:
            return np.empty((0, self.FEATURE_DIM)), [], fps

        return np.array(all_features), frame_numbers, fps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hand_color_hs_histogram(segmented_rgb: Optional[np.ndarray]) -> np.ndarray:
        """8-bin hue + 4-bin saturation histogram on hand crop, L1-normalised (12D)."""
        out = np.zeros(12, dtype=np.float64)
        if segmented_rgb is None or segmented_rgb.size == 0:
            return out
        if segmented_rgb.shape[0] < 2 or segmented_rgb.shape[1] < 2:
            return out
        hsv = cv2.cvtColor(segmented_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [4], [0, 256])
        feat = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        s = float(feat.sum())
        if s > 0:
            feat = feat / s
        return feat.astype(np.float64)

    @staticmethod
    def _compute_hand_center(
        landmarks: List[List[float]], frame_shape: Tuple[int, ...]
    ) -> Tuple[float, float]:
        """Return (x_px, y_px) hand centre using ``hand_pos``."""
        pos = hand_pos(landmarks, np.zeros(frame_shape, dtype=np.uint8))
        if not pos:
            return (0.0, 0.0)
        return pos

    def _compute_motion(
        self, hand_center_norm: np.ndarray, frame_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity and acceleration from normalised position."""
        if self._prev_hand_pos is None or self._prev_frame_time is None:
            self._prev_hand_pos = hand_center_norm.copy()
            self._prev_frame_time = frame_time
            self._prev_velocity = np.zeros(2)
            return np.zeros(2), np.zeros(2)

        dt = frame_time - self._prev_frame_time
        if dt <= 0:
            dt = 1e-6

        velocity = (hand_center_norm - self._prev_hand_pos) / dt
        acceleration = (
            (velocity - self._prev_velocity) / dt
            if self._prev_velocity is not None
            else np.zeros(2)
        )

        self._prev_hand_pos = hand_center_norm.copy()
        self._prev_velocity = velocity.copy()
        self._prev_frame_time = frame_time
        return velocity, acceleration

    def _compute_bbox_features(
        self,
        landmarks: List[List[float]],
        frame_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Return (width_norm, height_norm, d_width, d_height)."""
        dummy_img = np.zeros(frame_shape, dtype=np.uint8)
        _, (h, w) = hand_bounding_box(landmarks, dummy_img)
        width_norm = w / frame_shape[1]
        height_norm = h / frame_shape[0]
        size = np.array([width_norm, height_norm])

        if self._prev_bbox_size is None:
            self._prev_bbox_size = size.copy()
            return np.array([width_norm, height_norm, 0.0, 0.0])

        delta = size - self._prev_bbox_size
        self._prev_bbox_size = size.copy()
        return np.array([width_norm, height_norm, delta[0], delta[1]])

    @staticmethod
    def _compute_hand_orientation(
        landmarks: List[List[float]],
    ) -> np.ndarray:
        """Cross product of (palm->thumb) x (palm->middle finger).

        Uses landmarks 0 (wrist), 4 (thumb tip), 12 (middle finger tip).
        Returns normalised 3-D vector; zeros if degenerate.
        """
        if len(landmarks) < 21:
            return np.zeros(3)

        wrist = np.array(landmarks[0])
        thumb = np.array(landmarks[4])
        middle = np.array(landmarks[12])

        v1 = thumb - wrist   # palm -> thumb
        v2 = middle - wrist  # palm -> middle finger
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return np.zeros(3)
        v1 = v1 / n1
        v2 = v2 / n2

        # Match cponly convention: middle x thumb (direction sign matters).
        cross = np.cross(v2, v1)
        norm = np.linalg.norm(cross)
        if norm < 1e-8:
            return np.zeros(3)
        return cross / norm

    def _stabilize_orientation(self, orientation: np.ndarray) -> np.ndarray:
        """Smooth and gate orientation changes to reduce frame-to-frame jitter."""
        if orientation is None or np.linalg.norm(orientation) < 1e-8:
            orientation = np.zeros(3)
        self._orientation_window.append(orientation)

        valid = [v for v in self._orientation_window if np.linalg.norm(v) >= 1e-8]
        if not valid:
            return self._prev_orientation.copy() if self._prev_orientation is not None else np.zeros(3)

        avg = np.mean(np.vstack(valid), axis=0)
        avg_norm = np.linalg.norm(avg)
        if avg_norm < 1e-8:
            return self._prev_orientation.copy() if self._prev_orientation is not None else np.zeros(3)
        avg = avg / avg_norm

        # Debounce tiny orientation changes (similar intent to cponly gating).
        if self._prev_orientation is not None:
            dot = float(np.clip(np.dot(avg, self._prev_orientation), -1.0, 1.0))
            change = 1.0 - dot
            if change < self.orientation_change_threshold:
                avg = self._prev_orientation.copy()

        # Maintain a hysteresis state for "up" orientation; this stabilizes the
        # sign interpretation around the decision boundary.
        z_dot = float(avg[2])
        if self._orientation_up:
            if z_dot < self.orientation_down_threshold:
                self._orientation_up = False
        else:
            if z_dot > self.orientation_up_threshold:
                self._orientation_up = True

        self._prev_orientation = avg.copy()
        return avg

    def _compute_object_confidence(
        self, segmented_hand: Optional[np.ndarray]
    ) -> float:
        """Return max softmax confidence from CLIP classifier.

        Falls back to 0.0 if no recogniser / model is available.
        """
        if segmented_hand is None:
            return 0.0

        if self.clip_model is not None and self.clip_processor is not None:
            try:
                inputs = self.clip_processor(
                    images=[segmented_hand], return_tensors="pt"
                ).to(self.clip_model.device)
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                return float(torch.sigmoid(features.max()).item())
            except Exception:
                return 0.0

        return 0.0


__all__ = ["FeatureExtractor"]
