"""ARUCO marker detection and weighted bin context computation.

Detects ARUCO markers in video frames and computes a weighted bin-context
score used as a single feature in the HTK HMM state detection pipeline.
"""

import json
from typing import Dict, Tuple, Optional
import cv2
import numpy as np


# Map string dictionary names to OpenCV constants
_ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
}


class ArucoDetector:
    """Detect ARUCO markers and compute weighted bin context."""

    def __init__(self, aruco_dict_type: str = "DICT_5X5_1000",
                 distance_decay: float = 5.0):
        """Initialise ARUCO detector.

        Args:
            aruco_dict_type: Name of the ARUCO dictionary (e.g. "DICT_5X5_1000" for IDs 0-999).
            distance_decay: Controls how fast weight decays with distance.
        """
        cv_dict_id = _ARUCO_DICT_MAP.get(aruco_dict_type, cv2.aruco.DICT_5X5_1000)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Bin configuration: {int_id: {"type": "pick"|"place", "object": str}}
        self.bin_config: Dict[int, dict] = {}
        self.distance_decay = distance_decay

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_markers(self, frame: np.ndarray) -> Dict:
        """Detect all ARUCO markers in *frame* (BGR or RGB).

        Returns dict with keys ``ids``, ``centers``, ``types``, ``corners``.
        """
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is None:
            return {"ids": [], "centers": [], "types": [], "corners": []}

        centers = []
        types = []
        for marker_id, corner_set in zip(ids.flatten(), corners):
            center = corner_set[0].mean(axis=0)
            centers.append(center)
            bin_info = self.bin_config.get(int(marker_id), {})
            types.append(bin_info.get("type", "unknown"))

        return {
            "ids": ids.flatten().tolist(),
            "centers": centers,
            "types": types,
            "corners": corners,
        }

    # ------------------------------------------------------------------
    # Weighted bin context
    # ------------------------------------------------------------------

    def compute_bin_context_weight(
        self,
        frame: np.ndarray,
        hand_position: Tuple[float, float],
    ) -> float:
        """Compute weighted bin context score.

        Returns:
            float in ``[-1, 1]``.
            ``+1`` = strong PICK context (near pick bins);
            ``-1`` = strong PLACE context (near place bins);
            ``0``  = neutral / no bins.
        """
        markers = self.detect_markers(frame)
        if len(markers["ids"]) == 0:
            return 0.0

        frame_diag = np.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2)
        weight = 0.0

        for center, bin_type in zip(markers["centers"], markers["types"]):
            if bin_type == "unknown":
                continue
            distance = np.linalg.norm(np.array(hand_position) - np.array(center))
            normalized_dist = distance / frame_diag
            proximity = np.exp(-normalized_dist * self.distance_decay)

            if bin_type == "pick":
                weight += proximity
            elif bin_type == "place":
                weight -= proximity

        return float(np.clip(weight, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Visualization (for test_aruco_detection tool)
    # ------------------------------------------------------------------

    def visualize_bin_context(
        self,
        frame: np.ndarray,
        hand_position: Tuple[float, float],
    ) -> Tuple[np.ndarray, float]:
        """Draw detected markers, hand position, and weight overlay.

        Returns ``(annotated_frame, weight)``.
        """
        markers = self.detect_markers(frame)
        annotated = frame.copy()

        if len(markers["ids"]) > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated, markers["corners"], np.array(markers["ids"])
            )
            for marker_id, center, bin_type in zip(
                markers["ids"], markers["centers"], markers["types"]
            ):
                center_int = tuple(center.astype(int))
                color = (
                    (0, 255, 0) if bin_type == "pick"
                    else (0, 0, 255) if bin_type == "place"
                    else (128, 128, 128)
                )
                cv2.circle(annotated, center_int, 10, color, -1)
                cv2.putText(
                    annotated,
                    f"ID:{marker_id}",
                    (center_int[0] + 15, center_int[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        hand_int = tuple(np.array(hand_position).astype(int))
        cv2.circle(annotated, hand_int, 15, (255, 0, 255), 3)

        weight = self.compute_bin_context_weight(frame, hand_position)
        weight_color = (
            (0, 255, 0) if weight > 0.5
            else (0, 0, 255) if weight < -0.5
            else (0, 255, 255)
        )
        cv2.putText(
            annotated,
            f"Bin Context: {weight:+.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, weight_color, 3,
        )

        bar_length = int(abs(weight) * 200)
        bar_x = 300
        bar_y = 30
        if weight > 0:
            cv2.rectangle(annotated, (bar_x, bar_y - 10),
                          (bar_x + bar_length, bar_y + 10), (0, 255, 0), -1)
            cv2.putText(annotated, "PICK",
                        (bar_x + bar_length + 10, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif weight < 0:
            cv2.rectangle(annotated, (bar_x - bar_length, bar_y - 10),
                          (bar_x, bar_y + 10), (0, 0, 255), -1)
            cv2.putText(annotated, "PLACE",
                        (bar_x - bar_length - 70, bar_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated, weight

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_bin_config(self, config_path: str) -> None:
        """Load ARUCO-to-bin mapping from a JSON file.

        Expected format::

            {
              "marker_dict": "DICT_5X5_1000",
              "bins": { "0": {"type": "pick", "object": "apple"}, ... },
              "distance_decay": 5.0
            }
        """
        with open(config_path, "r") as f:
            config = json.load(f)
        self.bin_config = {}
        for marker_id_str, bin_info in config.get("bins", {}).items():
            self.bin_config[int(marker_id_str)] = bin_info
        if "distance_decay" in config:
            self.distance_decay = float(config["distance_decay"])


__all__ = ["ArucoDetector"]
