#
# Hand Detection helper methods
#

import mediapipe as mp
from matplotlib import pyplot as plt
import cv2
import copy
import numpy as np
from typing import Optional

def hand_pos(landmarks: list[tuple[int, int, int]], image: np.ndarray) -> tuple[int, int]:
    """Determine the location of the center of the palm of a detected hand"""
    if len(landmarks) < 21:
        # need at least 21 for a hand
        return ()

    # center of palm appears to be a better indicator for the location of the hand (more mass concentrated at that point)
    fingers = [[np.array(landmarks[i]) for i in range(j * 4 + 1, j * 4 + 5)] for j in range(5)]
    # base of the palm
    total_x = landmarks[0][0] * image.shape[1]
    total_y = landmarks[0][1] * image.shape[0]
    for finger in fingers:
        # look at base of the finger
        curr = finger[0]
        total_x += (curr[0] * image.shape[1])
        total_y += (curr[1] * image.shape[0])
    return (total_x / 6, total_y / 6)

def hand_bounding_box(
    landmarks: list[tuple[int, int, int]],
    image: np.ndarray,
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]], tuple[int, int]]:
    """Draw a bounding box around a detected hand"""
    left = image.shape[1]
    right = -1
    top = image.shape[0]
    bottom = -1
    for (x, y, _) in landmarks:
        #adjust for pixel values
        x = x * image.shape[1]
        y = y * image.shape[0]
        if x < left:
            # point further left than current left
            left = x
        if x > right:
            # point further right than current right
            right = x
        if y < top:
            #point that is higher than top
            top = y
        if y > bottom:
            bottom = y

    #add a little buffer for the rest of the finger (the landmarks aren't exactly at the tip)
    top, bottom, left, right = int(top - 20), int(bottom + 20), int(left - 20), int(right + 20)
    # return points (top left and clockwise from there, (x, y) for points), size (height, width)
    return ((left, top), (right, top), (right, bottom), (left, bottom)), (bottom - top, right - left)

def draw_hand(image: np.ndarray) -> np.ndarray:
    """Draw the joints of a hand onto an image"""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=2)
    results = hands.process(image)
    draw_image = copy.deepcopy(image)

    hand_points = []
    if results.multi_hand_landmarks:
        for (i, hand_landmarks) in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(draw_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            points = []
            for i in hand_landmarks.landmark[:21]:
                points.append([i.x, i.y, i.z])
            hand_points.append(points)

    hand_positions = [hand_pos(hand_point, image) for hand_point in hand_points]
    left_hand_points = None
    left_hand_position = (1e99, 1e99)
    for (i, hand_position) in enumerate(hand_positions):
        if hand_position[0] < left_hand_position[0]:
            left_hand_position = hand_position
            left_hand_points = hand_points[i]

    bounding_box, bounding_box_size = hand_bounding_box(left_hand_points, image)
    for point_index in range(len(bounding_box)):
        cv2.line(draw_image, bounding_box[point_index], bounding_box[(point_index + 1) % len(bounding_box)], (0, 0, 255), 4)

    plt.imshow(draw_image)
    plt.show()

    plt.imshow(image[bounding_box[0][1]:bounding_box[2][1], bounding_box[0][0]:bounding_box[2][0]])
    plt.show()

    return image[bounding_box[0][1]:bounding_box[2][1], bounding_box[0][0]:bounding_box[2][0]]

def _select_hand_points(hand_points: list, image: np.ndarray) -> Optional[list]:
    """Pick the rightmost hand when multiple are detected."""
    if not hand_points:
        return None
    if len(hand_points) == 1:
        return hand_points[0]

    hand_positions = [hand_pos(hand_point, image) for hand_point in hand_points]
    right_hand_points = None
    right_hand_position = (-1, -1)
    for i, hand_position in enumerate(hand_positions):
        if hand_position[0] > right_hand_position[0]:
            right_hand_position = hand_position
            right_hand_points = hand_points[i]
    return right_hand_points


def _landmarks_to_crop_box(
    landmarks: list,
    image: np.ndarray,
) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) pixel indices for numpy slicing."""
    bounding_box, _ = hand_bounding_box(landmarks, image)
    left = int(bounding_box[0][0])
    top = int(bounding_box[0][1])
    right = int(bounding_box[2][0])
    bottom = int(bounding_box[2][1])
    return left, top, right, bottom


def detect_hand_crop_box(
    image: np.ndarray,
    hands_detector=None,
) -> Optional[tuple[int, int, int, int]]:
    """
    Detect hand and return crop box as (left, top, right, bottom) pixel indices.

    Coordinates are relative to ``image`` and suitable for numpy slicing
    ``image[top:bottom, left:right]``.
    """
    mp_hands = mp.solutions.hands

    should_close = False
    if hands_detector is None:
        hands_detector = mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=2
        )
        should_close = True

    try:
        results = hands_detector.process(image)

        hand_points = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for landmark in hand_landmarks.landmark[:21]:
                    points.append([landmark.x, landmark.y, landmark.z])
                hand_points.append(points)
        else:
            return None

        selected = _select_hand_points(hand_points, image)
        if selected is None:
            return None

        return _landmarks_to_crop_box(selected, image)
    finally:
        if should_close:
            hands_detector.close()


def crop_with_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``image`` using (left, top, right, bottom) indices, clamped to bounds."""
    left, top, right, bottom = box
    height, width = image.shape[:2]
    left = max(0, min(left, width))
    right = max(0, min(right, width))
    top = max(0, min(top, height))
    bottom = max(0, min(bottom, height))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[top:bottom, left:right]


def scale_crop_box(
    box: tuple[int, int, int, int],
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """
    Scale a crop box from ``src_shape`` coordinates to ``dst_shape`` coordinates.

    Uses independent x/y scale factors so rounded resize dimensions are handled
    correctly when projecting from a downscaled frame back to the original frame.
    """
    left, top, right, bottom = box
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source shape must have positive width and height")

    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    return (
        int(round(left * scale_x)),
        int(round(top * scale_y)),
        int(round(right * scale_x)),
        int(round(bottom * scale_y)),
    )


def downscale_for_hand_detection(
    image: np.ndarray,
    max_width: int = 1920,
    max_height: int = 1080,
) -> tuple[np.ndarray, float, float]:
    """
    Downscale ``image`` for MediaPipe when it exceeds ``max_width`` x ``max_height``.

    Returns:
        Tuple of (possibly resized image, x_scale, y_scale) where scales map
        processing coordinates to original coordinates (orig = proc * scale).
    """
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image, 1.0, 1.0

    scale = min(max_width / width, max_height / height)
    new_w = int(width * scale)
    new_h = int(height * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_x = width / float(new_w)
    scale_y = height / float(new_h)
    return resized, scale_x, scale_y


def segment_hand(image: np.ndarray, hands_detector=None) -> np.ndarray:
    """Segment an image such that the right hand is all that remains in the output image
    
    Args:
        image: Input image (RGB)
        hands_detector: Optional MediaPipe Hands detector. If None, creates a new one (not recommended for loops)
    
    Returns:
        Segmented hand image, or None if no hand detected
    """
    crop_box = detect_hand_crop_box(image, hands_detector)
    if crop_box is None:
        return None
    return crop_with_box(image, crop_box)


def segment_hand_at_full_resolution(
    image_full_res: np.ndarray,
    hands_detector=None,
    max_width: int = 1920,
    max_height: int = 1080,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect/blur-check on a downscaled copy, crop from the original-resolution frame.

    Returns:
        Tuple of (full_resolution_crop, processing_resolution_crop). Either may be
        None when hand detection fails or the projected crop is empty.
    """
    image_proc, _, _ = downscale_for_hand_detection(
        image_full_res, max_width=max_width, max_height=max_height
    )
    crop_box_proc = detect_hand_crop_box(image_proc, hands_detector)
    if crop_box_proc is None:
        return None, None

    crop_proc = crop_with_box(image_proc, crop_box_proc)
    if crop_proc.size == 0:
        return None, None

    crop_box_full = scale_crop_box(
        crop_box_proc, image_proc.shape, image_full_res.shape
    )
    crop_full = crop_with_box(image_full_res, crop_box_full)
    if crop_full.size == 0:
        return None, crop_proc

    return crop_full, crop_proc
