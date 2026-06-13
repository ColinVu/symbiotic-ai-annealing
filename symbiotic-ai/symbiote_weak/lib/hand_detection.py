#
# Hand Detection helper methods
#

import mediapipe as mp
from matplotlib import pyplot as plt
import cv2
import copy
import numpy as np

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

def segment_hand(image: np.ndarray, hands_detector=None) -> np.ndarray:
    """Segment an image such that the right hand is all that remains in the output image
    
    Args:
        image: Input image (RGB)
        hands_detector: Optional MediaPipe Hands detector. If None, creates a new one (not recommended for loops)
    
    Returns:
        Segmented hand image, or None if no hand detected
    """
    mp_hands = mp.solutions.hands
    
    # Use provided detector or create a new one (for backward compatibility)
    should_close = False
    if hands_detector is None:
        hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=2)
        should_close = True
    
    try:
        results = hands_detector.process(image)

        hand_points = []
        if results.multi_hand_landmarks:
            for (i, hand_landmarks) in enumerate(results.multi_hand_landmarks):
                points = []
                for i in hand_landmarks.landmark[:21]:
                    points.append([i.x, i.y, i.z])
                hand_points.append(points)
        else:
            return None

        if len(hand_points) > 1:
            hand_positions = [hand_pos(hand_point, image) for hand_point in hand_points]
            right_hand_points = None
            right_hand_position = (1e99, 1e99)
            for (i, hand_position) in enumerate(hand_positions):
                if hand_position[0] < right_hand_position[0]:
                    right_hand_position = hand_position
                    right_hand_points = hand_points[i]
        else:
            right_hand_points = hand_points[0]

        bounding_box, _ = hand_bounding_box(right_hand_points, image)

        return image[bounding_box[0][1]:bounding_box[2][1], bounding_box[0][0]:bounding_box[2][0]]
    finally:
        if should_close:
            hands_detector.close()
