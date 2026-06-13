#
# Helper functions for detecting the current state of the program
#

import cv2
import enum
from typing import Optional
import mediapipe as mp
import numpy as np


class Orientation(enum.Enum):
    """The orientation of the hand"""
    # The palm is oriented upwards
    Up = 0
    # The palm is oriented downards
    Down = 1

def detect_orientation(
    image: cv2.typing.MatLike,
    hand_detector: mp.solutions.hands.Hands
) -> Optional[Orientation]:
    """Detect the orientation of a hand from an image"""
    results = hand_detector.process(image)
    if results.multi_hand_landmarks:
        right_thumb = None
        right_middle_finger = None
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark[:21]
            thumb = np.array([landmarks[4].x, landmarks[4].y, 0.0])
            middle_finger = np.array([landmarks[12].x, landmarks[12].y, 0.0])
            if right_thumb is None:
                right_thumb = thumb
                right_middle_finger = middle_finger
            elif thumb[0] > right_thumb[0]:
                right_thumb = thumb
                right_middle_finger = middle_finger
        if right_thumb is not None:
            cross_product = np.cross(right_thumb, right_middle_finger)

            if cross_product[2] < 0:
                return Orientation.Down
            else:
                return Orientation.Up
        else:
            return None
    else:
        return None


class State(enum.Enum):
    """The current state of the pick process"""
    # The user is picking up an item
    PICK = 0
    # The user is carrying an item to the bins
    CARRY_WITH = 1
    # The user is placing an item into the bins
    PLACE = 2
    # The user is walking back to the picking area
    CARRY_EMPTY = 3

# The most recent 10 detections
recent_detections = np.array([State.PICK.value for _ in range(10)])
# The index of the current detection
detection_idx = 0

def detect_state(
    image: cv2.typing.MatLike,
    hand_detector: mp.solutions.hands.Hands
) -> State:
    """Detect the current state of the pick process"""
    orientation = detect_orientation(image, hand_detector)
    current_state = State(np.bincount(recent_detections).argmax())
    next_state = current_state

    match (orientation, current_state):
        case (Orientation.Up, State.PICK):
            next_state = State.CARRY_WITH
        case (Orientation.Up, State.PLACE):
            next_state = State.CARRY_EMPTY
        case (Orientation.Down, State.CARRY_WITH):
            next_state = State.PLACE
        case (Orientation.Down, State.CARRY_EMPTY):
            next_state = State.PICK
    
    recent_detections[detection_idx] = next_state
    detection_idx += 1
    return current_state
        
