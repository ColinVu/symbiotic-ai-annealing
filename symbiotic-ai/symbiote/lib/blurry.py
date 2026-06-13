import os
import cv2
import sys
import argparse

# Handle both running as script and importing as module
try:
    from .hand_detection import segment_hand
except ImportError:
    from hand_detection import segment_hand


def save_non_blurry_frames(capture, threshold, frame_skip=4, output_all=False):
    non_blurry_dir = "non-blurry"
    if not os.path.exists(non_blurry_dir):
        os.makedirs(non_blurry_dir)

    count = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        count += 1

        if count % frame_skip != 0:
            continue

        # Crop frame to hand region
        hand_frame = segment_hand(frame)
        
        # Skip frames where hand is not visible
        if hand_frame is None:
            continue
        
        laplacian_value = calculate_laplacian(hand_frame)
        
        # Save frame if it meets the threshold or if output_all is enabled
        if output_all or laplacian_value >= threshold:
            # Annotate the cropped hand frame with the laplacian value
            annotated_frame = annotate_frame(hand_frame.copy(), laplacian_value)
            filename = f"frame_{count}.png"
            file_path = os.path.join(non_blurry_dir, filename)
            cv2.imwrite(file_path, annotated_frame)


def calculate_laplacian(frame):
    """Calculate the laplacian variance of a frame."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance


def annotate_frame(frame, laplacian_value):
    """Annotate the frame with the laplacian value."""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Format the text
    text = f"Laplacian: {laplacian_value:.2f}"
    
    # Set font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    
    # Get text size to create a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Position the text in the top-left corner with some padding
    padding = 10
    text_x = padding
    text_y = padding + text_height
    
    # Draw a semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + baseline + 5),
                  (0, 0, 0), 
                  -1)
    
    # Blend the overlay with the original frame for transparency
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw the text in white
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    
    return frame


def is_blurry(frame, threshold, crop_to_hand=True):
    """
    Check if a frame is blurry.
    
    Args:
        frame: The input frame
        threshold: Laplacian variance threshold
        crop_to_hand: If True, crop frame to hand region before checking blur
    
    Returns:
        tuple: (is_blurry: bool, hand_detected: bool)
               If crop_to_hand is False, hand_detected is always True
    """
    if crop_to_hand:
        hand_frame = segment_hand(frame)
        if hand_frame is None:
            # No hand detected
            return (True, False)  # Consider as blurry if no hand detected
        frame = hand_frame
    
    variance = calculate_laplacian(frame)
    return (variance < threshold, True)


if __name__ == "__main__":
    # Probably should replace to loop through videos in a folder
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True,
                    help="Provide a video using the --video option.")
    ap.add_argument("--threshold", type=float, default=100.0,
                    help="Provide threshold for considering an image blurry")
    ap.add_argument("--all", action="store_true",
                    help="Output all frames (including blurry ones) instead of just non-blurry frames")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print("File does not exist")
        sys.exit(1)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print("Can't open.")
        sys.exit(1)

    save_non_blurry_frames(capture, args.threshold, output_all=args.all)
    capture.release()
