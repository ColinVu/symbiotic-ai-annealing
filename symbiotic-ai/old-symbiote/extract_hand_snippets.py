#
# Extract and annotate unique hand snippets from a video using inference
# Outputs a folder of images with predicted item labels
#

# Suppress MediaPipe "Feedback manager" / inference warnings (set before importing mediapipe)
import os
os.environ["GLOG_minloglevel"] = "2"  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

from transformers import AutoModel, AutoProcessor
import chromadb
import cv2
import argparse
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Import mediapipe BEFORE modifying path (hand_detection/state_detection use mp.solutions)
import mediapipe as mp
mp_hands = mp.solutions.hands

# Add lib folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from embedding import MODEL, embed_image
from inference import perform_inference as _perform_inference
import state_detection
from state_detection import State
from hand_detection import segment_hand

# Wrapper for perform_inference to handle empty ChromaDB and errors
def perform_inference(embedding, collection):
    """Wrapped inference that handles empty collections and errors"""
    try:
        # Check if collection is empty
        if collection.count() == 0:
            return ("unknown", 0)
        # Call original function with query_embeddings as list
        return _perform_inference([embedding.tolist()] if hasattr(embedding, 'tolist') else [embedding], collection)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        # Handle empty bins, missing keys, etc.
        return ("unknown", 0)

# Patch state_detection to fix the bugs in the original
_original_detect_state = state_detection.detect_state
def patched_detect_state(image, hand_detector):
    """Patched version that fixes assignment issues"""
    global _detection_idx
    orientation = state_detection.detect_orientation(image, hand_detector)
    current_state = State(np.bincount(state_detection.recent_detections).argmax())
    next_state = current_state
    
    if orientation == state_detection.Orientation.Up and current_state == State.PICK:
        next_state = State.CARRY_WITH
    elif orientation == state_detection.Orientation.Up and current_state == State.PLACE:
        next_state = State.CARRY_WITHOUT
    elif orientation == state_detection.Orientation.Down and current_state == State.CARRY_WITH:
        next_state = State.PLACE
    elif orientation == state_detection.Orientation.Down and current_state == State.CARRY_WITHOUT:
        next_state = State.PICK
    
    state_detection.recent_detections[_detection_idx % 10] = next_state.value
    _detection_idx = (_detection_idx + 1) % 10000
    return current_state

_detection_idx = 0
detect_state = patched_detect_state


def images_are_similar(img1_embedding: np.ndarray, img2_embedding: np.ndarray, threshold: float = 0.95) -> bool:
    """Check if two images are similar based on their embeddings using cosine similarity"""
    # Normalize embeddings
    img1_norm = img1_embedding / np.linalg.norm(img1_embedding)
    img2_norm = img2_embedding / np.linalg.norm(img2_embedding)
    
    # Calculate cosine similarity
    similarity = np.dot(img1_norm, img2_norm)
    
    return similarity > threshold


def annotate_image(image: cv2.typing.MatLike, predicted_item: str) -> cv2.typing.MatLike:
    """Annotate an image with the predicted item label"""
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Create drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add background rectangle for text
    text = f"Predicted: {predicted_item}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position at top of image
    padding = 10
    rect_coords = [padding, padding, text_width + padding * 2, text_height + padding * 2]
    draw.rectangle(rect_coords, fill=(0, 0, 0, 180))
    draw.text((padding * 2, padding * 2), text, fill=(255, 255, 255), font=font)
    
    # Convert back to BGR for OpenCV
    annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return annotated_image


def main(
    model: AutoModel,
    processor: AutoProcessor,
    collection: chromadb.Collection,
    capture: cv2.VideoCapture,
    output_folder: str,
    similarity_threshold: float
):
    """
    Process video and extract unique hand snippets with predicted item labels
    
    Args:
        model: CLIP model for embeddings
        processor: CLIP processor
        collection: ChromaDB collection for inference
        capture: OpenCV video capture object
        output_folder: Path to folder where images will be saved
        similarity_threshold: Threshold for considering images as duplicates (0-1, higher = more strict)
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe hands with static_image_mode to reduce crashes
    hand_detector = mp_hands.Hands(
        static_image_mode=True,  # Process each frame independently
        min_detection_confidence=0.5,
        max_num_hands=2
    )
    
    # Track unique images and their embeddings
    saved_embeddings = []
    saved_count = 0
    
    # Track embeddings during carry phase
    carry_embeddings = []
    
    # Process every Nth frame for state detection to reduce MediaPipe load
    STATE_DETECTION_INTERVAL = 5  # Check state every 5 frames
    
    frame_count = 0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    current_state = State.PICK  # Initialize to PICK state
    
    print(f"Processing video with {total_frames} frames...")
    print(f"Output folder: {output_folder}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"State detection interval: every {STATE_DETECTION_INTERVAL} frames")
    
    try:
        while True:
            ret, img = capture.read()
            
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processing frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
            
            # Resize and convert color
            img = cv2.resize(img, (1920, 1080))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect current state (only every N frames to reduce MediaPipe load)
            if frame_count % STATE_DETECTION_INTERVAL == 0:
                try:
                    current_state = detect_state(img_rgb, hand_detector)
                except Exception as e:
                    print(f"  State detection error: {e}")
                    # Keep previous state on error
            
            # Debug: show state on every 30th frame
            if frame_count % 30 == 0:
                print(f"  Current state: {current_state.name}, carry_embeddings: {len(carry_embeddings)}")
            
            match current_state:
                case State.PICK:
                    pass
                    
                case State.CARRY_WITH:
                    # Collect embeddings while carrying
                    embedding = embed_image(img, model, processor)
                    if embedding is not None:
                        carry_embeddings.append(embedding)
                        if frame_count % 30 == 0:
                            print(f"    → Collected embedding (total: {len(carry_embeddings)})")
                    elif frame_count % 30 == 0:
                        print(f"    → Failed to get embedding (segment_hand returned None)")
                    
                case State.PLACE:
                    # When placing, perform inference and save unique image
                    print(f"\n  *** PLACE detected! Carry embeddings: {len(carry_embeddings)} ***")
                    if len(carry_embeddings) > 0:
                        average_embedding = np.mean(np.array(carry_embeddings), axis=0)
                    else:
                        # No embeddings collected during carry (e.g. hand rarely detected) – use current frame
                        average_embedding = embed_image(img, model, processor)
                        if average_embedding is None:
                            carry_embeddings = []
                            continue

                    # Perform inference to get predicted item
                    try:
                        (item_prediction, confidence) = perform_inference(average_embedding, collection)
                        item_prediction = str(item_prediction)
                        print(f"  Detected item: {item_prediction} (confidence: {confidence})")
                    except Exception as e:
                        print(f"  Inference failed: {e}, using 'unknown' label")
                        item_prediction = "unknown"

                    # Check if this image is unique compared to already saved images
                    is_unique = True
                    for saved_emb in saved_embeddings:
                        if images_are_similar(average_embedding, saved_emb, similarity_threshold):
                            is_unique = False
                            print(f"  Skipping duplicate image")
                            break

                    if is_unique:
                        hand_snippet = segment_hand(img_rgb)
                        if hand_snippet is not None:
                            annotated_snippet = annotate_image(hand_snippet, item_prediction)
                            output_path = os.path.join(output_folder, f"hand_snippet_{saved_count:04d}_{item_prediction}.jpg")
                            cv2.imwrite(output_path, annotated_snippet)
                            saved_embeddings.append(average_embedding)
                            saved_count += 1
                            print(f"  Saved unique image: {output_path}")
                        else:
                            print(f"  Failed to segment hand from frame")

                    carry_embeddings = []
                    
                case State.CARRY_WITHOUT:
                    pass
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        capture.release()
        hand_detector.close()
        print(f"\nProcessing complete!")
        print(f"Total unique hand snippets saved: {saved_count}")
        print(f"Images saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and annotate unique hand snippets from a video using inference"
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="Path to the input video",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output folder for hand snippet images",
        required=True
    )
    parser.add_argument(
        "-s", "--similarity-threshold",
        type=float,
        help="Similarity threshold for duplicate detection (0-1, default: 0.95)",
        default=0.95
    )
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Open video capture
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"Error: Could not open video file: {args.video}")
        sys.exit(1)
    
    # Load CLIP model
    print("Loading CLIP model...")
    model = AutoModel.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)
    print("Model loaded successfully")
    
    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="symbiotic-ai",
        metadata={
            "hnsw:space": "cosine"
        }
    )
    print(f"ChromaDB initialized with {collection.count()} embeddings")
    
    # Run main processing
    main(
        model,
        processor,
        collection,
        capture,
        args.output,
        args.similarity_threshold
    )
