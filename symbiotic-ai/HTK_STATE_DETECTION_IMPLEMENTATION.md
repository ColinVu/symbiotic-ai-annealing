# HTK Hidden Markov Model State Detection Implementation Guide

> **🐳 Docker Deployment**: This system requires HTK toolkit. For Windows users, Docker containerization (see `DOCKER_SETUP.md`) is the recommended approach to avoid HTK compilation issues. All commands in this guide work inside Docker with appropriate path adjustments.

## Overview

This document describes how to implement an HTK-based Hidden Markov Model (HMM) system for detecting hand manipulation states in the symbiote pipeline. This system will replace the placeholder in `state_detection/detector.py` with a real, trainable state detection algorithm.

**Date Created**: February 15, 2026  
**Status**: Planning Document (Not Yet Implemented)  
**Target Location**: `symbiote/state_detection/`

---

## State Model

### Four-State Cyclic Model

The HMM enforces a strict state cycle:

```
PICK → CARRY_WITH → PLACE → CARRY_EMPTY → (back to PICK)
```

**State Definitions:**
- **PICK**: Hand grasping object from a bin (near ARUCO "pick" markers)
- **CARRY_WITH**: Hand transporting object through space
- **PLACE**: Hand releasing object into a bin (near ARUCO "place" markers)
- **CARRY_EMPTY**: Hand moving without object (returning to pick location)

**Critical Constraints:**
- Items can ONLY be picked from bins marked with "pick" ARUCO markers
- Items can ONLY be placed into bins marked with "place" ARUCO markers
- Each bin contains ONE type of item (identified by ARUCO marker ID)
- Only RIGHT hand is used (enforced by existing `segment_hand()` function)

**ARUCO Marker Role:**
- ARUCO markers are NOT direct HMM features
- Instead, they provide a **weighted context signal** to help disambiguate PICK vs PLACE
- When HMM detects hand entering a bin, ARUCO weights inform which transition is more likely

---

## Feature Vector Design

### Total Dimensionality: 17-19 Features

The HMM will be trained on multi-dimensional time-series feature vectors extracted from video frames. These features leverage **existing pipeline infrastructure** where possible.

### Feature Groups

#### 1. Hand Position & Motion (6D)
**Source**: `hand_detection.py::hand_pos()`

```python
# Already computed by MediaPipe
1. x_norm: Normalized hand center x-position (0-1, relative to frame width)
2. y_norm: Normalized hand center y-position (0-1, relative to frame height)
3. vx: Horizontal velocity (Δx between frames)
4. vy: Vertical velocity (Δy between frames)
5. ax: Horizontal acceleration (ΔΔx)
6. ay: Vertical acceleration (ΔΔy)
```

**State Signatures:**
- **PICK**: Low y, increasing vy (upward), high ay
- **CARRY_WITH**: Mid-high y, moderate vx, low ay
- **PLACE**: Decreasing y (downward), high negative vy
- **CARRY_EMPTY**: Variable position, low velocity magnitude

#### 2. Hand Bounding Box (4D)
**Source**: `hand_detection.py::hand_bounding_box()`

```python
# Box dimensions and change rates
7. width_norm: Bounding box width / frame_width
8. height_norm: Bounding box height / frame_height
9. Δwidth: Change in width between frames (grasping indicator)
10. Δheight: Change in height between frames
```

**State Signatures:**
- **PICK**: Rapid shrinking (negative Δwidth, Δheight) as fingers close
- **CARRY_WITH**: Stable dimensions
- **PLACE**: Expanding (positive Δwidth, Δheight) as fingers open
- **CARRY_EMPTY**: Stable dimensions

#### 3. Hand Orientation (3D) - NEW
**Source**: MediaPipe landmarks (to be extracted from `hand_detection.py`)

MediaPipe provides 21 3D hand landmarks. Use these specific points:
- **Landmark 0**: Base of palm (wrist)
- **Landmark 4**: Thumb tip
- **Landmark 12**: Middle finger tip

```python
# Compute two vectors from palm base
palm_to_thumb = landmark[4] - landmark[0]  # Vector 1
palm_to_middle = landmark[12] - landmark[0]  # Vector 2

# Cross product gives normal to palm plane
cross = palm_to_thumb × palm_to_middle

11. orientation_x: Cross product x-component (normalized)
12. orientation_y: Cross product y-component (normalized)
13. orientation_z: Cross product z-component (normalized)
```

**Why This Matters:**
- Right hand palm orientation changes predictably during pick/place
- Cross product is robust to individual finger tracking errors
- Provides 3D spatial context for grasp direction

**State Signatures:**
- **PICK**: Palm facing down/forward (toward bin)
- **CARRY_WITH**: Palm horizontal or slightly upward
- **PLACE**: Palm facing down (releasing)
- **CARRY_EMPTY**: Variable but distinct from WITH states

#### 4. Object Presence (1D)
**Source**: Existing CLIP inference pipeline

```python
# Maximum confidence score from CLIP classifier
13. object_confidence: max(softmax(classifier_logits))  # index 13 in 29D layout
```

**Reuse**: This is already computed in `inference/recognizer.py::ObjectRecognizer`

**State Signatures:**
- **PICK**: Rapidly increasing (0.1 → 0.8)
- **CARRY_WITH**: High and stable (>0.7)
- **PLACE**: Rapidly decreasing (0.8 → 0.1)
- **CARRY_EMPTY**: Low and stable (<0.3)

#### 5. Hand color histogram (12D) — `symbiote_weak`
**Source**: Cropped hand RGB (`segmented_hand`), converted to HSV.

```python
# L1-normalised histograms on the valid hand crop (zeros if empty)
14–21. hue_hist_0..7   (8 bins over OpenCV H range [0, 180))
22–25. sat_hist_0..3   (4 bins over S range [0, 256))
```

#### 6. ARUCO bin proximity (3D) — `symbiote_weak`
**Source**: ARUCO detection module (`aruco_detection.py`)

**IMPORTANT**: Markers are not one-hot bin IDs; they feed **distance-weighted** channels (with persistence + temporal smoothing in `FeatureExtractor`).

```python
26. aruco_signed_context: float (scaled, ≈ pick − place, clipped)
27. aruco_pick_proximity: float (scaled [0, aruco_weight])
28. aruco_place_proximity: float (scaled [0, aruco_weight])
```

Legacy note: the signed channel matches the former single `bin_context_weight` semantics before scaling.

```python
# Raw geometry (before aruco_weight / deadband / smoothing)
pick_proximity, place_proximity in [0, 1] (sum of exp decay scores, clipped)
signed_context = clip(pick_raw - place_raw, -1, 1)
```

**How It Works:**

1. **Detection Phase**: System detects ALL ARUCO markers in frame
2. **Weight Calculation**: Compute weighted score based on:
   - Distance from hand to each marker (closer = higher weight)
   - Type of each marker (pick = positive, place = negative)
   - Number of markers visible
   
3. **Weighting Formula**:
```python
weight = 0.0
for marker in detected_markers:
    distance = norm(hand_pos - marker_center) / frame_diagonal
    proximity_weight = exp(-distance * 5)  # Exponential decay
    
    if marker_type == "pick":
        weight += proximity_weight
    elif marker_type == "place":
        weight -= proximity_weight

# Clamp to [-1, 1]
bin_context_weight = clip(weight, -1.0, 1.0)
```

4. **HMM Usage**: This single weighted feature helps HMM decide:
   - When entering bin (low hand position + hand slowing):
     - If `bin_context_weight > 0.5` → More likely PICK transition
     - If `bin_context_weight < -0.5` → More likely PLACE transition
     - If `bin_context_weight ≈ 0` → Use other features to decide

**Resulting HMM feature dim (`symbiote_weak`): 29D total** (14 motion/orientation/object + 12 color + 3 ARUCO). `HTKConfig.feature_dim` must stay in sync with `FeatureExtractor.FEATURE_DIM`.

**Why This Approach:**
- More sophisticated than binary one-hot encoding
- Handles multiple visible markers gracefully
- Natural decay with distance (far markers have less influence)
- Separate pick/place proximity channels plus signed context for richer HMM observability

**State Signatures:**
- **PICK**: bin_context_weight > 0.5 (near pick bins)
- **CARRY_WITH**: bin_context_weight ≈ 0 (away from bins)
- **PLACE**: bin_context_weight < -0.5 (near place bins)
- **CARRY_EMPTY**: bin_context_weight variable (moving between areas)

#### 7. Derived Motion Features (2D) — optional / not in core 29D stream
**Source**: Computed from above features

```python
16. velocity_magnitude: sqrt(vx² + vy²)
17. [optional] acceleration_magnitude: sqrt(ax² + ay²)
```

**State Signatures:**
- **PICK/PLACE** (transitions): Low velocity magnitude (hand hovering/careful)
- **CARRY_WITH**: Moderate-high velocity magnitude (transporting)
- **CARRY_EMPTY**: Variable (returning to pick)

---

### Total Feature Dimensionality: **29D** (`symbiote_weak` HTK stream)
- Hand position & motion: 6D
- Bounding box: 4D
- Hand orientation: 3D
- Object confidence: 1D
- **Hand HSV histograms: 12D**
- **ARUCO signed + pick + place: 3D**

Feature cache manifests include `feature_dim` so upgrading code invalidates stale `.npy` caches.

---

## Implementation Architecture

### Module Structure

```
symbiote/
├── state_detection/
│   ├── __init__.py                    [EXISTING - update exports]
│   ├── detector.py                    [EXISTING - replace placeholder]
│   ├── feature_extraction.py         [NEW]
│   ├── aruco_detection.py            [NEW]
│   ├── htk_interface.py              [NEW]
│   ├── training.py                   [NEW]
│   └── config.py                     [NEW]
```

### File Responsibilities

#### `feature_extraction.py` - Feature Vector Generator
**Purpose**: Extract 17-19D feature vectors from video frames

**Key Functions:**

```python
class FeatureExtractor:
    """Extract HMM features from video frames."""
    
    def __init__(self, clip_model, clip_processor, aruco_detector):
        """Initialize with CLIP model and ARUCO detector."""
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.aruco_detector = aruco_detector
        self.prev_hand_pos = None  # For velocity
        self.prev_velocity = None  # For acceleration
    
    def extract_frame_features(
        self,
        frame: np.ndarray,
        hand_landmarks: List[List[float]],  # From MediaPipe
        segmented_hand: np.ndarray,
        frame_time: float
    ) -> np.ndarray:
        """
        Extract 15D feature vector from single frame.
        
        Args:
            frame: Full RGB frame (for ARUCO detection)
            hand_landmarks: 21x3 MediaPipe landmarks
            segmented_hand: Cropped hand image (for CLIP)
            frame_time: Timestamp in seconds
            
        Returns:
            Feature vector of shape (29,)
        """
        # 1. Hand position & motion (6D)
        hand_center = self._compute_hand_center(hand_landmarks, frame.shape)
        velocity, acceleration = self._compute_motion(hand_center, frame_time)
        
        # 2. Bounding box (4D)
        bbox_features = self._compute_bbox_features(hand_landmarks, frame.shape)
        
        # 3. Hand orientation (3D)
        orientation = self._compute_hand_orientation(hand_landmarks)
        
        # 4. Object confidence (1D)
        obj_conf = self._compute_object_confidence(segmented_hand)
        
        # 5. ARUCO weighted bin context (1D) - NEW WEIGHTED APPROACH
        bin_weight = self.aruco_detector.compute_bin_context_weight(frame, hand_center)
        
        # 6. Derived motion (2D)
        vel_mag = np.linalg.norm(velocity)
        acc_mag = np.linalg.norm(acceleration)
        
        # Concatenate all features
        features = np.concatenate([
            hand_center,          # 2D
            velocity,             # 2D
            acceleration,         # 2D
            bbox_features,        # 4D
            orientation,          # 3D
            [obj_conf],          # 1D
            [bin_weight],        # 1D - WEIGHTED ARUCO CONTEXT
            [vel_mag, acc_mag]   # 2D
        ])
        
        return features  # Total: 15D
    
    def extract_video_features(
        self,
        video_path: str,
        frame_skip: int = 4
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Extract features for entire video.
        
        Returns:
            features: (n_frames, 29) feature matrix
            frame_numbers: List of processed frame indices
            fps: Video frame rate
        """
        # Implementation uses existing video_processor.py pattern
        pass
```

**Integration Points:**
- **Reuses**: `hand_detection.py::segment_hand()` for hand extraction
- **Reuses**: `preprocessing.blur_detection::is_blurry()` for quality filtering
- **Reuses**: `inference.recognizer::ObjectRecognizer` for object confidence
- **New**: ARUCO detection module with weighted context computation

#### `aruco_detection.py` - ARUCO Marker Detection
**Purpose**: Detect ARUCO markers and compute weighted bin context

**Key Functions:**

```python
class ArucoDetector:
    """Detect ARUCO markers and compute weighted bin context."""
    
    def __init__(self, aruco_dict_type=cv2.aruco.DICT_4X4_1000):
        """Initialize ARUCO detector with dictionary type."""
        self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # Bin configuration: maps ARUCO ID to bin type and object
        # Example: {0: ("pick", "apple"), 1: ("pick", "banana"), 
        #           10: ("place", "apple"), 11: ("place", "banana")}
        self.bin_config = {}  # Load from config file
        
        # Weighting parameters
        self.distance_decay = 5.0  # Controls how fast weight decays with distance
    
    def detect_markers(self, frame: np.ndarray) -> Dict:
        """
        Detect all ARUCO markers in frame.
        
        Returns:
            {
                'ids': List of detected marker IDs,
                'centers': List of (x, y) marker centers,
                'types': List of bin types ('pick' or 'place'),
                'corners': List of 4x2 corner coordinates
            }
        """
        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params
        )
        
        if ids is None:
            return {'ids': [], 'centers': [], 'types': [], 'corners': []}
        
        # Compute marker centers
        centers = []
        types = []
        for marker_id, corner_set in zip(ids.flatten(), corners):
            center = corner_set[0].mean(axis=0)  # Average of 4 corners
            centers.append(center)
            
            # Look up bin type from config
            bin_type = self.bin_config.get(marker_id, {}).get('type', 'unknown')
            types.append(bin_type)
        
        return {
            'ids': ids.flatten().tolist(),
            'centers': centers,
            'types': types,
            'corners': corners
        }
    
    def compute_bin_context_weight(
        self,
        frame: np.ndarray,
        hand_position: Tuple[float, float]
    ) -> float:
        """
        Compute weighted bin context score: +1 = PICK, -1 = PLACE, 0 = neutral.
        
        This is the CORE of the weighted ARUCO approach. It aggregates evidence
        from ALL visible markers into a single continuous score.
        
        Args:
            frame: RGB frame image
            hand_position: (x, y) hand center position in pixels
            
        Returns:
            weight: float in [-1.0, 1.0]
                +1.0 = Strong PICK context (near pick bins)
                -1.0 = Strong PLACE context (near place bins)
                 0.0 = No bins or ambiguous/far from bins
        """
        markers = self.detect_markers(frame)
        
        if len(markers['ids']) == 0:
            # No markers detected → neutral
            return 0.0
        
        # Frame diagonal for normalization
        frame_diag = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
        
        weight = 0.0
        
        for marker_id, center, bin_type in zip(markers['ids'], 
                                                 markers['centers'], 
                                                 markers['types']):
            if bin_type == 'unknown':
                continue  # Ignore unconfigured markers
            
            # Compute distance from hand to marker
            distance = np.linalg.norm(
                np.array(hand_position) - np.array(center)
            )
            normalized_dist = distance / frame_diag
            
            # Exponential decay: closer markers have more influence
            proximity_weight = np.exp(-normalized_dist * self.distance_decay)
            
            # Add to weight based on bin type
            if bin_type == 'pick':
                weight += proximity_weight
            elif bin_type == 'place':
                weight -= proximity_weight
        
        # Clamp to [-1, 1]
        weight = np.clip(weight, -1.0, 1.0)
        
        return weight
    
    def visualize_bin_context(
        self,
        frame: np.ndarray,
        hand_position: Tuple[float, float]
    ) -> Tuple[np.ndarray, float]:
        """
        Draw detected markers and bin context weight on frame.
        
        Used for debugging and testing. Returns annotated frame and weight.
        """
        markers = self.detect_markers(frame)
        annotated = frame.copy()
        
        # Draw all detected markers
        if len(markers['ids']) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, markers['corners'], 
                                           np.array(markers['ids']))
            
            # Draw marker centers and types
            for marker_id, center, bin_type in zip(markers['ids'], 
                                                     markers['centers'], 
                                                     markers['types']):
                center_int = tuple(center.astype(int))
                
                # Color code: green = pick, red = place, gray = unknown
                color = (0, 255, 0) if bin_type == 'pick' else \
                        (0, 0, 255) if bin_type == 'place' else \
                        (128, 128, 128)
                
                cv2.circle(annotated, center_int, 10, color, -1)
                cv2.putText(annotated, f"ID:{marker_id}", 
                           (center_int[0] + 15, center_int[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw hand position
        hand_int = tuple(np.array(hand_position).astype(int))
        cv2.circle(annotated, hand_int, 15, (255, 0, 255), 3)
        
        # Compute and display weight
        weight = self.compute_bin_context_weight(frame, hand_position)
        
        # Draw weight indicator
        weight_text = f"Bin Context: {weight:+.2f}"
        weight_color = (0, 255, 0) if weight > 0.5 else \
                       (0, 0, 255) if weight < -0.5 else \
                       (0, 255, 255)
        
        cv2.putText(annotated, weight_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, weight_color, 3)
        
        # Draw weight bar
        bar_length = int(abs(weight) * 200)
        bar_start_x = 300
        bar_y = 30
        if weight > 0:
            cv2.rectangle(annotated, (bar_start_x, bar_y - 10),
                         (bar_start_x + bar_length, bar_y + 10),
                         (0, 255, 0), -1)
            cv2.putText(annotated, "PICK", (bar_start_x + bar_length + 10, bar_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(annotated, (bar_start_x - bar_length, bar_y - 10),
                         (bar_start_x, bar_y + 10),
                         (0, 0, 255), -1)
            cv2.putText(annotated, "PLACE", (bar_start_x - bar_length - 70, bar_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated, weight
    
    def load_bin_config(self, config_path: str):
        """Load ARUCO-to-bin mapping from JSON file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Convert to simplified format: {id: {type, object}}
            self.bin_config = {}
            for marker_id_str, bin_info in config.get('bins', {}).items():
                self.bin_config[int(marker_id_str)] = bin_info
```

**Configuration File Format** (`aruco_bins.json`):
```json
{
  "marker_dict": "DICT_4X4_1000",
  "bins": {
    "0": {"type": "pick", "object": "apple", "color": "red"},
    "1": {"type": "pick", "object": "banana", "color": "yellow"},
    "2": {"type": "pick", "object": "orange", "color": "orange"},
    "10": {"type": "place", "object": "apple", "color": "red"},
    "11": {"type": "place", "object": "banana", "color": "yellow"},
    "12": {"type": "place", "object": "orange", "color": "orange"}
  },
  "distance_decay": 5.0
}
```

#### `htk_interface.py` - HTK Training & Inference
**Purpose**: Interface between Python and HTK toolkit

**Key Functions:**

```python
class HTKStateDetector:
    """HTK-based HMM state detector."""
    
    def __init__(self, model_dir: str):
        """Load trained HMM models."""
        self.model_dir = model_dir
        self.hmm_def = None  # HTK model definition
        self.state_map = {
            0: HandState.PICK,
            1: HandState.CARRY_WITH,
            2: HandState.PLACE,
            3: HandState.CARRY_EMPTY
        }
    
    def train(
        self,
        training_data: List[Tuple[np.ndarray, pd.DataFrame]],
        output_dir: str
    ):
        """
        Train HTK HMM from annotated videos.
        
        Args:
            training_data: List of (features, annotations) pairs
                features: (n_frames, 17-19) array
                annotations: DataFrame with columns [timestamp_start, timestamp_end, state]
            output_dir: Directory to save trained models
        """
        # 1. Write features to HTK format (.mfc files)
        for i, (features, annotations) in enumerate(training_data):
            self._write_htk_features(features, f"{output_dir}/train_{i}.mfc")
            self._write_htk_labels(annotations, f"{output_dir}/train_{i}.lab")
        
        # 2. Create HTK configuration file
        self._create_htk_config(output_dir)
        
        # 3. Run HTK training commands via subprocess
        self._run_htk_training(output_dir)
    
    def decode(
        self,
        features: np.ndarray,
        fps: float
    ) -> pd.DataFrame:
        """
        Decode state sequence from feature matrix.
        
        Args:
            features: (n_frames, 29) feature array
            fps: Video frame rate
            
        Returns:
            DataFrame with columns [timestamp_start, timestamp_end, state]
        """
        # 1. Write features to HTK format
        temp_mfc = self._write_htk_features(features, "temp.mfc")
        
        # 2. Run HTK Viterbi decoding
        state_sequence = self._run_htk_decode(temp_mfc)
        
        # 3. Segment continuous state sequences
        segments = self._segment_states(state_sequence, fps)
        
        return segments
    
    def _write_htk_features(self, features: np.ndarray, output_path: str):
        """Convert numpy array to HTK binary feature file format."""
        import struct
        
        n_frames, n_features = features.shape
        sample_period = 100000  # 10ms in 100ns units (arbitrary for video)
        
        # HTK header format: nSamples, sampPeriod, sampSize, parmKind
        header = struct.pack('>IIHH', 
                            n_frames,
                            sample_period,
                            n_features * 4,  # 4 bytes per float
                            9)  # USER feature type
        
        with open(output_path, 'wb') as f:
            f.write(header)
            # Write features as big-endian floats
            for frame_features in features:
                f.write(struct.pack(f'>{n_features}f', *frame_features))
    
    def _write_htk_labels(self, annotations: pd.DataFrame, output_path: str):
        """Convert annotations to HTK label file format."""
        with open(output_path, 'w') as f:
            for _, row in annotations.iterrows():
                start_time = int(row['timestamp_start'] * 1e7)  # 100ns units
                end_time = int(row['timestamp_end'] * 1e7)
                state = row['state']
                f.write(f"{start_time} {end_time} {state}\n")
    
    def _create_htk_config(self, output_dir: str):
        """Create HTK configuration and prototype files."""
        # Create HMM prototype with:
        # - 4 states (one per hand state: PICK, CARRY_WITH, PLACE, CARRY_EMPTY)
        # - Gaussian mixture models for each state
        # - Enforced left-to-right topology (PICK→CARRY_WITH→PLACE→CARRY_EMPTY)
        # - Cyclic transition: CARRY_EMPTY → PICK allowed
        
        proto_content = """~o
<STREAMINFO> 1 29  # 29 dimensional features (see HTKConfig.feature_dim)
<VECSIZE> 29<USER>
~h "proto"
<BEGINHMM>
<NUMSTATES> 6  # 4 emitting + entry/exit
<STATE> 2  # PICK state
<MEAN> 29
  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<VARIANCE> 29
  1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<STATE> 3  # CARRY_WITH state
<MEAN> 29
  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<VARIANCE> 29
  1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<STATE> 4  # PLACE state
<MEAN> 29
  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<VARIANCE> 29
  1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<STATE> 5  # CARRY_EMPTY state
<MEAN> 29
  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<VARIANCE> 29
  1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
<TRANSP> 6
  0.0 1.0 0.0 0.0 0.0 0.0  # Entry → PICK
  0.0 0.6 0.4 0.0 0.0 0.0  # PICK → PICK or CARRY_WITH
  0.0 0.0 0.6 0.4 0.0 0.0  # CARRY_WITH → CARRY_WITH or PLACE
  0.0 0.0 0.0 0.6 0.4 0.0  # PLACE → PLACE or CARRY_EMPTY
  0.0 0.3 0.0 0.0 0.6 0.1  # CARRY_EMPTY → cycle back or exit
  0.0 0.0 0.0 0.0 0.0 0.0  # Exit
<ENDHMM>
"""
        
        with open(f"{output_dir}/proto", 'w') as f:
            f.write(proto_content)
    
    def _run_htk_training(self, output_dir: str):
        """Execute HTK training pipeline."""
        import subprocess
        
        # 1. HCompV: Compute global mean/variance for initialization
        subprocess.run([
            'HCompV', '-T', '1',
            '-S', f'{output_dir}/train.scp',  # List of .mfc files
            '-M', output_dir,
            f'{output_dir}/proto'
        ])
        
        # 2. HERest: Baum-Welch re-estimation (multiple iterations)
        for iteration in range(10):
            subprocess.run([
                'HERest', '-T', '1',
                '-S', f'{output_dir}/train.scp',
                '-I', f'{output_dir}/labels.mlf',  # Master label file
                '-M', output_dir,
                '-w', '1.0',  # State transition weight
                f'{output_dir}/hmm_iter{iteration}'
            ])
    
    def _run_htk_decode(self, mfc_path: str) -> List[str]:
        """Run HTK Viterbi decoding."""
        import subprocess
        
        result = subprocess.run([
            'HVite', '-T', '1',
            '-H', f'{self.model_dir}/hmm_final',
            '-S', mfc_path,
            '-w', f'{self.model_dir}/network',  # State transition grammar
            '-o', 'STW'  # Output format
        ], capture_output=True)
        
        # Parse HTK output to extract state sequence
        # Format: start_time end_time state_label
        return self._parse_htk_output(result.stdout)
```

#### `training.py` - Training Pipeline
**Purpose**: High-level training workflow

**Key Functions:**

```python
def train_state_detector(
    video_paths: List[str],
    annotation_paths: List[str],
    output_dir: str,
    aruco_config_path: str,
    clip_model_path: str,
    frame_skip: int = 4,
    verbose: bool = True
):
    """
    Train HTK HMM state detector from annotated videos.
    
    Args:
        video_paths: List of training video paths
        annotation_paths: List of corresponding CSV annotation files
        output_dir: Directory to save trained HMM models
        aruco_config_path: Path to ARUCO bin configuration JSON
        clip_model_path: Path to trained CLIP classifier model
        frame_skip: Process every Nth frame
        verbose: Show progress
    
    Training Data Requirements:
        - Each video must have corresponding CSV with columns:
          [timestamp_start, timestamp_end, state]
        - States must follow cycle: PICK → CARRY_WITH → PLACE → CARRY_WITHOUT
        - Videos should contain multiple complete cycles
        - Recommend 10-20 videos with 5-10 cycles each
    """
    # 1. Initialize components
    clip_model = AutoModel.from_pretrained(clip_model_path)
    clip_processor = AutoProcessor.from_pretrained(clip_model_path)
    aruco_detector = ArucoDetector()
    aruco_detector.load_bin_config(aruco_config_path)
    
    feature_extractor = FeatureExtractor(
        clip_model, clip_processor, aruco_detector
    )
    
    # 2. Extract features from all training videos
    training_data = []
    for video_path, annotation_path in zip(video_paths, annotation_paths):
        if verbose:
            print(f"Processing {video_path}...")
        
        # Extract features
        features, frame_numbers, fps = feature_extractor.extract_video_features(
            video_path, frame_skip
        )
        
        # Load annotations
        annotations = pd.read_csv(annotation_path)
        
        # Validate state cycle
        _validate_state_sequence(annotations)
        
        training_data.append((features, annotations))
    
    # 3. Train HTK HMM
    htk_detector = HTKStateDetector(output_dir)
    htk_detector.train(training_data, output_dir)
    
    if verbose:
        print(f"Training complete! Model saved to {output_dir}")
```

#### `detector.py` - Updated Main Interface
**Purpose**: Replace placeholder with real HTK-based detection

**Updated Function:**

```python
def detect_states_from_video(
    video_path: str,
    embeddings: List[np.ndarray],  # Keep for backward compatibility
    frame_numbers: List[int],
    fps: float,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Detect hand states for each frame in video using HTK HMM.
    
    Args:
        video_path: Path to video file
        embeddings: CLIP embeddings (not used by HTK, kept for compatibility)
        frame_numbers: Frame numbers corresponding to embeddings
        fps: Video frames per second
        htk_model_dir: Path to trained HTK HMM models
        aruco_config_path: Path to ARUCO configuration JSON
    
    Returns:
        DataFrame with columns [timestamp_start, timestamp_end, state]
    """
    if htk_model_dir is None:
        # Fall back to placeholder if no model provided
        return _placeholder_detection(frame_numbers, fps)
    
    # 1. Initialize components
    from .feature_extraction import FeatureExtractor
    from .aruco_detection import ArucoDetector
    from .htk_interface import HTKStateDetector
    
    # Load CLIP model for object confidence
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_processor = AutoProcessor.from_pretrained(MODEL)
    
    # Load ARUCO detector
    aruco_detector = ArucoDetector()
    if aruco_config_path:
        aruco_detector.load_bin_config(aruco_config_path)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        clip_model, clip_processor, aruco_detector
    )
    
    # 2. Extract features from video
    features, frame_nums, fps = feature_extractor.extract_video_features(
        video_path, frame_skip=4
    )
    
    # 3. Run HTK decoding
    htk_detector = HTKStateDetector(htk_model_dir)
    state_segments = htk_detector.decode(features, fps)
    
    return state_segments
```

#### `config.py` - Configuration
**Purpose**: Centralize HTK-related configuration

```python
"""Configuration for HTK HMM state detection."""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class HTKConfig:
    """HTK HMM configuration."""
    
    # Feature extraction
    feature_dim: int = 29  # 29D: motion/orientation/obj + HSV + ARUCO channels
    frame_skip: int = 4
    
    # HMM architecture
    num_states: int = 4  # PICK, CARRY_WITH, PLACE, CARRY_EMPTY
    num_mixtures: int = 3  # Gaussian mixtures per state
    
    # State cycle transition probabilities
    # Format: {from_state: {to_state: probability}}
    transition_probs: Dict[str, Dict[str, float]] = None
    
    # ARUCO configuration
    aruco_dict_type: str = "DICT_4X4_1000"
    aruco_distance_decay: float = 5.0  # Controls weight decay with distance
    
    # HTK paths
    htk_bin_dir: str = "/usr/local/bin"  # Location of HTK executables
    
    def __post_init__(self):
        if self.transition_probs is None:
            # Default cyclic transition probabilities
            self.transition_probs = {
                "PICK": {"PICK": 0.6, "CARRY_WITH": 0.4},
                "CARRY_WITH": {"CARRY_WITH": 0.6, "PLACE": 0.4},
                "PLACE": {"PLACE": 0.6, "CARRY_EMPTY": 0.4},
                "CARRY_EMPTY": {"CARRY_EMPTY": 0.6, "PICK": 0.3, "EXIT": 0.1}
            }

DEFAULT_HTK_CONFIG = HTKConfig()
```

---

## Integration with Existing Pipeline

### 1. Training Pipeline Integration

**Location**: `pipelines/video_training.py`

**Modifications Needed**:

```python
# In run_video_training():

# BEFORE state detection (if training HMM):
if args.train_hmm:
    from ..state_detection.training import train_state_detector
    
    train_state_detector(
        video_paths=[video_path],
        annotation_paths=[annotation_csv_path],  # User-provided
        output_dir=os.path.join(base_output_dir, "htk_models"),
        aruco_config_path=args.aruco_config,
        clip_model_path=MODEL,
        verbose=verbose
    )

# EXISTING: Video processing with state detection
video_embeddings, video_labels, video_paths, state_results = process_video_frames(
    video_path, label, clip_model, processor, cache_dir,
    save_frame_to_cache,
    threshold=threshold, 
    frame_skip=frame_skip,
    state_filter={HandState.CARRY_WITH.value},
    state_detection_func=lambda vp, emb, fn, fps: detect_states_from_video(
        vp, emb, fn, fps,
        htk_model_dir=os.path.join(base_output_dir, "htk_models"),
        aruco_config_path=args.aruco_config
    ),
    verbose=verbose
)
```

### 2. CLI Updates

**Current entrypoints**:
- `python -m symbiote.hmm_train`
- `python -m symbiote.hmm_tune`
- `python -m symbiote.hmm_infer`

**Pipeline modes**:
- Default: `--pipeline two-stage` (coarse HTK decode + subtype HTK models for 4-label output)
- Legacy: `--legacy` (single-stage 4-state decode)

**Stage-specific masks**:
- `--coarse-feature-mask` / `--coarse-feature-top-k`
- `--interact-feature-mask` / `--interact-feature-top-k`
- `--carry-feature-mask` / `--carry-feature-top-k`

### 3. Inference Pipeline Integration

**Location**: `pipelines/video_inference.py`

**No changes needed** - state detection is optional and not used during inference-only workflows.

---

## Data Requirements

### Training Data Format

#### 1. Annotation CSV Format
**Required columns**: `timestamp_start`, `timestamp_end`, `state`

**Example** (`video1_annotations.csv`):
```csv
timestamp_start,timestamp_end,state
0.0,1.2,CARRY_EMPTY
1.2,2.8,PICK
2.8,8.5,CARRY_WITH
8.5,10.1,PLACE
10.1,15.3,CARRY_EMPTY
15.3,17.0,PICK
17.0,22.8,CARRY_WITH
22.8,24.5,PLACE
24.5,30.0,CARRY_EMPTY
```

**Validation Rules**:
- States must follow cycle: PICK → CARRY_WITH → PLACE → CARRY_EMPTY → (repeat)
- No state skipping allowed
- Timestamps must be monotonically increasing
- No gaps between consecutive states

#### 2. ARUCO Configuration
**File**: `aruco_bins.json`

```json
{
  "marker_dict": "DICT_4X4_1000",
  "bins": {
    "0": {
      "type": "pick",
      "object": "apple",
      "description": "Red apple pick bin"
    },
    "1": {
      "type": "pick",
      "object": "banana",
      "description": "Yellow banana pick bin"
    },
    "2": {
      "type": "pick",
      "object": "orange",
      "description": "Orange pick bin"
    },
    "10": {
      "type": "place",
      "object": "apple",
      "description": "Red apple place bin"
    },
    "11": {
      "type": "place",
      "object": "banana",
      "description": "Yellow banana place bin"
    },
    "12": {
      "type": "place",
      "object": "orange",
      "description": "Orange place bin"
    }
  },
  "distance_decay": 5.0
}
```

**Setup Requirements**:
- Print ARUCO markers from DICT_4X4_1000 dictionary (IDs 0-999)
- Affix markers to bins in clear view
- Ensure markers are visible in all training/test videos
- Keep consistent lighting to avoid detection failures

### Training Data Volume

**Recommended**:
- **10-20 training videos**
- Each video: **5-10 complete pick-place cycles**
- Total: **50-200 state cycles**
- **Diverse conditions**: Different objects, lighting, hand speeds

**Minimum**:
- **5 training videos**
- Each video: **3-5 cycles**
- Total: **15-25 state cycles**

**Why This Much**:
- HTK HMMs need examples of within-state variability
- Transition probabilities estimated from cycle frequencies
- ARUCO context needs examples of different bin approaches

---

## ARUCO Detection Testing Tool

### Purpose
Before implementing the full HTK system, it's critical to validate that ARUCO detection and weighted context computation are working correctly. This testing tool allows visual verification of the ARUCO system.

### File Location
`symbiote/state_detection/test_aruco_detection.py`

### Implementation

```python
"""
ARUCO Detection Testing Tool

This script processes a video and outputs an annotated version showing:
- Detected ARUCO markers with bounding boxes
- Marker IDs and types (pick/place)
- Frame-by-frame weighted bin context score
- Hand position tracking

Usage:
    python -m symbiote.state_detection.test_aruco_detection \
        --video path/to/test_video.mp4 \
        --output path/to/output_annotated.mp4 \
        --aruco-config config/aruco_bins.json
"""

import argparse
import cv2
import sys
import os
import numpy as np

# Import from the actual pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from symbiote.lib.hand_detection import segment_hand, hand_pos
from symbiote.state_detection.aruco_detection import ArucoDetector


def test_aruco_detection(
    video_path: str,
    output_path: str,
    aruco_config_path: str,
    frame_skip: int = 1,
    verbose: bool = True
):
    """
    Process video and create annotated output with ARUCO detection visualization.
    
    Args:
        video_path: Input video file path
        output_path: Output annotated video file path
        aruco_config_path: Path to ARUCO configuration JSON
        frame_skip: Process every Nth frame (default 1 = all frames)
        verbose: Print progress
    """
    # Initialize ARUCO detector
    aruco_detector = ArucoDetector()
    aruco_detector.load_bin_config(aruco_config_path)
    
    # Initialize MediaPipe for hand detection
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3,
        max_num_hands=2
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frame skip: {frame_skip}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    # Statistics tracking
    weights_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames if needed
        if frame_count % frame_skip != 0:
            continue
        
        processed_count += 1
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand landmarks
        results = hands.process(frame_rgb)
        
        hand_position = None
        if results.multi_hand_landmarks:
            # Get right hand (leftmost in frame)
            hand_points = []
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for landmark in hand_landmarks.landmark[:21]:
                    points.append([landmark.x, landmark.y, landmark.z])
                hand_points.append(points)
            
            # Find leftmost hand (right hand from user perspective)
            hand_positions = [hand_pos(hp, frame_rgb) for hp in hand_points]
            right_hand_pos = min(hand_positions, key=lambda p: p[0])
            hand_position = right_hand_pos
        
        # Annotate frame using ARUCO detector's visualization
        if hand_position is not None:
            annotated_frame, weight = aruco_detector.visualize_bin_context(
                frame, hand_position
            )
            weights_history.append(weight)
        else:
            # No hand detected - just draw ARUCOs
            markers = aruco_detector.detect_markers(frame)
            annotated_frame = frame.copy()
            
            if len(markers['ids']) > 0:
                cv2.aruco.drawDetectedMarkers(
                    annotated_frame, markers['corners'], np.array(markers['ids'])
                )
            
            # Draw "No hand detected" message
            cv2.putText(annotated_frame, "No hand detected", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            weight = 0.0
            weights_history.append(0.0)
        
        # Add frame number and timestamp
        timestamp = frame_count / fps
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Time: {timestamp:.2f}s",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        if verbose and processed_count % 30 == 0:
            print(f"  Processed {processed_count} frames ({frame_count}/{total_frames})")
    
    # Release resources
    cap.release()
    out.release()
    hands.close()
    
    # Print statistics
    if verbose:
        print(f"\nProcessing complete!")
        print(f"  Total frames processed: {processed_count}")
        print(f"  Output saved to: {output_path}")
        
        if len(weights_history) > 0:
            print(f"\nWeight Statistics:")
            print(f"  Mean: {np.mean(weights_history):.3f}")
            print(f"  Std: {np.std(weights_history):.3f}")
            print(f"  Min: {np.min(weights_history):.3f}")
            print(f"  Max: {np.max(weights_history):.3f}")
            print(f"  PICK frames (>0.5): {np.sum(np.array(weights_history) > 0.5)}")
            print(f"  PLACE frames (<-0.5): {np.sum(np.array(weights_history) < -0.5)}")
            print(f"  Neutral frames: {np.sum(np.abs(weights_history) < 0.5)}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ARUCO detection and weighted bin context computation"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output annotated video file path"
    )
    parser.add_argument(
        "--aruco-config",
        type=str,
        default="config/aruco_bins.json",
        help="Path to ARUCO configuration JSON"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information"
    )
    
    args = parser.parse_args()
    
    test_aruco_detection(
        video_path=args.video,
        output_path=args.output,
        aruco_config_path=args.aruco_config,
        frame_skip=args.frame_skip,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
```

### Output Format

The annotated video will show:

1. **ARUCO Markers**: 
   - Green boxes/circles: PICK bins
   - Red boxes/circles: PLACE bins
   - Gray boxes/circles: Unknown/unconfigured markers
   - Marker IDs labeled next to each marker

2. **Hand Position**:
   - Purple circle: Detected hand center

3. **Bin Context Weight**:
   - Text at top: "Bin Context: +0.85" (example)
   - Color-coded bar:
     - Green bar extending right = Positive weight (PICK context)
     - Red bar extending left = Negative weight (PLACE context)
     - Length indicates strength

4. **Frame Info**:
   - Bottom of frame: Frame number and timestamp

### Usage Examples

**Basic usage:**
```bash
python -m symbiote.state_detection.test_aruco_detection \
    --video videos/test_pick_place.mp4 \
    --output videos/test_pick_place_annotated.mp4 \
    --aruco-config config/aruco_bins.json
```

**Skip frames for faster processing:**
```bash
python -m symbiote.state_detection.test_aruco_detection \
    --video videos/long_video.mp4 \
    --output videos/long_video_annotated.mp4 \
    --aruco-config config/aruco_bins.json \
    --frame-skip 5
```

### Validation Checklist

Use this tool to verify:

- [ ] **Marker Detection**: All physical ARUCOs are detected
- [ ] **Marker Classification**: IDs correctly mapped to pick/place types
- [ ] **Color Coding**: Green for pick bins, red for place bins
- [ ] **Hand Tracking**: Purple circle follows hand accurately
- [ ] **Weight Behavior**: 
  - [ ] Weight > 0.5 near pick bins
  - [ ] Weight < -0.5 near place bins
  - [ ] Weight ≈ 0 far from bins
  - [ ] Smooth transitions as hand moves
- [ ] **Distance Decay**: Weight decreases as hand moves away from bins
- [ ] **Multiple Markers**: Weights aggregate correctly with multiple visible markers

### Troubleshooting

**Issue: No markers detected**
- Check lighting (avoid glare)
- Verify marker size is sufficient (recommend 6-8 inches)
- Confirm ARUCO dictionary matches (DICT_4X4_1000 for 3-digit IDs)

**Issue: Wrong marker colors**
- Check `aruco_bins.json` configuration
- Verify marker IDs match physical setup

**Issue: Weights always near zero**
- Check `distance_decay` parameter (increase for stronger signal)
- Verify hand is actually near bins in test video
- Check marker visibility

**Issue: Hand tracking fails**
- Ensure hand is clearly visible
- Check lighting conditions
- Verify single hand in frame (or leftmost hand is tracked)

---

## Implementation Workflow

### Phase 1: Feature Extraction (Week 1)
1. Create `aruco_detection.py`
   - Implement ARUCO marker detection
   - Test on sample frames with printed markers
   - Validate bin distance computation

2. Create `feature_extraction.py`
   - Implement hand orientation cross product
   - Integrate ARUCO features
   - Integrate existing hand position/velocity
   - Add object confidence from CLIP
   - Test feature vector extraction on sample video

3. Validate feature quality
   - Visualize features over time
   - Check for NaN/inf values
   - Verify feature ranges are reasonable

### Phase 2: HTK Interface (Week 2)
1. Install HTK toolkit
   - Download from Cambridge University
   - Compile for your platform
   - Test basic HTK commands

2. Create `htk_interface.py`
   - Implement HTK file I/O (.mfc, .lab formats)
   - Create HMM configuration templates
   - Test training on synthetic data

3. Create `training.py`
   - Implement full training pipeline
   - Add validation checks
   - Test on 2-3 annotated videos

### Phase 3: Integration (Week 3)
1. Update `detector.py`
   - Replace placeholder implementation
   - Add HTK model loading
   - Test state detection on validation videos

2. Update `cli/main.py`
   - Add HMM training arguments
   - Add ARUCO config argument
   - Test CLI workflows

3. Update `pipelines/video_training.py`
   - Integrate HTK state detection
   - Test full pipeline with state filtering

### Phase 4: Validation (Week 4)
1. Collect training data
   - Record 10-15 annotated videos
   - Annotate state boundaries manually
   - Validate annotations follow state cycle

2. Train HMM
   - Run training pipeline
   - Check convergence (log-likelihood increases)
   - Validate transition probabilities

3. Evaluate performance
   - Frame-level accuracy
   - State boundary detection accuracy (within ±0.5s)
   - Transition validity (no illegal state jumps)

---

## Testing Strategy

### Unit Tests

**Test ARUCO Detection**:
```python
def test_aruco_detection():
    """Test ARUCO marker detection on synthetic image."""
    # Create test image with known ARUCO markers
    detector = ArucoDetector()
    markers = detector.detect_markers(test_image)
    
    assert len(markers['ids']) == 4
    assert markers['types'][0] == 'pick'
    # ... more assertions
```

**Test Feature Extraction**:
```python
def test_feature_extraction():
    """Test feature vector has correct shape and range."""
    extractor = FeatureExtractor(clip_model, clip_processor, aruco_detector)
    features = extractor.extract_frame_features(frame, landmarks, segmented, 0.0)
    
    assert features.shape == (17,)  # or 19
    assert np.all(np.isfinite(features))
    assert np.all(features[:2] >= 0) and np.all(features[:2] <= 1)  # Normalized positions
```

**Test HTK File I/O**:
```python
def test_htk_file_io():
    """Test HTK feature file writing and reading."""
    features = np.random.randn(100, 17)
    htk_detector = HTKStateDetector(".")
    htk_detector._write_htk_features(features, "test.mfc")
    
    # Read back and verify
    read_features = htk_detector._read_htk_features("test.mfc")
    assert np.allclose(features, read_features)
```

### Integration Tests

**Test Full Feature Extraction Pipeline**:
```python
def test_video_feature_extraction():
    """Test feature extraction on real video."""
    extractor = FeatureExtractor(clip_model, clip_processor, aruco_detector)
    features, frame_nums, fps = extractor.extract_video_features("test_video.mp4")
    
    assert features.shape[0] == len(frame_nums)
    assert features.shape[1] == 17
    assert fps > 0
```

**Test State Detection**:
```python
def test_state_detection():
    """Test HTK state detection on annotated video."""
    # Load test video with known annotations
    true_annotations = pd.read_csv("test_annotations.csv")
    
    # Run detection
    detected_states = detect_states_from_video(
        "test_video.mp4", [], [], 30.0,
        htk_model_dir="trained_models",
        aruco_config_path="aruco_bins.json"
    )
    
    # Check accuracy
    accuracy = compute_frame_accuracy(detected_states, true_annotations)
    assert accuracy > 0.8  # 80% frame-level accuracy
```

---

## Performance Targets

### Accuracy Metrics

**Frame-Level Accuracy**: >85%
- Percentage of frames correctly classified
- Computed by mapping state segments to frames

**State Boundary Accuracy**: ±0.5 seconds
- Transition detection within half-second of ground truth
- Critical for CARRY_WITH filtering

**Transition Validity**: 100%
- No illegal state transitions (e.g., PICK → PLACE)
- Enforced by HTK transition matrix

### Runtime Performance

**Feature Extraction**: <100ms per frame
- MediaPipe hand detection: ~5ms
- CLIP inference: ~50ms (with GPU)
- ARUCO detection: ~10ms
- Feature computation: <1ms

**HTK Decoding**: <50ms for 30-second video
- Viterbi algorithm scales linearly
- 17D features are lightweight

**Total Pipeline**: Real-time capable at 10-15 FPS

---

## Troubleshooting Guide

### Common Issues

**Issue**: HTK cannot find executables
- **Solution**: Add HTK bin directory to PATH or specify full paths in `htk_interface.py`

**Issue**: ARUCO markers not detected
- **Solution**: 
  - Check lighting (avoid glare)
  - Verify marker dictionary matches (DICT_4X4_1000)
  - Increase marker size (print larger)
  - Check camera focus

**Issue**: Hand orientation cross product gives NaN
- **Solution**:
  - MediaPipe landmarks may be missing (hand out of frame)
  - Add checks for landmark validity
  - Use previous frame's orientation as fallback

**Issue**: HMM gets stuck in one state
- **Solution**:
  - Increase transition probabilities (less "sticky")
  - Add more training data
  - Check feature normalization (large variance in one feature can dominate)

**Issue**: Object confidence always low
- **Solution**:
  - CLIP classifier may not be trained on correct objects
  - Check segmented hand images (may be too blurry)
  - Reduce blur threshold

---

## Dependencies

### New Python Packages
```bash
pip install opencv-contrib-python  # For ARUCO detection
pip install scipy  # For feature processing (if needed)
```

### HTK Toolkit
- **Download**: http://htk.eng.cam.ac.uk/
- **License**: HTK License (free for research)
- **Installation**: Follow platform-specific instructions
- **Required Tools**: HCompV, HERest, HVite, HHEd

### Configuration Files
```
config/
├── aruco_bins.json           # ARUCO marker configuration
└── htk_config.py             # HTK HMM hyperparameters
```

---

## Future Enhancements

### Short Term
1. **Adaptive thresholds**: Learn optimal bin distance threshold from data
2. **Multi-hand support**: Track both hands independently (requires changes)
3. **Online learning**: Update HMM incrementally with new data

### Long Term
1. **Deep HMM**: Replace Gaussian mixtures with neural feature extractors
2. **3D hand tracking**: Use depth cameras for better orientation estimation
3. **Object-specific models**: Train separate HMMs for different object types

---

## File Checklist

When implementing, create/modify these files in order:

- [ ] `symbiote/state_detection/config.py` - Configuration
- [ ] `symbiote/state_detection/aruco_detection.py` - ARUCO module
- [ ] `symbiote/state_detection/feature_extraction.py` - Feature extraction
- [ ] `symbiote/state_detection/htk_interface.py` - HTK interface
- [ ] `symbiote/state_detection/training.py` - Training pipeline
- [ ] `symbiote/state_detection/detector.py` - Update main function
- [ ] `symbiote/state_detection/__init__.py` - Update exports
- [ ] `symbiote/cli/main.py` - Add CLI arguments
- [ ] `symbiote/pipelines/video_training.py` - Integrate HMM
- [ ] `config/aruco_bins.json` - ARUCO configuration
- [ ] `tests/test_state_detection.py` - Unit tests

---

## Questions to Resolve Before Implementation

1. **ARUCO Dictionary**: Default DICT_4X4_1000 (IDs 0-999); choose different if needed.
2. **Bin Layout**: Physical setup of pick/place bins (distances, heights)?
3. **Camera Setup**: Fixed camera or hand-held? Resolution? Frame rate?
4. **Training Data**: Who will annotate videos? What annotation tool?
5. **HTK Installation**: Platform (Windows/Linux/Mac)? Pre-installed?
6. **Feature Dimensionality**: Use all 17-19 features or subset initially?
7. **Validation Split**: Hold out videos or use cross-validation?

---

**Implementation Effort Estimate**: 3-4 weeks (1 engineer)
**Training Data Collection**: 1-2 weeks (includes annotation)
**Total Time to Production**: 5-6 weeks

---

## symbiote_weak constrained boundary mode

The `symbiote_weak` variant now includes old-style HTK constraints designed for
boundary quality:

- **Picklist-specific sequences (default on)**: `hmm_tune` / two-stage infer use
  per-file label CSVs to build expected coarse / interact / carry sequences;
  use `--no-sequence-constraint` to disable. For `hmm_infer`, pass
  `--sequence-label-csv` or place `<video_stem>.csv` beside the video.
- ARUCO persistence (`--aruco-persistence-frames`) and smoothing
  (`--aruco-smoothing-window`) apply to **all three** ARUCO channels.
- boundary cleanup (`--min-segment-seconds`)
- boundary RMSE as primary tune objective in `symbiote_weak.hmm_tune`

Boundary reporting command:

`python -m symbiote_weak.hmm_boundary_eval --pred-dir <pred_csv_dir> --label-dir <gt_label_dir> --output-csv <summary.csv>`

---

**END OF IMPLEMENTATION GUIDE**
