# Docker Containerization: Complete Task List

## Prerequisites (Do First)

### 1. Install Docker Desktop
- [ ] Download Docker Desktop for Windows from https://www.docker.com/products/docker-desktop/
- [ ] Install Docker Desktop
- [ ] Restart your computer
- [ ] Open Docker Desktop and wait for it to start
- [ ] Verify installation: Open PowerShell and run `docker --version`

### 2. Download HTK Toolkit
- [ ] Go to http://htk.eng.cam.ac.uk/
- [ ] Register/login to download HTK
- [ ] Download `HTK-3.4.1.tar.gz`
- [ ] Save it to: `C:\Users\colin\Documents\coding\muc\symbiosis01252026\symbiotic-ai\HTK-3.4.1.tar.gz`

**Alternative**: Skip this and install HTK manually inside container later (see Option 2 in DOCKER_SETUP.md)

---

## Building the Container

### 3. Prepare the Dockerfile (if using HTK source)

- [ ] Open `Dockerfile` in your editor
- [ ] Find lines 28-35 (the commented HTK build section)
- [ ] Uncomment these lines by removing the `#` at the start:
  ```dockerfile
  COPY HTK-3.4.1.tar.gz /opt/htk/
  RUN cd /opt/htk && \
      tar -xzf HTK-3.4.1.tar.gz && \
      cd htk && \
      ./configure --prefix=/usr/local --disable-hlmtools && \
      make all && \
      make install && \
      cd / && \
      rm -rf /opt/htk
  ```
- [ ] Save the Dockerfile

### 4. Build the Docker Image

```powershell
# Open PowerShell
cd C:\Users\colin\Documents\coding\muc\symbiosis01252026\symbiotic-ai

# Build the image (takes 10-20 minutes first time)
docker build -t symbiotic-ai:latest .
```

**Apple Silicon (M1/M2/M3) note**: if you hit `pip install` failures, build as amd64:
```powershell
docker build --platform=linux/amd64 -t symbiotic-ai:latest .
```

- [ ] Run the build command
- [ ] Wait for build to complete
- [ ] Watch for any errors (particularly during HTK compilation)

### 5. Verify HTK Installation

```powershell
# Test HTK commands
docker run --rm symbiotic-ai:latest HCompV -h
docker run --rm symbiotic-ai:latest HERest -h
docker run --rm symbiotic-ai:latest HVite -h
docker run --rm symbiotic-ai:latest HParse -h
```

- [ ] Run each command
- [ ] Each should show HTK help output (not "command not found")
- [ ] If commands fail, HTK is not installed → see DOCKER_SETUP.md Option 2

---

## Setting Up Your Workspace

### 6. Create Required Directories

```powershell
cd C:\Users\colin\Documents\coding\muc\symbiosis01252026\symbiotic-ai

# Create directories if they don't exist
mkdir videos -ErrorAction SilentlyContinue
mkdir annotations -ErrorAction SilentlyContinue
mkdir config -ErrorAction SilentlyContinue
mkdir models -ErrorAction SilentlyContinue
mkdir outputs -ErrorAction SilentlyContinue
```

- [ ] Run the commands
- [ ] Verify directories exist in File Explorer

### 7. Prepare Your Data

**Place videos**:
- [ ] Copy your `.mp4` or `.mov` video files into `symbiotic-ai/videos/`

**Create ARUCO config** (if using state detection):
- [ ] Create `symbiotic-ai/config/aruco_bins.json`
- [ ] Use the template from DATA_REQUIREMENTS.md
- [ ] Fill in your ARUCO marker IDs and bin types
- [ ] Example structure:
  ```json
  {
    "marker_dict": "DICT_4X4_1000",
    "bins": {
      "91": {"type": "pick", "object": "apple"},
      "92": {"type": "place", "object": "apple"}
    },
    "distance_decay": 5.0
  }
  ```

**Create annotation CSV files** (for HTK HMM training):
- [ ] Create CSV files in `symbiotic-ai/annotations/` for each training video
- [ ] Format: `timestamp_start,timestamp_end,state`
- [ ] States must follow cycle: PICK → CARRY_WITH → PLACE → CARRY_EMPTY
- [ ] See DATA_REQUIREMENTS.md for detailed format

---

## Testing the Container

### 8. Basic Tests

```powershell
# Test 1: Show CLI help
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help

# Test 2: List your videos (verify volume mounts work)
docker-compose run --rm symbiotic-ai ls -la /data/videos

# Test 3: Check ARUCO config
docker-compose run --rm symbiotic-ai ls -la /data/aruco_config

# Test 4: Verify Python imports
docker-compose run --rm symbiotic-ai python -c "from symbiote.state_detection import HandState; print('Imports OK')"
```

- [ ] Run each test command
- [ ] Verify outputs are correct
- [ ] Videos should be listed
- [ ] Imports should not error

---

## Running Your First Pipeline

### 9. Train a Classifier (Basic Test)

```powershell
# Replace VIDEO.mov with your actual video filename
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/videos/VIDEO.mov \
    --label '["test_object_1","test_object_2","test_object_3"]' \
    --pca-dim 64 \
    --ilr-epochs 500 \
    --output-dir /models/classifier \
    --verbose
```

- [ ] Run the command with your video
- [ ] Watch for progress output
- [ ] Check `symbiotic-ai/models/classifier/VIDEO/` folder is created on Windows
- [ ] Verify files exist: `model_weights.pth`, `model_metadata.json`, plots

### 10. Run Inference (Get Predictions CSV)

```powershell
# Replace VIDEO_NAME with the folder created during training
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main infer \
    --video /data/videos/VIDEO.mov \
    --model-dir /models/classifier/VIDEO_NAME \
    --output /outputs/results.csv \
    --verbose
```

- [ ] Run the command
- [ ] Check `symbiotic-ai/outputs/results.csv` appears on Windows
- [ ] Open CSV and verify it has predictions

---

## ARUCO & State Detection Testing

### 11. Test ARUCO Detection (Optional)

```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.state_detection.test_aruco_detection \
    --video /data/videos/VIDEO.mov \
    --output /outputs/aruco_test.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --verbose
```

- [ ] Run the command
- [ ] Check `symbiotic-ai/outputs/aruco_test.mp4` is created
- [ ] Open the video in VLC or similar
- [ ] Verify ARUCO markers are detected and labeled
- [ ] Check bin context weight is displayed

---

## HTK HMM Training (Advanced)

### 12. Train HTK HMM State Detector

```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --verbose
```

**Prerequisites**:
- [ ] You have annotated CSV files with state labels
- [ ] CSVs follow PICK → CARRY_WITH → PLACE → CARRY_EMPTY cycle
- [ ] ARUCO config exists

**Run training**:
- [ ] Execute the command
- [ ] Wait for HTK training to complete (can take time)
- [ ] Check `symbiotic-ai/models/htk/models/hmm_final/` exists
- [ ] Verify HTK files: `macros`, `hmmdefs`, `wordlist`, `grammar`

### 13. Use HTK Model in Training Pipeline

```powershell
# Train classifier WITH state detection filtering
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/videos/VIDEO.mov \
    --label '["object_a","object_b","object_c"]' \
    --pca-dim 64 \
    --ilr-epochs 500 \
    --output-dir /models/classifier \
    --htk-model-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --verbose
```

- [ ] Run with HTK model
- [ ] Only CARRY_WITH frames will be cached for training
- [ ] Check `state_detection.csv` in output directory

---

### 14. Ground-Truth Overlay QA (Label Timing Check)

```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_gt_overlay \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /outputs/gt_overlay
```

- [ ] Run command
- [ ] Open one or more `_gt_overlay.mp4` files
- [ ] Verify label transitions align with visible hand actions
- [ ] Inspect `/outputs/gt_overlay/gt_overlay_index.csv`

---

## Troubleshooting Checklist

If something doesn't work:

- [ ] Docker Desktop is running
- [ ] You're in the `symbiotic-ai/` directory when running commands
- [ ] Volume mounts are correct (check `docker-compose.yml`)
- [ ] HTK is installed in container: `docker run --rm symbiotic-ai:latest HCompV -h`
- [ ] Files exist in expected locations: `docker-compose run --rm symbiotic-ai ls /data/videos`
- [ ] ARUCO config is valid JSON
- [ ] Annotation CSVs have correct format and state cycle

**Common fixes**:
- Rebuild image: `docker build -t symbiotic-ai:latest . --no-cache`
- Check container logs for errors
- Verify file permissions on Windows
- See DOCKER_SETUP.md Troubleshooting section

---

## Portability Test (Optional)

To verify your setup works on other machines:

### 14. Export Docker Image

```powershell
# Save image to file
docker save symbiotic-ai:latest | gzip > symbiotic-ai-image.tar.gz
```

- [ ] Create the image archive
- [ ] This file can be shared with others

### 15. Share Setup

What to share for someone else to run your system:
- [ ] `Dockerfile`
- [ ] `docker-compose.yml`
- [ ] `.dockerignore`
- [ ] Your `symbiote/` code folder
- [ ] `DOCKER_SETUP.md` documentation
- [ ] **OR** just the `symbiotic-ai-image.tar.gz` file (they load with `docker load`)

They can then:
1. Install Docker Desktop on their machine (Windows/Mac/Linux)
2. Either build from Dockerfile or load your image
3. Add their videos/configs
4. Run the same commands

---

## Summary Checklist

**Initial Setup**:
- [ ] Docker Desktop installed
- [ ] HTK downloaded (or will install manually)
- [ ] Container built successfully
- [ ] HTK verified in container
- [ ] Directories created

**Data Prepared**:
- [ ] Videos in `videos/` folder
- [ ] ARUCO config in `config/` folder (if using)
- [ ] Annotations in `annotations/` folder (if training HMM)

**Testing Complete**:
- [ ] Help command works
- [ ] Volume mounts verified
- [ ] Classifier training works
- [ ] Inference produces CSV
- [ ] ARUCO detection tested (optional)
- [ ] HTK HMM training tested (optional)

**Next Steps**:
- Use the system for your actual data
- See `DOCKER_QUICK_REFERENCE.md` for command shortcuts
- See `DATA_REQUIREMENTS.md` for detailed data format specs
- See `HTK_STATE_DETECTION_IMPLEMENTATION.md` for HMM system design

---

## Command Quick Reference

See `DOCKER_QUICK_REFERENCE.md` for a one-page reference of all Docker commands.
