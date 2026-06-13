# Docker Setup Guide for Symbiotic AI

## What Containerization Does (Layman's Terms)

**Think of Docker like a "virtual computer in a box":**

- **Normal setup**: You install Python, libraries, HTK directly on your Windows machine. Different machines need different installations.
- **Docker setup**: You create a "recipe" (Dockerfile) that describes a complete Linux environment with Python + libraries + HTK. Docker builds a self-contained package that runs the same way on *any* machine (Windows, Mac, Linux).

**Benefits:**
1. **No Windows HTK headaches**: HTK runs in Linux inside the container.
2. **Perfect portability**: Send someone the Docker image or Dockerfile → they run the exact same environment.
3. **Clean isolation**: Container has only what it needs; doesn't mess with your Windows Python install.
4. **Reproducible**: Same code + same Docker image = same results always.

---

## Step-by-Step Setup Instructions

### Prerequisites

1. **Install Docker Desktop** for Windows:
   - Download from: https://www.docker.com/products/docker-desktop/
   - Install and restart your computer
   - Open Docker Desktop and let it finish starting up

2. **Download HTK** (required for HMM training):
   - Go to: http://htk.eng.cam.ac.uk/
   - Register and download `HTK-3.4.1.tar.gz`
   - Place it in `symbiotic-ai/` directory (same folder as Dockerfile)
   
   **Alternative**: Skip this for now and install HTK manually in the container later (see Option 2 below)

---

### Building the Container

#### Option 1: With HTK Source (Recommended)

If you have `HTK-3.4.1.tar.gz` in your `symbiotic-ai/` folder:

1. **Edit Dockerfile** - Uncomment lines 28-35:
   ```dockerfile
   # Change from:
   # COPY HTK-3.4.1.tar.gz /opt/htk/
   
   # To:
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

2. **Build the image**:
   ```powershell
   cd C:\Users\colin\Documents\coding\muc\symbiosis01252026\symbiotic-ai
   docker build -t symbiotic-ai:latest .
   ```

   **Apple Silicon (M1/M2/M3) note**: if you hit `pip install` failures (common for `mediapipe` on arm64),
   build as amd64:
   ```powershell
   docker build --platform=linux/amd64 -t symbiotic-ai:latest .
   ```
   
   This takes 10-20 minutes the first time (downloading base image, compiling HTK, installing Python packages).

3. **Verify HTK is installed**:
   ```powershell
   docker run --rm symbiotic-ai:latest HCompV -h
   ```
   
   You should see HTK help output.

#### Option 2: Without HTK (Install Later)

If you don't have HTK yet or want to try the container first:

1. **Build without HTK**:
   ```powershell
   docker build -t symbiotic-ai:latest .
   ```

   **Apple Silicon (M1/M2/M3) note**:
   ```powershell
   docker build --platform=linux/amd64 -t symbiotic-ai:latest .
   ```

2. **Install HTK inside running container** (see "Manual HTK Installation" section below)

---

### Directory Structure Setup

Create the following directories in `symbiotic-ai/` if they don't exist:

```powershell
cd C:\Users\colin\Documents\coding\muc\symbiosis01252026\symbiotic-ai
mkdir videos -ErrorAction SilentlyContinue
mkdir annotations -ErrorAction SilentlyContinue
mkdir config -ErrorAction SilentlyContinue
mkdir models -ErrorAction SilentlyContinue
mkdir outputs -ErrorAction SilentlyContinue
```

Put your files here:
- **videos/**: Your `.mp4`, `.mov` video files
- **annotations/**: CSV files with state annotations (for HMM training)
- **config/**: `aruco_bins.json` configuration file
- **models/**: Training outputs (created automatically)
- **outputs/**: Inference results (created automatically)

---

## Using the Containerized Program

### Quick Commands Reference

#### Using Docker Compose (Easier)

**View help**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help
```

**Train classifier from video**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/videos/IMG_5293.mov \
    --label '["apple","apple","banana"]' \
    --pca-dim 64 \
    --ilr-epochs 500 \
    --output-dir /models/classifier
```

**Run inference on video (output CSV)**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main infer \
    --video /data/videos/test.mp4 \
    --model-dir /models/classifier/IMG_5293 \
    --output /outputs/results.csv
```

**Train HTK HMM state detector (two-stage default)**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage
```

**Train with stage-specific masks**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --coarse-feature-top-k 8 \
    --interact-feature-top-k 6 \
    --carry-feature-top-k 6 \
    --tune-decode
```

**Test ARUCO detection**:
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.state_detection.test_aruco_detection \
    --video /data/videos/test.mp4 \
    --output /outputs/aruco_test.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json
```

#### Using Docker Run (More Verbose)

Same commands but longer syntax:

```powershell
docker run --rm -it \
    -v ${PWD}/videos:/data/videos:ro \
    -v ${PWD}/annotations:/data/annotations:ro \
    -v ${PWD}/config:/data/aruco_config:ro \
    -v ${PWD}/models:/models:rw \
    -v ${PWD}/outputs:/outputs:rw \
    symbiotic-ai:latest \
    python -m symbiote_weak.cli.main train --video /data/videos/IMG_5293.mov --label '["apple","apple","banana"]'
```

---

## How Inputs/Outputs Work in Docker

### Path Mapping (Volume Mounts)

Docker creates a mapping between your Windows folders and container folders:

| Windows Path | Container Path | Purpose |
|--------------|----------------|---------|
| `./videos/` | `/data/videos/` | Input videos |
| `./annotations/` | `/data/annotations/` | CSV annotations |
| `./config/` | `/data/aruco_config/` | ARUCO config JSON |
| `./models/` | `/models/` | Output models (read-write) |
| `./outputs/` | `/outputs/` | Output CSVs/videos (read-write) |

**Example:**
- You put `IMG_5293.mov` in `C:\Users\colin\...\symbiotic-ai\videos\`
- In the Docker command, reference it as `/data/videos/IMG_5293.mov`
- Output goes to `/models/classifier/IMG_5293/` → appears in Windows at `C:\Users\colin\...\symbiotic-ai\models\classifier\IMG_5293\`

### Important Notes

1. **Use container paths** in commands (e.g. `/data/videos/...` not `./videos/...`)
2. **Output directories are automatically created** inside container
3. **Files appear immediately** on Windows after container writes them (volumes are live-synced)

---

## Manual HTK Installation (If Skipped During Build)

If you built the container without HTK, install it manually:

1. **Start an interactive container**:
   ```powershell
   docker run --rm -it symbiotic-ai:latest /bin/bash
   ```

2. **Inside the container**, download and build HTK:
   ```bash
   cd /opt
   wget http://htk.eng.cam.ac.uk/prot-docs/HTK-3.4.1.tar.gz  # Requires auth
   tar -xzf HTK-3.4.1.tar.gz
   cd htk
   ./configure --prefix=/usr/local --disable-hlmtools
   make all
   make install
   ```

3. **Commit the container as a new image**:
   - In another terminal (keep container running):
   ```powershell
   docker ps  # Get container ID
   docker commit <container-id> symbiotic-ai:latest
   ```

4. **Verify**:
   ```powershell
   docker run --rm symbiotic-ai:latest HCompV -h
   ```

---

## Does Containerization Break Imports?

**No.** Your Python code doesn't change at all. The imports work identically inside the container:

```python
from symbiote.state_detection import HandState, detect_states_from_video
```

This works because:
- Your code is copied into `/app/symbiote/` in the container
- `PYTHONPATH` is set to `/app`
- Python finds modules the same way as on your Windows machine

**The only thing that changes**: You run commands via `docker-compose run` instead of directly via `python`.

---

## Portability: Mac/Linux

**Yes, containerization makes porting trivial:**

1. **Send someone**:
   - Your Dockerfile
   - Your code (`symbiote/` folder)
   - Optional: Pre-built Docker image (export with `docker save`)

2. **They run** (on Mac/Linux):
   ```bash
   docker build -t symbiotic-ai .
   docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help
   ```

3. **It works identically** because:
   - Same Linux environment inside container regardless of host OS
   - Same Python version, same libraries, same HTK
   - Only difference: volume mount syntax might vary slightly on Mac/Linux (usually automatic)

**Bottom line**: "Build once, run anywhere" (Windows, Mac, Linux, cloud servers)

---

## Troubleshooting

### Docker Desktop Not Starting
- Restart Docker Desktop
- Check "WSL 2 based engine" is enabled in Docker Desktop settings
- Update WSL: `wsl --update` in PowerShell (admin)

### Build Fails During HTK Compilation
- Make sure you uncommented the HTK build section in Dockerfile
- Verify `HTK-3.4.1.tar.gz` is in the same folder as Dockerfile
- Try Option 2 (manual installation) instead

### "Permission Denied" on Output Directories
- On Windows: Usually not an issue
- On Linux: Run `chmod -R 777 models outputs` to allow container to write

### Container Can't Find Videos
- Check volume mounts in `docker-compose.yml`
- Use **absolute paths inside container**: `/data/videos/file.mp4`
- Verify file exists: `docker-compose run --rm symbiotic-ai ls /data/videos`

### Import Errors Inside Container
- Check code was copied: `docker-compose run --rm symbiotic-ai ls /app/symbiote`
- Verify PYTHONPATH: `docker-compose run --rm symbiotic-ai printenv PYTHONPATH`

---

## Next Steps

1. Build the container
2. Run a simple test: `docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help`
3. Try training on a small video
4. Set up ARUCO config and test detection
5. Train HTK HMM with annotated videos

See `DATA_REQUIREMENTS.md` for what data you need to prepare.
