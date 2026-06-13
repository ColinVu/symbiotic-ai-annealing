# Docker Quick Reference Card

## Build Container
```powershell
docker build -t symbiotic-ai:latest .
```

**Apple Silicon (M1/M2/M3) note**: if you hit `pip install` failures, build as amd64:
```powershell
docker build --platform=linux/amd64 -t symbiotic-ai:latest .
```

## Basic Commands

### Show Help
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help
```

### Train Classifier
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/videos/VIDEO.mov \
    --label '["item1","item2","item3"]' \
    --pca-dim 64 \
    --ilr-epochs 500 \
    --output-dir /models/classifier
```

### Inference (Get CSV)
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main infer \
    --video /data/videos/VIDEO.mp4 \
    --model-dir /models/classifier/VIDEO_NAME \
    --output /outputs/results.csv
```

### Train HTK HMM
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage
```

### Test ARUCO Detection
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.state_detection.test_aruco_detection \
    --video /data/videos/test.mp4 \
    --output /outputs/aruco_annotated.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json
```

### Ground-Truth Overlay QA
```powershell
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_gt_overlay \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /outputs/gt_overlay
```

## Path Mappings

| Your Windows Folder | Container Path |
|---------------------|----------------|
| `./videos/` | `/data/videos/` |
| `./annotations/` | `/data/annotations/` |
| `./config/` | `/data/aruco_config/` |
| `./models/` | `/models/` |
| `./outputs/` | `/outputs/` |

## Verify HTK Installation
```powershell
docker run --rm symbiotic-ai:latest HCompV -h
docker run --rm symbiotic-ai:latest HERest -h
docker run --rm symbiotic-ai:latest HVite -h
```

## Interactive Shell (Debug)
```powershell
docker-compose run --rm symbiotic-ai /bin/bash
```

## View Container Files
```powershell
docker-compose run --rm symbiotic-ai ls /app/symbiote
docker-compose run --rm symbiotic-ai ls /data/videos
```
