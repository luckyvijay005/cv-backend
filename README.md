# Classroom Analytics PoC — Backend Setup & Usage

## Overview

This Python backend pipeline analyzes kindergarten CCTV footage to identify students with low social interaction based on proximity and peer engagement metrics.

**Components:**
- `detector.py` — YOLOv8-based person detection
- `tracker.py` — Simple multi-object tracker (centroid-based, scalable to ByteTrack)
- `interaction_analyzer.py` — Proximity-based interaction scoring
- `pipeline.py` — Orchestration and metrics computation
- `utils.py` — Video I/O, visualization helpers
- `run.py` — CLI entry point

## Setup

### 1. Create Python Environment

```bash
cd cv-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` — YOLOv8 models
- `opencv-python` — Video processing
- `torch`, `torchvision` — Deep learning framework
- `scipy`, `numpy`, `tqdm` — Utilities

**GPU Support (Optional):** If you have CUDA-capable GPU (RTX 4070), dependencies above include CUDA-enabled PyTorch. Verify with:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Download YOLOv8 Weights (First Run)

On first execution, YOLOv8 weights (~80 MB) will auto-download. You can pre-download with:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

## Running the Pipeline

### Basic Usage

```bash
python run.py --video ../training_footages.mp4 --output data/outputs/report.json
```

### With Options

```bash
# Test on first 5000 frames (~2-3 min at 30 FPS)
python run.py --video ../training_footages.mp4 --output data/outputs/report_test.json --max_frames 5000

# Use smaller/faster model
python run.py --video ../training_footages.mp4 --model yolov8n.pt --output data/outputs/report.json

# Save visualization frames
python run.py --video ../training_footages.mp4 --output data/outputs/report.json --visualize

# Adjust proximity threshold (pixels)
python run.py --video ../training_footages.mp4 --proximity 100.0 --output data/outputs/report.json
```

### Output Format

Report JSON (`data/outputs/report.json`):

```json
{
  "date": "2026-04-18",
  "video_file": "../training_footages.mp4",
  "total_frames": 1234,
  "total_students": 15,
  "duration_seconds": 41.1,
  "proximity_threshold_px": 80.0,
  "top_3_least_interactive": [
    {
      "track_id": 3,
      "score": 0.15,
      "isolated_minutes": 0.45,
      "near_peer_minutes": 0.05,
      "approach_events": 2,
      "confidence": 0.8
    },
    ...
  ],
  "all_metrics": {
    "1": {...},
    "2": {...},
    ...
  }
}
```

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | — | Path to input video (required) |
| `--output` | `data/outputs/report.json` | Path for output JSON report |
| `--max_frames` | None | Limit frames to process (for testing) |
| `--proximity` | 80.0 px | Proximity threshold (~1.5m in typical classroom) |
| `--model` | `yolov8s.pt` | YOLOv8 variant: `yolov8n` (nano, fast), `yolov8s` (small), `yolov8m` (medium) |
| `--device` | 0 | GPU index (0=first GPU, -1=CPU) |
| `--visualize` | False | Save visualization frames to `data/outputs/` |

## Key Metrics

**Interaction Score** (per student, 0–1):
- **Low (0.0–0.3)**: Mostly isolated, should be flagged for teacher review
- **Medium (0.3–0.6)**: Some peer interaction, typical
- **High (0.6–1.0)**: Frequent peer interaction, very social

**Components:**
- `isolated_minutes` — Time spent with no peers nearby
- `near_peer_minutes` — Time spent with ≥1 peer within proximity threshold
- `approach_events` — Number of transitions from isolated to near peers (captures social movement)
- `confidence` — Reliability of score (1.0 = consistently visible, 0.0 = rarely visible)

## Known Limitations (PoC)

1. **Simple Tracker:** Centroid-based distance matching. Can drift on fast motion or occlusion. To improve: integrate ByteTrack or DeepSORT.

2. **Proximity Heuristic:** Fixed 80 px ≈ 1.5 m calibration. For better accuracy, provide classroom dimensions or known seat spacing.

3. **No Teacher Filtering:** All detected people are treated as students. Future: add lightweight teacher detector to skip teacher presence.

4. **No Pose/Orientation:** Interaction is proximity-only. Future: add head+torso orientation for "active engagement" detection.

5. **No Schedule Context:** All video time is "allowed." Future: ingest classroom schedule to skip lectures, exams, timed tasks.

6. **Single Camera:** No multi-view fusion. Seat-based isolation heatmaps will be less precise.

7. **Clothing Similarity:** Kindergarten kids often wear similar outfits. Re-ID across days will be challenging; 24h SID expiry is important.

## Troubleshooting

### GPU Not Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

If False, ensure NVIDIA drivers and CUDA 11.8+ are installed.

### Memory Error on Full Video

Reduce with:
```bash
python run.py --video ../training_footages.mp4 --max_frames 5000 --model yolov8n.pt
```

### Slow Inference

- Use smaller model: `--model yolov8n.pt` (faster, less accurate)
- Reduce video resolution (preprocess frames before detector)
- Ensure GPU is being used: `--device 0` (not CPU)

### Validation Report Not Found

Check if `data/outputs/` exists:
```bash
ls -la data/outputs/
```

Create manually if needed:
```bash
mkdir -p data/outputs
```

## Next Steps (After PoC)

1. **Manual Validation (Week 2):**
   - Label ~30 min of video with ground-truth tracklet IDs and interaction labels
   - Compare predicted vs. human labels
   - Compute precision/recall

2. **Refinement (Week 3):**
   - Add teacher detection + filter
   - Implement schedule-based windowing
   - Tune proximity threshold based on validation results

3. **Frontend Integration (Week 4):**
   - Add dashboard page to Electron app
   - Load and display JSON reports
   - Implement human-in-the-loop correction UI

4. **Re-ID & Multi-Camera (Future):**
   - Add clothing + appearance embeddings for same-person re-identification
   - Implement 24h SID expiry + re-assignment logic
   - Extend to multi-camera with cross-view matching

## Privacy & Ethical Notes

- **Raw Video:** Deleted after processing (not retained in `data/outputs/`)
- **Stored Metrics:** Only aggregated per-student scores and event counts; no face/biometric data stored
- **Embeddings:** Transient during inference; not saved long-term
- **Human Review Mandatory:** All recommendations must be reviewed by a teacher before any action
- **Confidence Flags:** Flagged low-reliability detections (occluded, noisy) for extra scrutiny
- **Bias Monitoring:** Use this PoC to **validate** that flags are fair (not biased by seat location, lighting, etc.)

## Contact & Support

For issues or questions, refer to:
- Main spec: `../SPEC.md` (if available)
- Electron frontend: `../my-app/`
- Data: `../training_footages.mp4`
