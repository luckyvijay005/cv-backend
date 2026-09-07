# Phase 2: Validation & Labeling — Workflow Guide

## Overview

Phase 2 takes the pipeline results and validates them against human judgment, measures accuracy, and tunes parameters.

**Timeline:** 1 week (5–7 days)

**Deliverables:**
- ✅ ~30 min of human-labeled interaction data
- ✅ Accuracy metrics (precision, recall, F1)
- ✅ Tuned thresholds and parameters
- ✅ Lesson learned for Phase 3

---

## Workflow (5 Days)

### Day 1: Generate Predictions

Run the pipeline on your full video to generate predictions:

```bash
cd cv-backend
python run.py --video ../training_footages.mp4 --output data/outputs/report.json --visualize
```

**Output:** `data/outputs/report.json` with top-3 students and all metrics.

### Day 2–3: Create Annotation Templates & Sample

Create skeleton files for labeling:

```bash
python create_annotation_template.py --video ../training_footages.mp4 --sample_minutes 5
```

**Output:**
- `data/labeled/frame_annotations.json` — (optional, for full frame-level tracking)
- `data/labeled/window_annotations.json` — (optional, for window classification)
- `data/labeled/interaction_labels.json` — **MAIN: Fill this in**

**Labeling Strategy (Minimal Effort):**

Option A: **Quick Spot-Check (2–3 hours)**
1. Watch your video with predicted tracklet IDs overlaid
2. Pick 5 random minute-long windows
3. For each window, manually mark each visible student as "interacting" or "isolated"
4. Record in `data/labeled/interaction_labels.json`

Option B: **Systematic Sampling (4–5 hours)**
1. Divide video into quartiles (first 25%, middle 50%, last 25%)
2. Sample 1–2 minute windows from each quartile
3. Label ~10 minutes total across the video
4. Record in `data/labeled/interaction_labels.json`

Option C: **Tool-Assisted (6–8 hours, more accurate)**
1. Use CVAT (https://github.com/opencv/cvat) to draw bboxes and assign tracklet IDs
2. Export annotations
3. Convert to schema in `ANNOTATION_SCHEMA.md`

**Recommended:** Option A or B for speed; Option C for maximum accuracy.

### Day 4–5: Validate & Tune

After labeling at least one minute sample, run validation:

```bash
python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report.json
```

**Output:**
- Accuracy (% correctly classified as isolated/interacting)
- Confusion matrix
- Precision/Recall per class
- Sample predictions vs ground truth

**Expected Performance (PoC):**
- Accuracy: ≥ 60% (2 of top 3 students match human judgment)
- Precision (isolation detection): ≥ 0.6
- Recall: ≥ 0.5

**If Performance is Weak:**

1. **Lower accuracy than expected?**
   - Check tracklet stability: are predicted tracklet IDs drifting?
   - Try: Lower tracker's `max_age` in `tracker.py` (line 30, default=30)
   - Run pipeline again and validate

2. **Too many false positives (flagging social kids as isolated)?**
   - Proximity threshold P too high (students counted as "far" when they're near)
   - Try: Lower `--proximity` flag (default 80 px → try 60 px)
   - Save different reports: `report_p60.json`, `report_p80.json`, compare

3. **Too many false negatives (missing isolated kids)?**
   - Proximity threshold P too low (overcounting peers)
   - Try: Raise `--proximity` flag (default 80 px → try 100 px)
   - Re-validate

**Tuning Loop Example:**

```bash
# Try proximity = 60 px
python run.py --video ../training_footages.mp4 --proximity 60 --output data/outputs/report_p60.json

# Validate
python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report_p60.json

# If accuracy improves, keep this setting; otherwise revert
```

### Day 5 (Continued): Tune Interaction Score Weights

The interaction score formula is (in `interaction_analyzer.py`):

```
score = (near_peer_minutes + 0.3 * approach_events) / total_minutes
```

If your validation shows systematic bias:

- **Low recall (missing isolated kids):** Increase weight of `isolated_minutes`:
  - Edit `interaction_analyzer.py`, line ~90
  - Change to: `score = (near_peer_minutes - 0.2 * isolated_minutes + 0.3 * approach_events) / total_minutes`
  - Re-run pipeline and validate

- **Too many false positives:** Reduce flexibility of interaction scoring:
  - Raise the isolation threshold: `threshold = 0.5` → `threshold = 0.3` (lower threshold = more sensitive)
  - Edit `validate.py`, line ~113

---

## Key Artifacts

| File | Format | Purpose |
|------|--------|---------|
| `data/outputs/report.json` | JSON | Predicted metrics for all students |
| `data/labeled/interaction_labels.json` | JSON | Human annotations (you fill this in) |
| `data/outputs/validation_metrics.json` | JSON | Validation report (accuracy, confusion matrix) |

---

## Example Annotation (Minimal)

**File:** `data/labeled/interaction_labels.json`

```json
{
  "metadata": {
    "sample_minutes": 2,
    "annotator": "Teacher or researcher",
    "date": "2026-04-18"
  },
  "minute_labels": [
    {
      "minute_id": 5,
      "frame_start": 9000,
      "frame_end": 10800,
      "timestamp_seconds": 300.0,
      "people": [
        {
          "tracklet_id": "1",
          "label": "interacting",
          "reason": "Playing with student 3 at table",
          "proximity_peers": 2
        },
        {
          "tracklet_id": "2",
          "label": "isolated",
          "reason": "Sitting alone playing with blocks",
          "proximity_peers": 0
        },
        {
          "tracklet_id": "3",
          "label": "interacting",
          "reason": "Playing with student 1",
          "proximity_peers": 1
        }
      ]
    },
    {
      "minute_id": 12,
      "frame_start": 21600,
      "frame_end": 23400,
      "timestamp_seconds": 720.0,
      "people": [
        {
          "tracklet_id": "1",
          "label": "interacting",
          "reason": "In group circle with teacher",
          "proximity_peers": 8
        },
        {
          "tracklet_id": "5",
          "label": "isolated",
          "reason": "Peeking in but not engaging",
          "proximity_peers": 0
        }
      ]
    }
  ]
}
```

**Then run validation:**

```bash
python validate.py \
  --labels data/labeled/interaction_labels.json \
  --report data/outputs/report.json
```

**Output:**
```
===========================================================================
VALIDATION REPORT
===========================================================================

Interaction Detection Accuracy: 80.00%
Samples Matched: 5
Isolation Threshold: 0.4 (score < threshold → isolated)

Confusion Matrix:
  (rows: ground truth, cols: predicted)
  Interacting (0) vs Isolated (1):
    True Negatives:  3  |  False Positives: 0
    False Negatives: 1  |  True Positives:  1

Per-Class Metrics:
  Interacting:
    Precision: 100.00%
    Recall: 75.00%
    F1-Score: 85.71%
  Isolated:
    Precision: 0.00%
    Recall: 50.00%
    F1-Score: 0.00%

Sample Predictions vs Ground Truth:
Tracklet         Predicted Score    Human Label     Match
1                0.45               interacting     ✗
2                0.12               isolated        ✓
3                0.52               interacting     ✓
5                0.18               isolated        ✓
...
```

---

## Success Criteria

| Metric | Target | Check |
|--------|--------|-------|
| Accuracy | ≥ 60% | Validation report |
| Labeled samples | ≥ 1 minute (5+ students) | Length of `interaction_labels.json` |
| Tuning iterations | ≥ 1 attempt | `data/outputs/report_pXX.json` variations |
| Documentation | Complete | This file + inline comments |

---

## Output for Phase 3

After Phase 2, you'll have:

1. ✅ Tuned proximity threshold and scoring weights
2. ✅ Validation accuracy metrics
3. ✅ Confidence in top-3 predictions (human-validated sample)
4. ✅ Identified failure modes and edge cases
5. ✅ Ready for Electron dashboard integration

---

## Troubleshooting

**Q: Tracklet IDs are inconsistent (student ID jumps frame-to-frame)**

A: Tracker is drifting. Options:
   - Lower `max_age` in `tracker.py` (line 30) from 30 → 20
   - Lower distance matching threshold (line 75) from 50 → 30
   - Re-run and re-validate

**Q: Validation script says "No matching tracklets"**

A: Check that your `interaction_labels.json` has `tracklet_id` as **string of the integer ID** (e.g., `"1"`, `"2"`, not `"GT-1"`).

```json
// ✗ Wrong
"tracklet_id": "GT-1"

// ✓ Correct
"tracklet_id": "1"
```

**Q: How do I know if accuracy is good?**

A: For a PoC, **≥ 60% accuracy is acceptable**. This means at least 2 of the top-3 flagged students are truly low-interaction per human judgment. In production, you'd aim for ≥ 80%.

---

## Next: Phase 3

Once Phase 2 is complete, proceed to:
- Build Electron dashboard to visualize reports
- Add manual correction UI for gather user feedback
- Prepare for pilot with real teachers
