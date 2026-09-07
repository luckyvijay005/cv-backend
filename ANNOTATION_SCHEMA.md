# Annotation Schema for Classroom Analytics Validation

## Overview

This document defines the format for manually labeling a ~30 minute sample of kindergarten classroom footage to validate the automated interaction detection pipeline.

## Annotation Types

We collect **three types of annotations**:

### 1. Frame-Level Bounding Boxes + Tracklet IDs

For every **Nth frame** (e.g., every 30th frame = 1 per second at 30 FPS), manually draw bounding boxes around each visible person and assign a **consistent tracklet ID**.

**File:** `data/labeled/frame_annotations.json`

**Format:**
```json
{
  "frames": [
    {
      "frame_id": 0,
      "timestamp_seconds": 0.0,
      "people": [
        {
          "person_id": 1,
          "tracklet_id": "GT-1",
          "bbox_x1": 100,
          "bbox_y1": 50,
          "bbox_x2": 180,
          "bbox_y2": 200,
          "visible": true,
          "occlusion_level": "none"
        },
        {
          "person_id": 2,
          "tracklet_id": "GT-2",
          "bbox_x1": 300,
          "bbox_y1": 80,
          "bbox_x2": 380,
          "bbox_y2": 250,
          "visible": true,
          "occlusion_level": "partial"
        }
      ]
    },
    ...
  ]
}
```

**Keys Explained:**
- `frame_id` — Frame index (0-based)
- `timestamp_seconds` — Video time (for reference)
- `person_id` — Unique person in this frame (1, 2, 3, ... per frame only)
- `tracklet_id` — **Consistent across frames** (e.g., "GT-1" always refers to the same student)
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2` — Bounding box corners (pixels)
- `visible` — Is the person visible? (true/false)
- `occlusion_level` — "none" | "partial" | "heavy" (for confidence flags)

**Sampling Strategy (to reduce labeling burden):**
- If video is 1 hour (108k frames), sample every 30 frames = ~1,200 labeled frames ≈ 30 min effort
- Or: label **key minutes only** (e.g., 5–10 minute windows) where you observe high/low interaction

### 2. Minute-Level Window Classifications

For each minute of the sampled period, classify the **activity type**:

**File:** `data/labeled/window_annotations.json`

**Format:**
```json
{
  "windows": [
    {
      "window_id": 0,
      "frame_start": 0,
      "frame_end": 1800,
      "timestamp_start_seconds": 0.0,
      "timestamp_end_seconds": 60.0,
      "window_type": "break",
      "notes": "Kids playing freely at recess"
    },
    {
      "window_id": 1,
      "frame_start": 1800,
      "frame_end": 3600,
      "timestamp_start_seconds": 60.0,
      "timestamp_end_seconds": 120.0,
      "window_type": "structured_activity",
      "notes": "Circle time, teacher-led"
    }
  ]
}
```

**Window Types:**
- `break` — Unstructured free play (high interaction expected)
- `structured_activity` — Teacher-led activity (interaction varies)
- `transition` — Moving between activities (chaotic, skip for now)
- `free_play` — Unstructured outdoor/indoor play
- `meal_time` — Lunch/snack (social interaction common)

### 3. Interaction Labels (Minute-Level, Key Frames Only)

For **5–10 key minutes**, manually label each visible person as **"interacting"** or **"isolated"**.

**File:** `data/labeled/interaction_labels.json`

**Format:**
```json
{
  "minute_labels": [
    {
      "minute_id": 5,
      "frame_start": 9000,
      "frame_end": 10800,
      "timestamp_seconds": 300.0,
      "people": [
        {
          "tracklet_id": "GT-1",
          "label": "interacting",
          "reason": "Playing with GT-3 and GT-5",
          "proximity_peers": 2
        },
        {
          "tracklet_id": "GT-2",
          "label": "interacting",
          "reason": "Standing in group circle",
          "proximity_peers": 4
        },
        {
          "tracklet_id": "GT-8",
          "label": "isolated",
          "reason": "Sitting alone in corner, no peers nearby",
          "proximity_peers": 0
        }
      ]
    },
    {
      "minute_id": 12,
      "frame_start": 21600,
      "frame_end": 23400,
      "people": [...]
    }
  ]
}
```

**Keys:**
- `label` — "interacting" or "isolated"
- `reason` — Short description (for validation pass/fail analysis)
- `proximity_peers` — Count of peers within ~1.5m (human observation)

---

## Labeling Workflow

### Option A: Semi-Automatic (Recommended for PoC)

1. **Run the pipeline** on your full video to get predicted tracklet IDs and bounding boxes
2. **Spot-check** 5–10 random minutes: 
   - View predicted bboxes & track IDs in video player or visualization
   - Manually verify tracklet IDs are correct
   - Correct any ID mismatches or missed detections
   - Annotate interaction labels (interacting vs. isolated)
3. **Save corrections** to `data/labeled/interaction_labels.json`

### Option B: Manual (More Accurate, Requires Tool)

1. **Use a labeling tool:**
   - **CVAT** (free, open-source): https://github.com/opencv/cvat
   - **VIA** (simple, in-browser): https://www.robots.ox.ac.uk/~vgg/software/via/
   - **LabelImg** (offline): https://github.com/heartexlabs/labelImg
   
2. **Label frame-level bboxes** (every 30th frame or key moments)
3. **Export annotations** and convert to schema above

### Option C: Quick Validation (Minimal Effort)

1. Just annotate **5 key minutes** with interaction labels
2. Compute human-AI agreement on those 5 minutes
3. Use agreement % as confidence score for full video ranking

---

## Annotation Tools & Templates

### Python Helper: Create Empty Annotation Template

```bash
cd cv-backend
python create_annotation_template.py --video ../training_footages.mp4 --sample_minutes 30
```

This generates a skeleton JSON file you can fill in.

### Visualization: View Predicted Bboxes

```bash
python visualize_predictions.py --report data/outputs/report.json --video ../training_footages.mp4
```

This plays the video with predicted tracklets overlaid, so you can spot-check IDs.

---

## Validation Metrics

After labeling, run:

```bash
python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report.json
```

This outputs:
- **Detection Recall:** % of labeled people found by detector
- **Tracking IDF1:** How well predicted tracklet IDs match ground truth
- **Interaction Accuracy:** % of students correctly classified as isolated/interacting
- **Confusion Matrix:** True/False positives for isolation detection

---

## Example: Minimal Labeling Plan (2–3 hours)

**Target:** Label 30 min of key footage

1. **Sample 5–10 minute windows** distributed throughout the video
   - Early (min 0–10): see initial grouping
   - Middle (min 20–30): typical activity
   - Late (min 50–60): fatigue effects?

2. **Per window:**
   - Watch video once to get sense of activity
   - Manually mark each visible person: "interacting" or "isolated"
   - Note any obvious outliers (very social or very quiet students)

3. **Output:** `data/labeled/interaction_labels.json` with ~100–200 person-minute labels

4. **Validation:** Run `validate.py` → see agreement %

---

## Privacy & Ethics Notes

- **Store only labels, not video or faces**
- **De-identify tracklets** (use "GT-1", "GT-2", etc., not student names)
- **Minimal retention:** Delete raw video after validation; keep only aggregated metrics
- **Use for validation only:** Do not distribute labeled data externally

---

## Questions?

Refer to:
- Plan: `../claude-code/plans/modular-seeking-clarke.md`
- Implementation: `../cv-backend/README.md`
- Metrics: See `validate.py` output
