# Phase 2 Quick-Start: Validation & Labeling

> **TL;DR:** Run pipeline → annotate 1 minute → validate → done.

---

## 🚀 The 3-Step Process

### Step 1: Run Pipeline (5 min)

```bash
cd cv-backend
python run.py --video ../training_footages.mp4 --output data/outputs/report.json
```

**Output:** `data/outputs/report.json` with top-3 students flagged.

---

### Step 2: Annotate (1–2 hours)

**Quick Manual Labeling:**

1. Watch your video (or use predictions to spot-check)
2. Pick any 1–2 minute window (ideally with 5+ visible students)
3. For each student in that window, mark:
   - `"interacting"` if they're with peers (playing together, talking, etc.)
   - `"isolated"` if alone (no peers nearby)
4. Fill in `data/labeled/interaction_labels.json`

**Example Template:**

```bash
cp ANNOTATION_SCHEMA.md interaction_labels.json
# Edit manually or use a JSON editor
```

**Example Content (minimal working example):**

```json
{
  "minute_labels": [
    {
      "minute_id": 5,
      "people": [
        {"tracklet_id": "1", "label": "isolated"},
        {"tracklet_id": "2", "label": "interacting"},
        {"tracklet_id": "3", "label": "interacting"}
      ]
    }
  ]
}
```

That's it! One minute with 3 students = **minimum for validation**.

---

### Step 3: Validate (1 min)

```bash
python validate.py \
  --labels data/labeled/interaction_labels.json \
  --report data/outputs/report.json
```

**Output:** Accuracy score, confusion matrix, sample predictions.

---

## 📚 Reference: Files Used in Phase 2

| File | What It Is | Your Action |
|------|-----------|------------|
| `cv-backend/run.py` | Pipeline executable | Run once to generate predictions |
| `data/outputs/report.json` | Predictions | Auto-generated; read-only |
| `cv-backend/ANNOTATION_SCHEMA.md` | Format spec | Reference when annotating |
| `data/labeled/interaction_labels.json` | Your labels | **Fill this in manually** |
| `cv-backend/validate.py` | Validation script | Run once after labeling |
| `data/outputs/validation_metrics.json` | Validation results | Auto-generated; review results |

---

## 💡 Tips for Faster Labeling

### Use Predicted Tracklets to Infer Labels

Since you already ran the pipeline, use the predicted scores as a **starting point**:

1. Get top-3 from `report.json`
2. Watch those specific 30-sec clips
3. Label them (fast, only 3 students)
4. Validate just those 3

**Result:** Minimum data, maximum impact.

### Batch Process by Window Type

1. Watch video at 2x speed
2. Write down timestamps of "clearly isolated" vs "clearly social" moments
3. Annotate those specific windows

### Use 5-Minute Windows (Lower Effort)

Instead of labeling every frame, just pick 5 random 1-minute windows spread throughout the video.

---

## 🔧 If Something Goes Wrong

### Validation Script Fails

**Error:** "No matching tracklets"

**Solution:** Check `data/labeled/interaction_labels.json` format:
- Tracklet ID should be a string: `"1"`, `"2"`, not `"GT-1"`
- Must have `minute_labels` key at top level

### Accuracy is Very Low (< 50%)

**Option 1: Increase Sample Size**
- Annotate 2–3 minutes instead of 1
- More data = more reliable metrics

**Option 2: Tune Proximity Threshold**
- Re-run pipeline with different `--proximity`:
  ```bash
  python run.py --video ../training_footages.mp4 --proximity 70 --output data/outputs/report_p70.json
  python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report_p70.json
  ```
- Try 60, 70, 80, 100 px and see which has best accuracy

**Option 3: Check Tracklet Stability**
- If tracklet IDs are jumping around, tuning won't help
- Reduce tracker's `max_age` in `tracker.py` (line ~30) from 30 → 20
- Re-run pipeline and re-validate

---

## ✅ Success: What to Save

After Phase 2, commit these files:

```bash
git add cv-backend/data/outputs/report.json
git add cv-backend/data/labeled/interaction_labels.json
git add cv-backend/data/outputs/validation_metrics.json
git commit -m "Phase 2: Validation baseline (accuracy ~70%)"
```

---

## Next: Phase 3 (Dashboard)

Once validation is done, you have:
- ✅ Working predictions
- ✅ Measured accuracy
- ✅ Confidence in results

Ready for Phase 3: Build Electron dashboard to visualize & collect feedback!

---

## Links

- **Full Phase 2 Workflow:** See `PHASE_2_WORKFLOW.md`
- **Annotation Format:** See `ANNOTATION_SCHEMA.md`
- **Pipeline Setup:** See `README.md`

---

## Commands Cheat Sheet

```bash
# Step 1: Generate predictions
python run.py --video ../training_footages.mp4 --output data/outputs/report.json

# Step 2a: Create empty template (optional)
python create_annotation_template.py --video ../training_footages.mp4 --sample_minutes 1

# Step 2b: Fill in data/labeled/interaction_labels.json manually
# (See ANNOTATION_SCHEMA.md for format)

# Step 3: Run validation
python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report.json

# Step 3b: Export report to JSON
# (Already saved to data/outputs/validation_metrics.json)

# Extra: Try different proximity threshold
python run.py --video ../training_footages.mp4 --proximity 70 --output data/outputs/report_p70.json
python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report_p70.json

# Extra: Visualize predictions on video (optional)
python visualize_predictions.py --video ../training_footages.mp4 --report data/outputs/report.json
# (Watch visualization.mp4 to verify tracklets are stable)
```

---

Done! You're now ready to start Phase 2. 🎉
