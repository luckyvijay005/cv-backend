"""
Validation script: Compare predicted metrics vs. human-labeled ground truth.

Usage:
    python validate.py --labels data/labeled/interaction_labels.json --report data/outputs/report.json
"""
import argparse
import json
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_labels(label_file: str) -> Dict:
    """Load human-annotated labels."""
    with open(label_file, 'r') as f:
        return json.load(f)


def load_predictions(report_file: str) -> Dict:
    """Load predicted report."""
    with open(report_file, 'r') as f:
        return json.load(f)


def compute_metrics(predicted_scores: Dict[int, float], human_labels: Dict[str, str]) -> Dict:
    """
    Compare predicted interaction scores vs. human labels.

    Args:
        predicted_scores: {tracklet_id: interaction_score (0-1)}
        human_labels: {tracklet_id: "interacting" or "isolated"}

    Returns:
        Metrics dict with accuracy, precision, etc.
    """
    # Threshold: score < 0.4 → isolated, >= 0.4 → interacting
    isolation_threshold = 0.4

    predictions = []
    ground_truths = []
    matched_tracklets = []

    for tracklet_id, label in human_labels.items():
        # Convert tracklet_id to int if possible
        try:
            tid_int = int(tracklet_id)
        except ValueError:
            continue

        if tid_int not in predicted_scores:
            print(
                f"  WARNING: Tracklet {tracklet_id} not found in predictions")
            continue

        score = predicted_scores[tid_int]
        predicted_isolated = 1 if score < isolation_threshold else 0
        true_isolated = 1 if label == "isolated" else 0

        predictions.append(predicted_isolated)
        ground_truths.append(true_isolated)
        matched_tracklets.append((tracklet_id, score, label))

    if len(predictions) == 0:
        print("ERROR: No matching tracklets between predictions and labels!")
        return {}

    # Compute metrics
    accuracy = accuracy_score(ground_truths, predictions)
    cm = confusion_matrix(ground_truths, predictions)
    report = classification_report(
        ground_truths, predictions,
        target_names=["Interacting", "Isolated"],
        output_dict=True
    )

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'matched_tracklets': matched_tracklets,
        'threshold_used': isolation_threshold,
        'num_samples': len(predictions),
    }


def compute_tracking_metrics(frame_labels: Dict, predicted_bboxes: Dict) -> Dict:
    """
    Compute tracking metrics (Precision, Recall, IDF1) if available.

    This is simplified; full MOT metrics require DetectionAPI.
    """
    # Placeholder for now
    return {
        'note': 'Full tracking metrics (IDF1, MOTA) require labeled tracklets with frame-level bboxes.',
        'recommendation': 'Use CVAT or similar tool for frame-level bbox annotations.',
    }


def print_report(metrics: Dict, output_file: str = None):
    """Pretty-print validation report."""
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)

    if 'accuracy' in metrics:
        print(f"\nInteraction Detection Accuracy: {metrics['accuracy']:.2%}")
        print(f"Samples Matched: {metrics['num_samples']}")
        print(
            f"Isolation Threshold: {metrics['threshold_used']} (score < threshold → isolated)")

        cm = np.array(metrics['confusion_matrix'])
        print("\nConfusion Matrix:")
        print(f"  (rows: ground truth, cols: predicted)")
        print(f"  Interacting (0) vs Isolated (1):")
        print(
            f"    True Negatives:  {cm[0, 0]:3d}  |  False Positives: {cm[0, 1]:3d}")
        print(
            f"    False Negatives: {cm[1, 0]:3d}  |  True Positives:  {cm[1, 1]:3d}")

        # Compute precision and recall
        cr = metrics['classification_report']
        print("\nPer-Class Metrics:")
        for label in ['Interacting', 'Isolated']:
            if label in cr:
                print(f"  {label}:")
                print(f"    Precision: {cr[label]['precision']:.2%}")
                print(f"    Recall:    {cr[label]['recall']:.2%}")
                print(f"    F1-Score:  {cr[label]['f1-score']:.2%}")

        # Print sample predictions vs ground truth
        print("\nSample Predictions vs Ground Truth:")
        print(
            f"{'Tracklet':<15} {'Predicted Score':<18} {'Human Label':<15} {'Match':<10}")
        print("-" * 60)
        for i, (tracklet_id, score, label) in enumerate(metrics['matched_tracklets'][:10]):
            predicted_label = "Isolated" if score < metrics['threshold_used'] else "Interacting"
            match = "✓" if predicted_label.lower() == label else "✗"
            print(f"{tracklet_id:<15} {score:<18.2f} {label:<15} {match:<10}")

        if len(metrics['matched_tracklets']) > 10:
            print(f"... and {len(metrics['matched_tracklets']) - 10} more")

    print("\n" + "="*70)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Report saved: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validation: Compare predictions vs. human labels")
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to human-labeled JSON')
    parser.add_argument('--report', type=str, required=True,
                        help='Path to predicted report JSON')
    parser.add_argument('--output', type=str, default='data/outputs/validation_metrics.json',
                        help='Path to save validation report')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Isolation score threshold (score < threshold → isolated)')

    args = parser.parse_args()

    try:
        print("Loading labels and predictions...")
        labels = load_labels(args.labels)
        report = load_predictions(args.report)

        # Extract interaction labels
        if 'minute_labels' in labels:
            print(f"Found {len(labels['minute_labels'])} minute-level labels")

            # Collect all human labels
            human_labels = {}
            for minute_data in labels['minute_labels']:
                for person in minute_data.get('people', []):
                    tracklet_id = person['tracklet_id']
                    label = person['label']
                    # Later entries override earlier
                    human_labels[tracklet_id] = label

            print(f"Collected {len(human_labels)} unique person-interactions")

            # Extract predicted scores
            predicted_scores = {}
            if 'all_metrics' in report:
                for track_id_str, metrics in report['all_metrics'].items():
                    try:
                        track_id = int(track_id_str)
                        predicted_scores[track_id] = metrics['score']
                    except (ValueError, KeyError):
                        pass

            print(f"Found {len(predicted_scores)} predictions")

            # Match and compute metrics
            metrics = compute_metrics(predicted_scores, human_labels)
            print_report(metrics, args.output)

        else:
            print("ERROR: No 'minute_labels' found in label file")
            print("Please ensure you've annotated data in the correct format")
            print(
                f"Expected: data/labeled/interaction_labels.json with 'minute_labels' key")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
