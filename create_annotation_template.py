"""
Helper script: Create empty annotation template for labeling.

Usage:
    python create_annotation_template.py --video ../training_footages.mp4 --sample_minutes 30 --sample_rate 30
"""
import argparse
import json
import cv2
from typing import List, Dict
from utils import load_video_frames


def create_frame_annotation_template(video_path: str, sample_rate: int = 30) -> Dict:
    """
    Create skeleton for frame-level annotations.

    Args:
        video_path: Path to video
        sample_rate: Sample every Nth frame (30 = 1 per second at 30 FPS)

    Returns:
        Template dict for frame annotations
    """
    frames, metadata = load_video_frames(video_path, max_frames=None)
    total_frames = len(frames)
    fps = metadata['fps']

    frames_to_label = []
    for frame_id in range(0, total_frames, sample_rate):
        timestamp_seconds = frame_id / fps
        frames_to_label.append({
            'frame_id': frame_id,
            'timestamp_seconds': round(timestamp_seconds, 2),
            'people': []  # Fill in manually: person_id, tracklet_id, bbox, occlusion_level
        })

    return {
        'metadata': {
            'video_file': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'sampled_frames': len(frames_to_label),
            'sample_rate': sample_rate,
        },
        'frames': frames_to_label
    }


def create_window_annotation_template(video_path: str, fps: float = 30.0) -> Dict:
    """
    Create skeleton for minute-level window classifications.

    Args:
        video_path: Path to video
        fps: Video frame rate

    Returns:
        Template dict for window annotations
    """
    frames, metadata = load_video_frames(video_path, max_frames=None)
    total_frames = len(frames)
    fps = metadata['fps']
    duration_seconds = total_frames / fps

    minutes = int(duration_seconds / 60) + 1
    windows = []

    for minute_id in range(minutes):
        frame_start = minute_id * int(fps * 60)
        frame_end = min((minute_id + 1) * int(fps * 60), total_frames)
        timestamp_start = minute_id * 60
        timestamp_end = (minute_id + 1) * 60

        windows.append({
            'window_id': minute_id,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'timestamp_start_seconds': timestamp_start,
            'timestamp_end_seconds': timestamp_end,
            # Fill in: break, structured_activity, free_play, etc.
            'window_type': 'unknown',
            'notes': ''
        })

    return {
        'metadata': {
            'video_file': video_path,
            'total_duration_seconds': duration_seconds,
            'total_windows': len(windows),
        },
        'windows': windows
    }


def create_interaction_annotation_template(sample_minutes: int = 5) -> Dict:
    """
    Create skeleton for interaction labels (must be filled with actual data).

    Args:
        sample_minutes: Number of minutes to label

    Returns:
        Template dict for interaction labels
    """
    minute_labels = []

    for minute_id in range(sample_minutes):
        fps = 30  # Assumption
        minute_labels.append({
            'minute_id': minute_id,
            'frame_start': minute_id * fps * 60,
            'frame_end': (minute_id + 1) * fps * 60,
            'timestamp_seconds': minute_id * 60,
            'people': [
                # Fill in with actual observations:
                # {
                #   'tracklet_id': 'GT-1',
                #   'label': 'interacting' or 'isolated',
                #   'reason': 'Description',
                #   'proximity_peers': 2
                # }
            ]
        })

    return {
        'metadata': {
            'sample_minutes': sample_minutes,
            'notes': 'Fill in "people" array per minute with tracklet IDs and labels'
        },
        'minute_labels': minute_labels
    }


def main():
    parser = argparse.ArgumentParser(description="Create annotation templates")
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--sample_minutes', type=int, default=5,
                        help='Minutes to label (for interaction labels)')
    parser.add_argument('--sample_rate', type=int,
                        default=30, help='Frame sampling rate')
    parser.add_argument('--output_dir', type=str,
                        default='data/labeled', help='Output directory')

    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Creating annotation templates for: {args.video}")

        # Frame annotations
        print("Creating frame annotation template...")
        frame_template = create_frame_annotation_template(
            args.video, sample_rate=args.sample_rate)
        frame_file = os.path.join(args.output_dir, 'frame_annotations.json')
        with open(frame_file, 'w') as f:
            json.dump(frame_template, f, indent=2)
        print(f"  Saved: {frame_file}")
        print(f"  Frames to label: {len(frame_template['frames'])}")

        # Window annotations
        print("Creating window annotation template...")
        window_template = create_window_annotation_template(args.video)
        window_file = os.path.join(args.output_dir, 'window_annotations.json')
        with open(window_file, 'w') as f:
            json.dump(window_template, f, indent=2)
        print(f"  Saved: {window_file}")
        print(f"  Windows to classify: {len(window_template['windows'])}")

        # Interaction annotations
        print(
            f"Creating interaction label template ({args.sample_minutes} minutes)...")
        interaction_template = create_interaction_annotation_template(
            args.sample_minutes)
        interaction_file = os.path.join(
            args.output_dir, 'interaction_labels.json')
        with open(interaction_file, 'w') as f:
            json.dump(interaction_template, f, indent=2)
        print(f"  Saved: {interaction_file}")

        print("\nTemplates created! Next steps:")
        print(
            f"1. Run pipeline: python run.py --video {args.video} --output data/outputs/report.json")
        print(f"2. View predictions and fill in: {interaction_file}")
        print(
            f"3. Run validation: python validate.py --labels {interaction_file} --report data/outputs/report.json")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
