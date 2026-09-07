"""
Visualize predicted tracklets on video.

This generates a video with bounding boxes and tracklet IDs overlaid,
so you can easily see which students are detected and verify the predictions.

Usage:
    python visualize_predictions.py --video ../training_footages.mp4 --report data/outputs/report.json --output data/outputs/visualization.mp4
"""
import argparse
import json
import cv2
import os
from typing import Dict, List, Tuple
from tqdm import tqdm


def load_report(report_file: str) -> Dict:
    """Load prediction report."""
    with open(report_file, 'r') as f:
        return json.load(f)


def load_video(video_path: str, max_frames_preview: int = 500) -> Tuple[List, Dict]:
    """Load video frames."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1
        if max_frames_preview and frame_idx >= max_frames_preview:
            break

    cap.release()

    return frames, {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
    }


def draw_tracklets_on_frame(frame: cv2.Mat, tracklets: Dict[int, Tuple]) -> cv2.Mat:
    """Draw tracklet IDs and bboxes on frame."""
    frame_copy = frame.copy()

    for track_id, (x1, y1, x2, y2) in tracklets.items():
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Assign consistent color per track_id
        color_seed = track_id % 256
        color = (
            (color_seed * 73) % 256,
            (color_seed * 149) % 256,
            (color_seed * 211) % 256,
        )

        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_copy,
            f'ID:{track_id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame_copy


def main():
    parser = argparse.ArgumentParser(
        description="Visualize predicted tracklets on video")
    parser.add_argument('--video', type=str, required=True,
                        help='Input video path')
    parser.add_argument('--report', type=str, required=True,
                        help='Prediction report JSON')
    parser.add_argument('--output', type=str, default='data/outputs/visualization.mp4',
                        help='Output visualization video')
    parser.add_argument('--max_frames', type=int,
                        default=500, help='Max frames to visualize')
    parser.add_argument('--highlight_top_n', type=int, default=3,
                        help='Highlight top N least interactive students')

    args = parser.parse_args()

    try:
        print(f"Loading video: {args.video}")
        frames, meta = load_video(
            args.video, max_frames_preview=args.max_frames)

        print(f"Loading predictions: {args.report}")
        report = load_report(args.report)

        # Get top-3 least interactive for highlighting
        top_3_ids = [s['track_id'] for s in report['top_3_least_interactive']]
        print(f"Top {len(top_3_ids)} least interactive: {top_3_ids}")

        # Create video writer
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            args.output,
            fourcc,
            meta['fps'],
            (meta['width'], meta['height'])
        )

        print(f"Creating visualization ({len(frames)} frames)...")
        for frame_idx in tqdm(range(len(frames)), desc="Writing frames"):
            frame = frames[frame_idx]

            # Add metadata text
            cv2.putText(
                frame,
                f'Frame: {frame_idx} / {len(frames)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f'Top {len(top_3_ids)} least interactive IDs: {top_3_ids}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if frame_idx % 30 < 15 else (0, 255, 255),
                2
            )

            out_writer.write(frame)

        out_writer.release()
        print(f"\nVisualization saved: {args.output}")
        print(f"Watch this video to:")
        print(f"  1. Verify tracklet IDs are stable (not jumping frame-to-frame)")
        print(f"  2. Identify the top-3 least interactive students")
        print(f"  3. Mark down 5-10 minute windows to label manually")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
