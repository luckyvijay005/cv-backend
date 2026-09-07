"""
Utility functions: video I/O, visualization, JSON export.
"""
import cv2
import json
import os
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime


def load_video_frames(video_path: str, max_frames: int = None) -> Tuple[object, Dict]:
    """
    Load video frames using a memory-efficient accessor.
    """
    class VideoFrameAccessor:
        def __init__(self, path, max_f):
            self.path = path
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {path}")
            self.max_frames = max_f
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.max_frames:
                self.total = min(self.total, self.max_frames)
            
        def __getitem__(self, idx):
            if idx < 0 or idx >= self.total:
                raise IndexError("Frame index out of range")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            return frame if ret else np.zeros((10, 10, 3), dtype=np.uint8)
            
        def __len__(self):
            return self.total
            
        def get_sequential_frame(self):
            ret, frame = self.cap.read()
            return frame if ret else None

        def get_next_sampled_frame(self, step=1):
            """Read next frame and fast-skip step - 1 subsequent frames."""
            ret, frame = self.cap.read()
            if not ret:
                return None
            for _ in range(step - 1):
                if not self.cap.grab():
                    break
            return frame

        def release(self):
            self.cap.release()

    accessor = VideoFrameAccessor(video_path, max_frames)
    fps = accessor.cap.get(cv2.CAP_PROP_FPS)
    width = int(accessor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(accessor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    metadata = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': len(accessor),
        'duration_seconds': len(accessor) / fps if fps > 0 else 0,
    }
    
    return accessor, metadata


def draw_bboxes(frame: np.ndarray, detections: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
    """
    Draw bounding boxes on a frame.

    Args:
        frame: Input frame (BGR)
        detections: [(x1, y1, x2, y2, confidence), ...]

    Returns:
        Frame with drawn bboxes
    """
    frame_copy = frame.copy()
    for x1, y1, x2, y2, conf in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_copy, f'{conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame_copy


def draw_tracks(frame: np.ndarray, tracks: Dict[int, Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Draw tracklet IDs and bboxes on a frame.

    Args:
        frame: Input frame (BGR)
        tracks: {track_id: (x1, y1, x2, y2), ...}

    Returns:
        Frame with drawn tracks
    """
    frame_copy = frame.copy()
    colors = {}
    for track_id, (x1, y1, x2, y2) in tracks.items():
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Assign a consistent color per track_id
        if track_id not in colors:
            np.random.seed(track_id)
            colors[track_id] = tuple(np.random.randint(0, 256, 3).tolist())

        color = colors[track_id]
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_copy,
            f'ID:{track_id}',
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return frame_copy


def save_report(report: Dict, output_path: str):
    """
    Save report to JSON file.

    Args:
        report: Report dict
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {output_path}")


def load_report(report_path: str) -> Dict:
    """Load report from JSON file."""
    with open(report_path, 'r') as f:
        return json.load(f)


def save_labeled_annotations(annotations: Dict, output_path: str):
    """Save labeled annotations for validation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotations saved: {output_path}")


def extract_video_clip(video_path: str, start_frame: int, end_frame: int, output_path: str):
    """
    Extract a video clip between frames.

    Args:
        video_path: Source video path
        start_frame: Start frame index
        end_frame: End frame index
        output_path: Output video path
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video clip saved: {output_path}")
