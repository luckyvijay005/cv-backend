"""
Extract and save face crops for tracked students.

Saves JPG images of students for visual verification.
"""
import cv2
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


def extract_student_crops(frames: List, track_history: Dict[int, List[Tuple]], output_dir: str = 'data/outputs/crops') -> Dict[int, str]:
    """
    Extract and save crop images for each student.

    Args:
        frames: List of video frames
        track_history: {track_id: [(x1, y1, x2, y2), ...]} bboxes per frame
        output_dir: Where to save crop images

    Returns:
        {track_id: 'path/to/crop.jpg', ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_paths = {}

    for track_id, bboxes in track_history.items():
        if len(bboxes) == 0:
            continue

        # Find best frame (middle of appearance, best quality)
        best_frame_idx = len(bboxes) // 2
        best_bbox = bboxes[best_frame_idx]
        best_frame = frames[best_frame_idx]

        # Extract crop with padding
        x1, y1, x2, y2 = best_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add padding (25% on each side)
        padding_x = int((x2 - x1) * 0.15)
        padding_y = int((y2 - y1) * 0.15)

        x1_padded = max(0, x1 - padding_x)
        y1_padded = max(0, y1 - padding_y)
        x2_padded = min(best_frame.shape[1], x2 + padding_x)
        y2_padded = min(best_frame.shape[0], y2 + padding_y)

        # Crop
        crop = best_frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if crop.size == 0:
            continue

        # Resize to consistent size (256x256)
        crop_resized = cv2.resize(crop, (256, 256))

        # Save
        crop_path = os.path.join(output_dir, f'student_{track_id:03d}.jpg')
        cv2.imwrite(crop_path, crop_resized)
        crop_paths[track_id] = crop_path

    return crop_paths


def extract_top_3_crops(frames: object, top_3_tracks: List[Tuple], track_history: Dict[int, List[Tuple]], track_start_frames: Dict[int, int],
                         output_dir: str = 'data/outputs/crops') -> Dict[int, str]:
    """
    Extract crops specifically for top-3 students.

    Args:
        frames: Video frames accessor
        top_3_tracks: [(track_id, metrics), ...] from InteractionAnalyzer
        track_history: {track_id: [(x1, y1, x2, y2), ...]}
        track_start_frames: {track_id: first_frame_idx}
        output_dir: Where to save images

    Returns:
        {track_id: 'path/to/crop.jpg', ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_paths = {}

    for track_id, metrics in top_3_tracks:
        if track_id not in track_history or len(track_history[track_id]) == 0:
            continue

        bboxes = track_history[track_id]

        # Find frame with clearest view (middle of sequence)
        idx_in_bboxes = len(bboxes) // 2
        best_frame_idx = track_start_frames.get(track_id, 0) + idx_in_bboxes
        
        if best_frame_idx >= len(frames):
            best_frame_idx = len(frames) - 1

        best_bbox = bboxes[idx_in_bboxes]
        best_frame = frames[best_frame_idx]

        # Extract crop with padding
        x1, y1, x2, y2 = best_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add 20% padding
        padding_x = int((x2 - x1) * 0.2)
        padding_y = int((y2 - y1) * 0.2)

        x1_padded = max(0, x1 - padding_x)
        y1_padded = max(0, y1 - padding_y)
        x2_padded = min(best_frame.shape[1], x2 + padding_x)
        y2_padded = min(best_frame.shape[0], y2 + padding_y)

        # Crop
        crop = best_frame[y1_padded:y2_padded, x1_padded:x2_padded]

        if crop.size == 0:
            continue

        # Resize to 256x256 for consistent display
        crop_resized = cv2.resize(crop, (256, 256))

        # Save with high quality
        crop_path = os.path.join(output_dir, f'top3_student_{track_id:03d}.jpg')
        cv2.imwrite(crop_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Use forward slashes for cross-platform compatibility
        crop_paths[track_id] = crop_path.replace(os.sep, '/')

    return crop_paths


def extract_all_crops(
    frames: object,
    student_ids: List[int],
    track_history: Dict[int, List[Tuple]],
    track_start_frames: Dict[int, int],
    cached_crops: Dict[int, np.ndarray] = None,
    output_dir: str = 'data/outputs/crops'
) -> Dict[int, str]:
    """
    Extract crops for all detected students using in-memory cached crops when available.
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_paths = {}

    for track_id in student_ids:
        crop_resized = None

        # 1. First priority: use in-memory cached crop captured during forward video pass
        if cached_crops and track_id in cached_crops and cached_crops[track_id] is not None:
            crop = cached_crops[track_id]
            if crop.size > 0:
                try:
                    crop_resized = cv2.resize(crop, (256, 256))
                except Exception:
                    crop_resized = None

        # 2. Fallback: seek video frame if not in memory cache
        if crop_resized is None:
            if track_id not in track_history or len(track_history[track_id]) == 0:
                continue
            bboxes = track_history[track_id]
            idx_in_bboxes = len(bboxes) // 2
            best_frame_idx = track_start_frames.get(track_id, 0) + idx_in_bboxes

            if best_frame_idx >= len(frames):
                best_frame_idx = len(frames) - 1

            best_bbox = bboxes[idx_in_bboxes]
            best_frame = frames[best_frame_idx]

            x1, y1, x2, y2 = [int(v) for v in best_bbox]
            padding_x = int((x2 - x1) * 0.2)
            padding_y = int((y2 - y1) * 0.2)

            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(best_frame.shape[1], x2 + padding_x)
            y2_padded = min(best_frame.shape[0], y2 + padding_y)

            crop = best_frame[y1_padded:y2_padded, x1_padded:x2_padded]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (256, 256))

        if crop_resized is not None:
            crop_path = os.path.join(output_dir, f'student_{track_id:03d}.jpg')
            cv2.imwrite(crop_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_paths[track_id] = crop_path.replace(os.sep, '/')

    return crop_paths
