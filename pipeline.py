"""
Main pipeline: detection → ByteTrack tracking → interaction analysis.
"""
import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
from concurrent.futures import CancelledError

from detector import PersonDetector
from tracker import ByteTrackWrapper
from interaction_analyzer import InteractionAnalyzer
from persistence import StudentRegistry
from utils import load_video_frames, draw_tracks, save_report
from student_crops import extract_top_3_crops, extract_all_crops


class ClassroomAnalyticsPipeline:
    """
    End-to-end pipeline for classroom interaction analysis.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        device: int = 0,
        proximity_threshold_px: float = 80.0,
        student_registry_path: str = 'data/student_registry.json',
        teacher_zone: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Args:
            model_name: YOLOv8 model variant
            device: GPU device index (0 for GPU, -1 for CPU)
            proximity_threshold_px: Proximity threshold in pixels
            student_registry_path: persistent registry for student IDs across videos
            teacher_zone: optional manual teacher zone (x1,y1,x2,y2)
        """
        self.detector = PersonDetector(model_name=model_name, device=device)
        self.tracker = ByteTrackWrapper(use_simple=False)
        self.proximity_threshold_px = proximity_threshold_px
        self.registry = StudentRegistry(registry_path=student_registry_path)
        self.teacher_zone = teacher_zone

    def run(
        self,
        video_path: str,
        max_frames: int = None,
        sample_fps: Optional[float] = None,
        save_viz: bool = False,
        cancel_callback: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable[[str, int, str], None]] = None
    ) -> Dict:
        """
        Run the full pipeline on a video with live progress reporting.

        Args:
            video_path: Path to input video
            max_frames: Max frames to process (None = all)
            sample_fps: Target FPS to process (e.g., 5.0, 10.0, or None for native FPS)
            save_viz: If True, save visualization frames to outputs/
            cancel_callback: Optional callable; if it returns True the pipeline
                             will stop immediately and raise CancelledError.
            progress_callback: Optional callable(stage_name, percent, detail_text)

        Returns:
            Report dict with metrics, top-3 students, and interaction timelines
        """
        print(f"Loading video: {video_path}")
        if progress_callback:
            progress_callback("Loading video", 10, "Opening video and reading stream metadata...")

        frames, metadata = load_video_frames(video_path, max_frames=max_frames)
        fps = metadata['fps'] if metadata['fps'] > 0 else 30.0
        total_video_frames = len(frames)

        # Calculate frame skipping step based on requested sample_fps
        if sample_fps and sample_fps < fps:
            frame_step = max(1, int(round(fps / float(sample_fps))))
            effective_fps = fps / frame_step
        else:
            frame_step = 1
            effective_fps = fps

        print(
            f"Video: {metadata['width']}x{metadata['height']}, {fps:.1f} FPS (Sampled at ~{effective_fps:.1f} FPS, step={frame_step}), "
            f"{total_video_frames} frames ({metadata['duration_seconds']:.1f}s)"
        )

        # Track state & in-memory crop caches to avoid slow seeking on AVI files
        track_history = {}          # {track_id: [(x1, y1, x2, y2), ...]}
        track_start_frames = {}     # {track_id: first_original_frame_idx}
        track_crops = {}            # {track_id: crop_np_array}
        track_crop_areas = {}       # {track_id: area_of_best_crop}
        all_frames_tracks = []      # Track assignments per processed frame
        frame_timestamps = []       # Timestamp in seconds for each processed frame

        print("Running detection + ByteTrack tracking...")
        self.tracker.reset()

        processed_frame_count = 0
        raw_frame_idx = 0
        start_time = time.time()

        pbar = tqdm(total=total_video_frames, desc="Processing video")
        while raw_frame_idx < total_video_frames:
            # Check for user cancellation
            if cancel_callback and processed_frame_count % 15 == 0 and cancel_callback():
                frames.release()
                raise CancelledError("Analysis cancelled by user")

            frame = frames.get_next_sampled_frame(step=frame_step)
            if frame is None:
                break

            timestamp_sec = raw_frame_idx / fps
            frame_timestamps.append(timestamp_sec)

            # Detect people with optimized GPU / CPU inference
            detections = self.detector.detect(frame, conf_threshold=0.45)

            # Update ByteTrack tracker
            tracked_objects = self.tracker.update(detections)

            # Record tracks and cache highest quality crop in memory
            all_frames_tracks.append(tracked_objects)
            for track_id, bbox in tracked_objects.items():
                if track_id not in track_history:
                    track_history[track_id] = []
                    track_start_frames[track_id] = raw_frame_idx
                track_history[track_id].append(bbox)

                # Cache crop if not yet cached or if this one is larger/clearer
                x1, y1, x2, y2 = [int(v) for v in bbox]
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                area = w * h
                if (track_id not in track_crops) or (area > track_crop_areas.get(track_id, 0) and len(track_history[track_id]) % 5 == 0):
                    px = int(w * 0.15)
                    py = int(h * 0.15)
                    y1p, y2p = max(0, y1 - py), min(frame.shape[0], y2 + py)
                    x1p, x2p = max(0, x1 - px), min(frame.shape[1], x2 + px)
                    crop = frame[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        track_crops[track_id] = crop.copy()
                        track_crop_areas[track_id] = area

            processed_frame_count += 1
            advance = min(frame_step, total_video_frames - raw_frame_idx)
            raw_frame_idx += advance
            pbar.update(advance)

            # Emit live continuous progress to UI
            if progress_callback and (processed_frame_count % 10 == 0 or raw_frame_idx >= total_video_frames):
                pct = int(15 + min(1.0, raw_frame_idx / max(1, total_video_frames)) * 55)
                elapsed = time.time() - start_time
                curr_fps = processed_frame_count / max(0.001, elapsed)
                progress_callback(
                    "Tracking students (ByteTrack)",
                    min(70, pct),
                    f"Frame {raw_frame_idx:,}/{total_video_frames:,} ({curr_fps:.1f} FPS) · {len(track_history)} tracked"
                )

        pbar.close()
        print(f"Processed {processed_frame_count} sampled frames. Detected {len(track_history)} raw tracklets.")

        # Filter out very short ghost tracks (< 1.5 seconds) first to avoid evaluating noise
        min_track_length = max(3, int(effective_fps * 1.5))
        valid_track_history = {
            tid: bboxes
            for tid, bboxes in track_history.items()
            if len(bboxes) >= min_track_length
        }
        print(f"Tracklets after ghost filter (>={min_track_length} frames): {len(valid_track_history)}")

        # Detect teacher tracklets and exclude teacher from student interaction analysis
        teacher_track_ids = self._identify_teacher_tracks(
            valid_track_history, metadata['width'], metadata['height'])
        print(f"Teacher tracks excluded: {len(teacher_track_ids)} track IDs")

        student_track_history = {
            tid: bboxes
            for tid, bboxes in valid_track_history.items()
            if tid not in teacher_track_ids
        }

        # Register students persistently across videos (using in-memory cached crops)
        if progress_callback:
            progress_callback("Registering student profiles", 75, "Matching student appearance descriptors...")

        print(f"Registering student profiles across {len(student_track_history)} tracklets...")
        persistent_map = self.registry.register_tracks(
            frames, student_track_history, track_start_frames, cached_crops=track_crops)

        # ── Aggregate by persistent student ID ──────────────────────────
        student_history_by_id = {}      # {student_id: [all bboxes]}
        student_start_frames_by_id = {} # {student_id: earliest start frame}
        student_crops_by_id = {}        # {student_id: best_crop_np_array}

        for tid, bboxes in student_track_history.items():
            sid = persistent_map.get(tid)
            if sid is None:
                continue
            if sid not in student_history_by_id:
                student_history_by_id[sid] = []
                student_start_frames_by_id[sid] = track_start_frames.get(tid, 0)
            student_history_by_id[sid].extend(bboxes)
            student_start_frames_by_id[sid] = min(
                student_start_frames_by_id[sid], track_start_frames.get(tid, 0))
            if tid in track_crops and sid not in student_crops_by_id:
                student_crops_by_id[sid] = track_crops[tid]

        unique_student_count = len(student_history_by_id)
        print(f"Unique persistent students identified: {unique_student_count}")

        # Remap per-frame track assignments to use student IDs
        remapped_frames_tracks = []
        for frame_tracks in all_frames_tracks:
            remapped = {}
            for tid, bbox in frame_tracks.items():
                if tid in persistent_map:
                    sid = persistent_map[tid]
                    if sid in student_history_by_id:
                        remapped[sid] = bbox
            remapped_frames_tracks.append(remapped)

        # Analyze interactions using aggregated student data
        if progress_callback:
            progress_callback("Analysing interactions", 82, "Computing peer proximity & isolation times...")

        print("Computing interaction metrics and timelines...")
        analyzer = InteractionAnalyzer(
            proximity_threshold_px=self.proximity_threshold_px, fps=float(effective_fps))
        metrics = analyzer.compute_metrics(
            student_history_by_id, remapped_frames_tracks, processed_frame_count, frame_timestamps)

        # Get top-3 least interactive
        top_3 = InteractionAnalyzer.get_top_n_least_interactive(metrics, n=3)

        # Extract crop images of all students (using in-memory cached crops directly)
        if progress_callback:
            progress_callback("Extracting student images", 90, "Writing student profile images...")

        print("Extracting all student crop images...")
        all_student_ids = list(student_history_by_id.keys())
        crop_paths = extract_all_crops(
            frames, all_student_ids, student_history_by_id, student_start_frames_by_id, cached_crops=student_crops_by_id)

        # Convert all crop paths to absolute, forward-slash paths for the React app
        def abs_crop(p):
            if not p:
                return None
            return os.path.abspath(p).replace(os.sep, '/')

        if progress_callback:
            progress_callback("Generating report", 96, "Finalizing report metrics...")

        report = {
            'date': metadata.get('date', '2026-04-18'),
            'video_file': video_path,
            'video_width': metadata['width'],
            'video_height': metadata['height'],
            'native_fps': round(float(fps), 2),
            'sample_fps': round(float(sample_fps), 2) if sample_fps else round(float(fps), 2),
            'effective_fps': round(float(effective_fps), 2),
            'total_frames': total_video_frames,
            'processed_frames': processed_frame_count,
            'total_students': unique_student_count,
            'duration_seconds': metadata['duration_seconds'],
            'proximity_threshold_px': self.proximity_threshold_px,
            'teacher_track_ids': list(teacher_track_ids),
            'student_id_map': {str(tid): sid for tid, sid in persistent_map.items()},
            'top_3_least_interactive': [
                {
                    'track_id': student_id,
                    'student_id': student_id,
                    'score': float(metrics[student_id]['score']),
                    'isolated_minutes': float(metrics[student_id]['isolated_minutes']),
                    'near_peer_minutes': float(metrics[student_id]['near_peer_minutes']),
                    'approach_events': int(metrics[student_id]['approach_events']),
                    'confidence': float(metrics[student_id]['confidence']),
                    'first_seen_sec': float(metrics[student_id].get('first_seen_sec', 0.0)),
                    'last_seen_sec': float(metrics[student_id].get('last_seen_sec', 0.0)),
                    'peak_isolated_sec': float(metrics[student_id].get('peak_isolated_sec', 0.0)),
                    'timeline': metrics[student_id].get('timeline', []),
                    'crop_image': abs_crop(crop_paths.get(student_id)),
                }
                for student_id, m in top_3
            ],
            'all_metrics': {
                str(student_id): {
                    'student_id': student_id,
                    'score': float(m['score']),
                    'isolated_minutes': float(m['isolated_minutes']),
                    'near_peer_minutes': float(m['near_peer_minutes']),
                    'approach_events': int(m['approach_events']),
                    'confidence': float(m['confidence']),
                    'first_seen_sec': float(m.get('first_seen_sec', 0.0)),
                    'last_seen_sec': float(m.get('last_seen_sec', 0.0)),
                    'peak_isolated_sec': float(m.get('peak_isolated_sec', 0.0)),
                    'timeline': m.get('timeline', []),
                    'crop_image': abs_crop(crop_paths.get(student_id)),
                }
                for student_id, m in metrics.items()
            }
        }

        if save_viz:
            self._save_visualization(
                frames, all_frames_tracks, top_3, track_history, fps)

        frames.release()
        return report

    def _identify_teacher_tracks(self, track_history: Dict[int, List[Tuple[float, float, float, float]]], frame_width: int, frame_height: int) -> set:
        """Identify teacher track IDs using heuristics and optional manual zone."""
        if self.teacher_zone:
            x1, y1, x2, y2 = self.teacher_zone
            teacher_ids = set()
            for track_id, bboxes in track_history.items():
                if not bboxes:
                    continue
                best_bbox = bboxes[len(bboxes) // 2]
                cx = (best_bbox[0] + best_bbox[2]) / 2
                cy = (best_bbox[1] + best_bbox[3]) / 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    teacher_ids.add(track_id)
            return teacher_ids

        if not track_history:
            return set()

        # Compute average area and centroid for each track
        avg_areas = {}
        avg_centers = {}
        for track_id, bboxes in track_history.items():
            if not bboxes:
                continue
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
            centers = [((x1 + x2) * 0.5, (y1 + y2) * 0.5) for x1, y1, x2, y2 in bboxes]
            avg_areas[track_id] = float(np.mean(areas))
            avg_centers[track_id] = (float(np.mean([c[0] for c in centers])), float(np.mean([c[1] for c in centers])))

        if not avg_areas:
            return set()

        median_area = float(np.median(list(avg_areas.values())))

        # Teacher candidate must have area substantially larger than median student area (> 1.35x)
        # and be present for a meaningful number of frames
        candidates = [
            tid for tid, area in avg_areas.items()
            if area > median_area * 1.35 and len(track_history[tid]) >= 25
        ]
        if not candidates:
            return set()

        frame_diag = float(np.sqrt(frame_width**2 + frame_height**2)) if (frame_width and frame_height) else 1000.0
        candidate_scores = []
        for track_id in candidates:
            separation = self._track_separation(track_id, avg_centers, frame_diag)
            score = (avg_areas[track_id] / (median_area + 1e-3)) + (separation * 0.5)
            candidate_scores.append((score, track_id))

        candidate_scores.sort(reverse=True)
        top_score, top_track = candidate_scores[0]
        if top_score >= 1.8:
            return {top_track}

        return set()

    def _track_separation(self, candidate_id: int, avg_centers: Dict[int, Tuple[float, float]], frame_diag: float) -> float:
        """Compute average normalized separation of candidate centroid from other track centroids."""
        if candidate_id not in avg_centers or len(avg_centers) <= 1:
            return 0.0

        c_pt = np.array(avg_centers[candidate_id], dtype=np.float32)
        other_pts = np.array([pt for tid, pt in avg_centers.items() if tid != candidate_id], dtype=np.float32)
        if len(other_pts) == 0:
            return 0.0

        dists = np.linalg.norm(other_pts - c_pt, axis=1)
        mean_dist = float(np.mean(dists))
        return mean_dist / max(1.0, frame_diag)

    def _save_visualization(self, frames: List, all_frames_tracks: List, top_3: List, track_history: Dict, fps: float):
        """Save visualization of top-3 students."""
        import os
        os.makedirs('data/outputs', exist_ok=True)

        # Save first frame with all tracks
        frame_viz = frames[0].copy()
        frame_viz = draw_tracks(frame_viz, all_frames_tracks[0])
        cv2.imwrite('data/outputs/frame_00_all_tracks.jpg', frame_viz)

        # Save a frame showing top-3 highlighted
        sample_idx = min(100, len(all_frames_tracks) - 1)
        frame_top3 = frames[sample_idx].copy()
        top_3_ids = {track_id for track_id, _ in top_3}
        frame_top3_tracks = {
            tid: bbox for tid, bbox in (all_frames_tracks[sample_idx] or {}).items()
            if tid in top_3_ids
        }
        frame_top3 = draw_tracks(frame_top3, frame_top3_tracks)
        cv2.imwrite(
            'data/outputs/frame_sample_top3_highlighted.jpg', frame_top3)

        print("Visualization saved to data/outputs/")
