"""
Interaction analysis: proximity-based scoring of social interaction.

Computes per-minute metrics, interaction score, and temporal engagement timelines for each tracklet.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict


class InteractionAnalyzer:
    """
    Analyzes interaction patterns based on proximity and proximity changes over time.
    """

    def __init__(self, proximity_threshold_px: float = 80.0, fps: float = 30.0):
        """
        Args:
            proximity_threshold_px: Pixel distance threshold for "near peer" (≈1.5m in typical classroom)
            fps: Video frame rate / effective sampling rate (for time window computation)
        """
        self.proximity_threshold_px = proximity_threshold_px
        self.fps = max(1.0, float(fps))
        self.seconds_per_window = 60.0
        self.frames_per_window = self.fps * self.seconds_per_window

    def compute_metrics(
        self,
        student_track_history: Dict[int, List[Tuple[float, float, float, float]]],
        all_frames_tracks: List[Dict[int, Tuple[float, float, float, float]]],
        total_frames: int,
        frame_timestamps: Optional[List[float]] = None
    ) -> Dict[int, Dict]:
        """
        Compute interaction metrics and state timeline for each tracklet.

        Args:
            student_track_history: {track_id: [(x1, y1, x2, y2), ...]}
            all_frames_tracks: list of dicts {track_id: bbox} per processed frame
            total_frames: Total number of frames in video
            frame_timestamps: Optional list of timestamps (seconds) for each entry in all_frames_tracks

        Returns:
            {track_id: {'isolated_min': float, 'near_peer_min': float, 'score': float, 'confidence': float, 'timeline': list, ...}}
        """
        metrics = {}
        total_minutes = total_frames / self.fps / 60.0

        # Initialize counters
        isolated_frames = {tid: 0 for tid in student_track_history.keys()}
        near_peer_frames = {tid: 0 for tid in student_track_history.keys()}
        approach_events = {tid: 0 for tid in student_track_history.keys()}
        prev_near = {tid: False for tid in student_track_history.keys()}
        
        # Timeline recording per student: list of (timestamp, state)
        # state: 0 for absent/untracked, 1 for isolated, 2 for near_peer
        student_state_series = {tid: [] for tid in student_track_history.keys()}
        first_seen_frame = {}
        last_seen_frame = {}

        num_processed = len(all_frames_tracks)
        thresh_sq = float(self.proximity_threshold_px ** 2)

        for frame_idx, frame_tracks in enumerate(all_frames_tracks):
            if not frame_tracks:
                continue

            current_time = frame_timestamps[frame_idx] if (frame_timestamps and frame_idx < len(frame_timestamps)) else (frame_idx / self.fps)

            # Filter to active students in this frame
            active_students = {tid: bbox for tid, bbox in frame_tracks.items() if tid in student_track_history}
            active_tids = list(active_students.keys())
            if not active_tids:
                continue

            n_active = len(active_tids)
            if n_active == 1:
                tid = active_tids[0]
                if tid not in first_seen_frame:
                    first_seen_frame[tid] = current_time
                last_seen_frame[tid] = current_time
                isolated_frames[tid] += 1
                student_state_series[tid].append((current_time, 'isolated'))
                if prev_near[tid]:
                    approach_events[tid] += 1
                prev_near[tid] = False
                continue

            # Vectorized centroid calculation & pairwise distance matrix
            boxes = [active_students[tid] for tid in active_tids]
            centroids = np.array([
                [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5] for b in boxes
            ], dtype=np.float32)

            diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(dist_sq, np.inf)
            near_any = np.any(dist_sq < thresh_sq, axis=1)

            for i, tid in enumerate(active_tids):
                if tid not in first_seen_frame:
                    first_seen_frame[tid] = current_time
                last_seen_frame[tid] = current_time

                if near_any[i]:
                    near_peer_frames[tid] += 1
                    student_state_series[tid].append((current_time, 'near_peer'))
                    prev_near[tid] = True
                else:
                    isolated_frames[tid] += 1
                    student_state_series[tid].append((current_time, 'isolated'))
                    if prev_near[tid]:
                        approach_events[tid] += 1  # Detected leaving a group
                    prev_near[tid] = False

        # Finalize metrics and condense timelines
        for tid in student_track_history.keys():
            iso_min = isolated_frames[tid] / self.fps / 60.0
            near_min = near_peer_frames[tid] / self.fps / 60.0
            vis_min = len(student_track_history[tid]) / self.fps / 60.0

            # Compute confidence: higher if tracklet was consistently visible
            visibility_fraction = vis_min / total_minutes if total_minutes > 0 else 0
            confidence = min(1.0, visibility_fraction * 1.5)  # Scale to [0,1]

            # Interaction score
            if total_minutes > 0:
                score = (near_min + 0.3 * approach_events[tid]) / total_minutes
            else:
                score = 0
            score = float(np.clip(score, 0, 1))

            # Build condensed timeline segments
            timeline_segments = self._build_timeline_segments(student_state_series[tid])
            
            # Find peak isolation timestamp
            peak_isolated_ts = self._find_peak_isolated_timestamp(timeline_segments, first_seen_frame.get(tid, 0.0))

            metrics[tid] = {
                'total_minutes': float(total_minutes),
                'isolated_minutes': float(iso_min),
                'near_peer_minutes': float(near_min),
                'approach_events': int(approach_events[tid]),
                'visibility_minutes': float(vis_min),
                'score': score,
                'confidence': float(confidence),
                'first_seen_sec': float(first_seen_frame.get(tid, 0.0)),
                'last_seen_sec': float(last_seen_frame.get(tid, 0.0)),
                'peak_isolated_sec': float(peak_isolated_ts),
                'timeline': timeline_segments
            }

        return metrics

    def _build_timeline_segments(self, state_series: List[Tuple[float, str]]) -> List[Dict]:
        """Condense frame-by-frame states into discrete interval segments."""
        if not state_series:
            return []

        segments = []
        cur_state = state_series[0][1]
        start_ts = state_series[0][0]
        prev_ts = start_ts

        for ts, st in state_series[1:]:
            # If gap between observations is > 2 seconds or state changed, flush segment
            if st != cur_state or (ts - prev_ts) > 2.5:
                segments.append({
                    'state': cur_state,
                    'start_sec': round(float(start_ts), 2),
                    'end_sec': round(float(prev_ts), 2),
                    'duration_sec': round(float(max(0.1, prev_ts - start_ts)), 2)
                })
                cur_state = st
                start_ts = ts
            prev_ts = ts

        if prev_ts >= start_ts:
            segments.append({
                'state': cur_state,
                'start_sec': round(float(start_ts), 2),
                'end_sec': round(float(prev_ts), 2),
                'duration_sec': round(float(max(0.1, prev_ts - start_ts)), 2)
            })

        return segments

    def _find_peak_isolated_timestamp(self, timeline_segments: List[Dict], default_ts: float) -> float:
        """Find the center timestamp of the longest continuous isolated period."""
        longest_duration = 0.0
        peak_ts = default_ts

        for seg in timeline_segments:
            if seg['state'] == 'isolated' and seg['duration_sec'] > longest_duration:
                longest_duration = seg['duration_sec']
                peak_ts = (seg['start_sec'] + seg['end_sec']) / 2.0

        return round(float(peak_ts), 2)

    @staticmethod
    def _bbox_distance(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
        """
        Compute Euclidean distance between bbox centroids.

        Args:
            bbox1, bbox2: (x1, y1, x2, y2)

        Returns:
            Distance in pixels
        """
        x1_c = (bbox1[0] + bbox1[2]) / 2
        y1_c = (bbox1[1] + bbox1[3]) / 2
        x2_c = (bbox2[0] + bbox2[2]) / 2
        y2_c = (bbox2[1] + bbox2[3]) / 2
        return float(np.sqrt((x1_c - x2_c)**2 + (y1_c - y2_c)**2))

    @staticmethod
    def get_top_n_least_interactive(metrics: Dict[int, Dict], n: int = 3) -> List[Tuple[int, Dict]]:
        """
        Get top N least interactive tracklets (lowest scores).

        Args:
            metrics: Metrics dict from compute_metrics()
            n: Number of results to return

        Returns:
            [(track_id, metrics_dict), ...] sorted by score (ascending)
        """
        sorted_tracks = sorted(metrics.items(), key=lambda x: x[1]['score'])
        return sorted_tracks[:n]
