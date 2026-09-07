"""
Multi-object tracking using ByteTrack with Kalman Filter & IoU association.

Maintains stable tracklet IDs across frames for each detected person and handles occlusions.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of bounding boxes.
    
    Args:
        bboxes1: shape (N, 4) in (x1, y1, x2, y2) format
        bboxes2: shape (M, 4) in (x1, y1, x2, y2) format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

    bboxes1 = np.asarray(bboxes1, dtype=np.float32)
    bboxes2 = np.asarray(bboxes2, dtype=np.float32)

    x11, y11, x12, y12 = bboxes1[:, 0:1], bboxes1[:, 1:2], bboxes1[:, 2:3], bboxes1[:, 3:4]
    x21, y21, x22, y22 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    inter_x1 = np.maximum(x11, x21)
    inter_y1 = np.maximum(y11, y21)
    inter_x2 = np.minimum(x12, x22)
    inter_y2 = np.minimum(y12, y22)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union_area = area1 + area2 - inter_area

    iou = inter_area / np.maximum(union_area, 1e-6)
    return iou


class KalmanBoxTracker:
    """
    Linear motion model for 2D bounding boxes.
    Maintains [x1, y1, x2, y2] state and velocity [vx1, vy1, vx2, vy2].
    """
    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float], score: float = 1.0):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.score = score
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

    def update(self, bbox: Tuple[float, float, float, float], score: float = 1.0):
        """Update tracker with newly observed bounding box."""
        new_bbox = np.array(bbox, dtype=np.float32)
        # Smooth velocity estimation
        self.velocity = 0.6 * self.velocity + 0.4 * (new_bbox - self.bbox)
        self.bbox = new_bbox
        self.score = score
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self) -> np.ndarray:
        """Predict bounding box based on velocity and increase age."""
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1
        # Apply velocity with dampening
        self.bbox = self.bbox + self.velocity * 0.7
        return self.bbox

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return current bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = self.bbox
        # Ensure valid coordinate bounds
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (float(x1), float(y1), float(x2), float(y2))


class ByteTrack:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    Uses two-stage IoU association with low and high confidence detections.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,    # 1.0 - IoU threshold (0.8 = IoU >= 0.2)
        max_age: int = 30,
        min_hits: int = 3
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: List[Tuple[float, float, float, float, float]]) -> Dict[int, Tuple]:
        """
        Update tracker with detections from current frame.

        Args:
            detections: List of [(x1, y1, x2, y2, confidence), ...]

        Returns:
            Dict {track_id: (x1, y1, x2, y2), ...} for confirmed tracks only
        """
        self.frame_count += 1

        # 1. Predict new locations for existing trackers
        predicted_boxes = []
        for t in self.trackers:
            predicted_boxes.append(t.predict())
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # 2. Separate detections into high-score and low-score pools
        dets_high = []
        dets_low = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            if conf >= self.high_thresh:
                dets_high.append(det)
            elif conf >= self.low_thresh:
                dets_low.append(det)

        dets_high_boxes = np.array([d[:4] for d in dets_high]) if dets_high else np.empty((0, 4))
        dets_low_boxes = np.array([d[:4] for d in dets_low]) if dets_low else np.empty((0, 4))

        # 3. First Association: Match High Confidence Detections with Trackers
        unmatched_trackers = list(range(len(self.trackers)))
        unmatched_high_dets = list(range(len(dets_high)))
        matched_1 = []

        if len(predicted_boxes) > 0 and len(dets_high_boxes) > 0:
            iou_matrix = compute_iou_matrix(predicted_boxes, dets_high_boxes)
            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.match_thresh:
                    matched_1.append((r, c))
                    if r in unmatched_trackers:
                        unmatched_trackers.remove(r)
                    if c in unmatched_high_dets:
                        unmatched_high_dets.remove(c)

        # Update matched trackers from round 1
        for trk_idx, det_idx in matched_1:
            det = dets_high[det_idx]
            self.trackers[trk_idx].update(det[:4], score=det[4])

        # 4. Second Association: Match Remaining Trackers with Low Confidence Detections
        matched_2 = []
        if len(unmatched_trackers) > 0 and len(dets_low_boxes) > 0:
            rem_pred_boxes = predicted_boxes[unmatched_trackers]
            iou_matrix_2 = compute_iou_matrix(rem_pred_boxes, dets_low_boxes)
            cost_matrix_2 = 1.0 - iou_matrix_2
            row_ind_2, col_ind_2 = linear_sum_assignment(cost_matrix_2)

            for r, c in zip(row_ind_2, col_ind_2):
                if cost_matrix_2[r, c] <= 0.7:  # Slightly stricter IoU (>= 0.3) for low conf
                    trk_idx = unmatched_trackers[r]
                    matched_2.append((trk_idx, c))

        # Update matched trackers from round 2
        for trk_idx, det_idx in matched_2:
            det = dets_low[det_idx]
            self.trackers[trk_idx].update(det[:4], score=det[4])
            if trk_idx in unmatched_trackers:
                unmatched_trackers.remove(trk_idx)

        # 5. Create new trackers for unmatched high-confidence detections
        for det_idx in unmatched_high_dets:
            det = dets_high[det_idx]
            new_trk = KalmanBoxTracker(det[:4], score=det[4])
            self.trackers.append(new_trk)

        # 6. Remove dead trackers
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        # 7. Collect confirmed tracks (hits >= min_hits or active recent)
        confirmed_tracks = {}
        for t in self.trackers:
            if t.hits >= self.min_hits and t.time_since_update == 0:
                confirmed_tracks[t.id] = t.get_state()

        return confirmed_tracks

    def reset(self):
        """Reset tracker state."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0


class ByteTrackWrapper:
    """
    Unified tracking interface used by pipeline.py.
    Provides robust ByteTrack tracking with fallback.
    """

    def __init__(self, use_simple: bool = False, max_age: int = 30, min_hits: int = 2):
        self.tracker = ByteTrack(max_age=max_age, min_hits=min_hits)

    def update(self, detections: List[Tuple[float, float, float, float, float]]) -> Dict[int, Tuple]:
        """Update and return confirmed tracks."""
        return self.tracker.update(detections)

    def reset(self):
        """Reset tracker."""
        self.tracker.reset()
