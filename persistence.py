"""
Student registry for persistent IDs across videos/days.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


class StudentRegistry:
    """Persistent registry of student appearance descriptors."""

    def __init__(self, registry_path: str = 'data/student_registry.json', match_threshold: float = 0.75):
        self.registry_path = registry_path
        self.match_threshold = match_threshold
        self.data = {
            'next_student_id': 1,
            'records': {}
        }
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {'next_student_id': 1, 'records': {}}
        else:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

    def save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    @staticmethod
    def _normalize_descriptor(descriptor: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            return descriptor / norm
        return descriptor

    @staticmethod
    def _frame_crop_descriptor(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((48,), dtype=np.float32)

        crop = cv2.resize(crop, (128, 128))
        descriptor = []
        for channel in range(3):
            hist = cv2.calcHist([crop], [channel], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            descriptor.append(hist)
        descriptor = np.concatenate(descriptor)
        return StudentRegistry._normalize_descriptor(descriptor)

    def _track_descriptor(self, frames: object, bboxes: List[Tuple[float, float, float, float]], start_frame: int) -> np.ndarray:
        if len(bboxes) == 0:
            return np.zeros((48,), dtype=np.float32)
        idx_in_bboxes = len(bboxes) // 2
        best_frame_idx = start_frame + idx_in_bboxes
        best_frame_idx = min(best_frame_idx, len(frames) - 1)
        bbox = bboxes[idx_in_bboxes]
        return self._frame_crop_descriptor(frames[best_frame_idx], bbox)

    def _compare_descriptors(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def match_descriptor(self, descriptor: np.ndarray) -> Optional[int]:
        best_id = None
        best_score = 0.0
        for student_id, record in self.data['records'].items():
            ref_vec = np.array(record.get('descriptor', []), dtype=np.float32)
            if ref_vec.size == 0:
                continue
            score = self._compare_descriptors(descriptor, ref_vec)
            if score > best_score:
                best_score = score
                best_id = int(student_id)

        if best_score >= self.match_threshold:
            return best_id
        return None

    def register_track(self, track_id: int, frames: object, bboxes: List[Tuple[float, float, float, float]], start_frame: int) -> int:
        descriptor = self._track_descriptor(frames, bboxes, start_frame)
        matched_id = self.match_descriptor(descriptor)
        if matched_id is None:
            matched_id = self.data['next_student_id']
            self.data['next_student_id'] += 1
            self.data['records'][str(matched_id)] = {
                'created_at': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'descriptor': descriptor.tolist(),
                'sample_count': 1,
            }
        else:
            record = self.data['records'][str(matched_id)]
            existing = np.array(record.get('descriptor', []), dtype=np.float32)
            if existing.size == descriptor.size:
                updated = np.mean([existing, descriptor], axis=0)
                record['descriptor'] = self._normalize_descriptor(
                    updated).tolist()
            record['last_seen'] = datetime.now().isoformat()
            record['sample_count'] = record.get('sample_count', 1) + 1

        return matched_id

    def register_tracks(
        self,
        frames: object,
        track_history: Dict[int, List[Tuple[float, float, float, float]]],
        track_start_frames: Dict[int, int],
        cached_crops: Optional[Dict[int, np.ndarray]] = None
    ) -> Dict[int, int]:
        mapping = {}

        for track_id, bboxes in track_history.items():
            if cached_crops and track_id in cached_crops and cached_crops[track_id] is not None:
                crop = cached_crops[track_id]
                crop_resized = cv2.resize(crop, (128, 128))
                descriptor = []
                for channel in range(3):
                    hist = cv2.calcHist([crop_resized], [channel], None, [16], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    descriptor.append(hist)
                descriptor = np.concatenate(descriptor)
                descriptor = StudentRegistry._normalize_descriptor(descriptor)
            else:
                descriptor = self._track_descriptor(frames, bboxes, track_start_frames.get(track_id, 0))

            matched_id = self.match_descriptor(descriptor)
            if matched_id is None:
                matched_id = self.data['next_student_id']
                self.data['next_student_id'] += 1
                self.data['records'][str(matched_id)] = {
                    'created_at': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'descriptor': descriptor.tolist(),
                    'sample_count': 1,
                }
            else:
                record = self.data['records'][str(matched_id)]
                existing = np.array(record.get('descriptor', []), dtype=np.float32)
                if existing.size == descriptor.size:
                    updated = np.mean([existing, descriptor], axis=0)
                    record['descriptor'] = self._normalize_descriptor(updated).tolist()
                record['last_seen'] = datetime.now().isoformat()
                record['sample_count'] = record.get('sample_count', 1) + 1

            mapping[track_id] = matched_id

        self.save_registry()
        return mapping

    def save_student_crops(self, frames: List[np.ndarray], track_history: Dict[int, List[Tuple[float, float, float, float]]], output_dir: str = 'data/registry'):
        os.makedirs(output_dir, exist_ok=True)
        for track_id, bboxes in track_history.items():
            student_id = self.register_track(track_id, frames, bboxes)
            if len(bboxes) == 0:
                continue
            best_frame_idx = min(len(bboxes) // 2, len(frames) - 1)
            x1, y1, x2, y2 = [int(v) for v in bboxes[best_frame_idx]]
            frame = frames[best_frame_idx]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (256, 256))
            cv2.imwrite(os.path.join(
                output_dir, f'student_{student_id:03d}.jpg'), crop)
        self.save_registry()


