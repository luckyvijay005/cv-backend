"""
Person detection using YOLOv8.

Yields bounding boxes (x, y, w, h, confidence) for all detected people in a frame.
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class PersonDetector:
    """Detects people in frames using YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", device: int = 0):
        """
        Args:
            model_name: YOLOv8 model variant (e.g., 'yolov8n.pt', 'yolov8s.pt')
            device: GPU device index (0 for first GPU, -1 for CPU)
        """
        import torch
        self.model = YOLO(model_name)

        # Auto-detect device: if GPU not available, fall back to CPU
        if device >= 0:
            if torch.cuda.is_available():
                self.model.to(device)
                self.device = device
            else:
                print(f"GPU device {device} not available. Falling back to CPU.")
                self.model.to('cpu')
                self.device = -1
        else:
            self.model.to('cpu')
            self.device = -1

        self.person_class_id = 0  # COCO class 0 = person

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect people in a frame.

        Args:
            frame: Input frame (numpy array, shape: (H, W, 3))
            conf_threshold: Confidence threshold (0.0-1.0)

        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
            where (x1, y1) is top-left, (x2, y2) is bottom-right
        """
        dev = self.device if self.device >= 0 else 'cpu'
        results = self.model.predict(frame, conf=conf_threshold, classes=[self.person_class_id], device=dev, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                detections.append((float(x1), float(y1), float(x2), float(y2), float(conf)))

        return detections

    def detect_batch(self, frames: List[np.ndarray], conf_threshold: float = 0.5) -> List[List[Tuple]]:
        """
        Detect people in multiple frames (more efficient for batch processing).

        Args:
            frames: List of input frames
            conf_threshold: Confidence threshold

        Returns:
            List of detection lists (one per frame)
        """
        results = self.model.predict(
            frames, conf=conf_threshold, verbose=False)

        batch_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == self.person_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    detections.append((float(x1), float(y1), float(
                        x2), float(y2), float(confidence)))
            batch_detections.append(detections)

        return batch_detections

    def bbox_to_center_size(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Convert bbox (x1, y1, x2, y2) to center format (cx, cy, w, h)."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return cx, cy, w, h
