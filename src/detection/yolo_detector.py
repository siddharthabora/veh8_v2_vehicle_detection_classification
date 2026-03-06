# src/detection/yolo_detector.py

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    conf: float

    def centroid(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLO model for inference.
    Keeps scripts clean and makes detector replaceable later (TFLite/NCNN/etc).
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, frame, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5) -> List[Detection]:
        results = self.model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        r = results[0]

        dets: List[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            dets.append(Detection(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                cls_id=int(k), conf=float(c)
            ))

        return dets
