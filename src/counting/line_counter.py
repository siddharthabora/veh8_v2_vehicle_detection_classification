# src/counting/line_counter.py

from dataclasses import dataclass
from typing import Dict
from collections import defaultdict

from src.geometry.line import HorizontalLine
from src.tracking.centroid_tracker import Track, CentroidTrackerConfig, CentroidTracker


@dataclass
class LineCounterConfig:
    line_frac: float = 0.75  # y = frac * height
    direction: str = "top_to_bottom"  # only supported mode for now


class LineCounter:
    """Counts unique line-cross events using tracker tracklets + majority class."""
    def __init__(self, tracker_cfg: CentroidTrackerConfig, counter_cfg: LineCounterConfig):
        self.tracker = CentroidTracker(tracker_cfg)
        self.counter_cfg = counter_cfg
        self.counts: Dict[int, int] = defaultdict(int)

    def process_detection(self, cls_id: int, cx: float, cy: float, frame_idx: int, line: HorizontalLine) -> None:
        trk: Track = self.tracker.update(cx, cy, cls_id, frame_idx)

        # Count once per track
        if trk.counted or trk.hits < self.tracker.cfg.min_hits:
            return

        if self.counter_cfg.direction == "top_to_bottom":
            crossed = (trk.prev_cy < line.y) and (trk.cy >= line.y)
        else:
            raise ValueError(f"Unsupported direction: {self.counter_cfg.direction}")

        if crossed:
            maj_cls = trk.majority_class()
            self.counts[maj_cls] += 1
            trk.counted = True

    def end_frame(self, frame_idx: int) -> None:
        self.tracker.expire(frame_idx)
