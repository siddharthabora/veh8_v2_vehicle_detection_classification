# src/counting/line_counter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

from src.geometry.line import HorizontalLine
from src.tracking.centroid_tracker import Track, CentroidTrackerConfig, CentroidTracker


@dataclass
class LineCounterConfig:
    line_frac: float = 0.75  # y = frac * height
    direction: str = "top_to_bottom"  # supported: "top_to_bottom"


@dataclass(frozen=True)
class CrossingEvent:
    """A single confirmed line-crossing event (counted once per track)."""
    frame: int
    class_id: int
    track_id: int


class LineCounter:
    """
    Counts unique line-cross events using tracker tracklets + majority class.
    Also records structured crossing events for downstream analytics/logging.
    """

    def __init__(self, tracker_cfg: CentroidTrackerConfig, counter_cfg: LineCounterConfig):
        self.tracker = CentroidTracker(tracker_cfg)
        self.counter_cfg = counter_cfg

        # class_id -> count
        self.counts: Dict[int, int] = defaultdict(int)

        # buffered events produced during processing
        self._events: List[CrossingEvent] = []

    def process_detection(
        self,
        cls_id: int,
        cx: float,
        cy: float,
        frame_idx: int,
        line: HorizontalLine,
    ) -> None:
        trk: Track = self.tracker.update(cx, cy, cls_id, frame_idx)

        # Count once per track, only after it has stabilized (min_hits)
        if trk.counted or trk.hits < self.tracker.cfg.min_hits:
            return

        if self.counter_cfg.direction != "top_to_bottom":
            raise ValueError(f"Unsupported direction: {self.counter_cfg.direction}")

        crossed = (trk.prev_cy < line.y) and (trk.cy >= line.y)
        if not crossed:
            return

        maj_cls = trk.majority_class()
        self.counts[maj_cls] += 1
        trk.counted = True

        self._events.append(
            CrossingEvent(frame=frame_idx, class_id=maj_cls, track_id=trk.track_id)
        )

    def pop_events(self) -> List[CrossingEvent]:
        """
        Returns all buffered crossing events since the last call, then clears the buffer.
        Call this once per frame from the runner script to stream events to CSV.
        """
        events = self._events
        self._events = []
        return events

    def end_frame(self, frame_idx: int) -> None:
        """Expire stale tracks at the end of each frame."""
        self.tracker.expire(frame_idx)
