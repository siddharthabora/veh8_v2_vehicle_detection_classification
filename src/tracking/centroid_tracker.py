# src/tracking/centroid_tracker.py

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math
from collections import defaultdict


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    prev_cy: float
    last_frame: int
    hits: int = 1
    counted: bool = False
    cls_hist: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, cx: float, cy: float, cls_id: int, frame_idx: int) -> None:
        self.prev_cy = self.cy
        self.cx = cx
        self.cy = cy
        self.last_frame = frame_idx
        self.hits += 1
        self.cls_hist[cls_id] += 1

    def majority_class(self) -> int:
        return max(self.cls_hist.items(), key=lambda kv: kv[1])[0]


@dataclass
class CentroidTrackerConfig:
    max_dist_px: int = 80
    ttl_frames: int = 15
    min_hits: int = 3


class CentroidTracker:
    """Minimal centroid-based tracker: distance association + TTL expiry."""
    def __init__(self, cfg: CentroidTrackerConfig):
        self.cfg = cfg
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def _match(self, cx: float, cy: float, frame_idx: int) -> Optional[int]:
        best_tid = None
        best_dist = 1e18
        for tid, trk in self.tracks.items():
            if frame_idx - trk.last_frame > self.cfg.ttl_frames:
                continue
            dist = math.hypot(cx - trk.cx, cy - trk.cy)
            if dist < self.cfg.max_dist_px and dist < best_dist:
                best_dist = dist
                best_tid = tid
        return best_tid

    def update(self, cx: float, cy: float, cls_id: int, frame_idx: int) -> Track:
        tid = self._match(cx, cy, frame_idx)
        if tid is None:
            tid = self._next_id
            self._next_id += 1
            trk = Track(
                track_id=tid,
                cx=cx,
                cy=cy,
                prev_cy=cy,
                last_frame=frame_idx,
            )
            trk.cls_hist[cls_id] += 1
            self.tracks[tid] = trk
        else:
            self.tracks[tid].update(cx, cy, cls_id, frame_idx)
        return self.tracks[tid]

    def expire(self, frame_idx: int) -> None:
        dead = [tid for tid, trk in self.tracks.items() if frame_idx - trk.last_frame > self.cfg.ttl_frames]
        for tid in dead:
            del self.tracks[tid]

    @property
    def total_tracks_created(self) -> int:
        return self._next_id - 1
