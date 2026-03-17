from src.counting.line_counter import LineCounter, LineCounterConfig
from src.tracking.centroid_tracker import CentroidTrackerConfig
from src.geometry.line import HorizontalLine

def test_counts_once_on_crossing():
    # tracker config tuned for deterministic test
    tracker_cfg = CentroidTrackerConfig(max_dist_px=99999, ttl_frames=1000, min_hits=1)
    counter_cfg = LineCounterConfig(line_frac=0.5, direction="top_to_bottom")

    counter = LineCounter(tracker_cfg, counter_cfg)

    h = 100
    line = HorizontalLine.from_height(h, counter_cfg.line_frac)  # y=50

    # simulate a single object moving down across the line
    # frame 0: above line (cy=40), frame 1: below line (cy=60)
    counter.process_detection(cls_id=2, cx=10, cy=40, frame_idx=0, line=line)
    counter.end_frame(0)
    assert counter.counts.get(2, 0) == 0

    counter.process_detection(cls_id=2, cx=10, cy=60, frame_idx=1, line=line)
    counter.end_frame(1)

    # should count exactly once
    assert counter.counts.get(2, 0) == 1

    # should not double count if still below line in later frames
    counter.process_detection(cls_id=2, cx=10, cy=70, frame_idx=2, line=line)
    counter.end_frame(2)
    assert counter.counts.get(2, 0) == 1