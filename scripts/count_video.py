# scripts/count_video.py

import argparse
import cv2
from ultralytics import YOLO

from src.geometry.line import HorizontalLine
from src.tracking.centroid_tracker import CentroidTrackerConfig
from src.counting.line_counter import LineCounter, LineCounterConfig
from src.utils.config_loader import load_yaml
from src.evaluation.event_logger import EventLogger


CLASS_NAMES = [
    "auto", "bus", "car", "light_motor_vehicle",
    "motorcycle", "multi-axle", "tractor", "truck"
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--tracker_config", default="configs/tracker.yaml")
    parser.add_argument("--counter_config", default="configs/counter.yaml")

    return parser.parse_args()


def main():
    args = parse_args()

    tracker_cfg_yaml = load_yaml(args.tracker_config) or {}
    counter_cfg_yaml = load_yaml(args.counter_config) or {}

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    height, width = frame.shape[:2]

    ttl_seconds = float(tracker_cfg_yaml.get("ttl_seconds", 0.5))
    ttl_frames = max(1, int(ttl_seconds * fps))

    tracker_cfg = CentroidTrackerConfig(
        max_dist_px=int(tracker_cfg_yaml.get("max_dist_px", 80)),
        ttl_frames=ttl_frames,
        min_hits=int(tracker_cfg_yaml.get("min_hits", 3)),
    )

    counter_cfg = LineCounterConfig(
        line_frac=float(counter_cfg_yaml.get("line_frac", 0.75)),
        direction=str(counter_cfg_yaml.get("direction", "top_to_bottom")),
    )

    line = HorizontalLine.from_height(height, counter_cfg.line_frac)

    counter = LineCounter(tracker_cfg, counter_cfg)

    logger = EventLogger()

    cap.release()
    cap = cv2.VideoCapture(args.video)

    print("Decoded FPS:", fps)
    print("Frame size:", width, "x", height)
    print("LINE_Y:", line.y)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cls_id in zip(boxes, classes):

                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                counter.process_detection(
                    cls_id=cls_id,
                    cx=cx,
                    cy=cy,
                    frame_idx=frame_idx,
                    line=line
                )

        counter.end_frame(frame_idx)

        # collect crossing events from counter
        events = counter.pop_events()

        for ev in events:
            logger.log_event(
                frame_idx=ev.frame,
                fps=fps,
                class_id=ev.class_id,
                class_name=CLASS_NAMES[ev.class_id],
                track_id=ev.track_id
            )

        frame_idx += 1

    cap.release()

    logger.save()

    print("\nFinal counts (class_name -> count):")

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:20s} {counter.counts.get(i, 0)}")

    print("\nTotal tracks created:", counter.tracker.total_tracks_created)


if __name__ == "__main__":
    main()