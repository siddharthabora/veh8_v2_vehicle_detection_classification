import argparse
import cv2
from ultralytics import YOLO

from src.geometry.line import HorizontalLine
from src.tracking.centroid_tracker import CentroidTrackerConfig
from src.counting.line_counter import LineCounter, LineCounterConfig
from src.utils.config_loader import load_yaml


CLASS_NAMES = [
    "auto", "bus", "car", "light_motor_vehicle",
    "motorcycle", "multi-axle", "tractor", "truck"
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/annotated_output.mp4")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--tracker_config", default="configs/tracker.yaml")
    parser.add_argument("--counter_config", default="configs/counter.yaml")

    return parser.parse_args()


def draw_counts(frame, counts):
    y = 40
    for class_id, name in enumerate(CLASS_NAMES):
        count = counts.get(class_id, 0)
        text = f"{name}: {count}"
        cv2.putText(
            frame,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 30


def main():
    args = parse_args()

    tracker_yaml = load_yaml(args.tracker_config) or {}
    counter_yaml = load_yaml(args.counter_config) or {}

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    ttl_seconds = tracker_yaml.get("ttl_seconds", 0.5)
    ttl_frames = max(1, int(ttl_seconds * fps))

    tracker_cfg = CentroidTrackerConfig(
        max_dist_px=int(tracker_yaml.get("max_dist_px", 80)),
        ttl_frames=ttl_frames,
        min_hits=int(tracker_yaml.get("min_hits", 3)),
    )

    counter_cfg = LineCounterConfig(
        line_frac=float(counter_yaml.get("line_frac", 0.75)),
        direction=str(counter_yaml.get("direction", "top_to_bottom")),
    )

    line = HorizontalLine.from_height(h, counter_cfg.line_frac)

    counter = LineCounter(tracker_cfg, counter_cfg)

    cap.release()
    cap = cv2.VideoCapture(args.video)

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
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):

                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                counter.process_detection(
                    cls_id=cls_id,
                    cx=cx,
                    cy=cy,
                    frame_idx=frame_idx,
                    line=line
                )

                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        counter.end_frame(frame_idx)

        cv2.line(
            frame,
            (0, line.y),
            (w, line.y),
            (0, 0, 255),
            3
        )

        draw_counts(frame, counter.counts)

        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    print("Annotated video saved to:", args.output)


if __name__ == "__main__":
    main()
