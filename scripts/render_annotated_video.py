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
    parser.add_argument("--line_frac", type=float, default=None)
    parser.add_argument("--direction", type=str, default=None)

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

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("Could not open video writer with avc1 codec")

    ttl_seconds = tracker_yaml.get("ttl_seconds", 0.5)
    ttl_frames = max(1, int(ttl_seconds * fps))

    tracker_cfg = CentroidTrackerConfig(
        max_dist_px=int(tracker_yaml.get("max_dist_px", 80)),
        ttl_frames=ttl_frames,
        min_hits=int(tracker_yaml.get("min_hits", 3)),
    )

    resolved_line_frac = (
    float(args.line_frac)
    if args.line_frac is not None
    else float(counter_yaml.get("line_frac", 0.75))
    )

    resolved_direction = (
    str(args.direction)
    if args.direction is not None
    else str(counter_yaml.get("direction", "top_to_bottom"))
    )

    counter_cfg = LineCounterConfig(
    line_frac=resolved_line_frac,
    direction=resolved_direction,
    )

    print({
    "counter_config_yaml": counter_yaml,
    "resolved_line_frac": resolved_line_frac,
    "resolved_direction": resolved_direction,
    })

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

        label_text = f"line_frac: {counter_cfg.line_frac:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 4

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        padding_x = 10
        padding_y = 8

        box_w = text_w + (padding_x * 2)
        box_h = text_h + (padding_y * 2)

        box_x2 = w - 16
        box_x1 = box_x2 - box_w

        box_y2 = max(50, int(line.y) - 16)
        box_y1 = box_y2 - box_h

        cv2.rectangle(
            frame,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (255, 255, 255),
            thickness=-1,
        )

        cv2.rectangle(
            frame,
            (box_x1, box_y1),
            (box_x2, box_y2),
            (220, 220, 220),
            thickness=1,
        )

        text_x = box_x1 + padding_x
        text_y = box_y2 - padding_y

        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            (25, 25, 25),
            font_thickness,
            cv2.LINE_AA,
        )

        draw_counts(frame, counter.counts)

        writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()

    print("Annotated video saved to:", args.output)


if __name__ == "__main__":
    main()
