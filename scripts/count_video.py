# scripts/count_video.py

import argparse
import cv2
from ultralytics import YOLO

from src.geometry.line import HorizontalLine
from src.tracking.centroid_tracker import CentroidTrackerConfig
from src.counting.line_counter import LineCounter, LineCounterConfig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)

    return parser.parse_args()


def main():

    args = parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read video")

    height, width = frame.shape[:2]

    line = HorizontalLine.from_height(height, 0.75)

    tracker_cfg = CentroidTrackerConfig()
    counter_cfg = LineCounterConfig()

    counter = LineCounter(tracker_cfg, counter_cfg)

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

        if r.boxes is not None:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cls_id in zip(boxes, classes):

                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                counter.process_detection(
                    cls_id,
                    cx,
                    cy,
                    frame_idx,
                    line
                )

        counter.end_frame(frame_idx)

        frame_idx += 1

    cap.release()

    print("Final counts:")
    print(counter.counts)


if __name__ == "__main__":
    main()
