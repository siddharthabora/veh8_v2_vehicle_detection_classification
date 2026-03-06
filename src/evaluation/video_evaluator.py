"""
Video evaluation utilities.

Purpose
-------
Provides tools to run the detection + tracking + counting pipeline
on a video and produce annotated outputs for manual verification.

Outputs
-------
- Annotated video (bounding boxes, class, confidence, track ID)
- Visual counting line
- Debug overlays
"""

import cv2


def draw_bbox(frame, box, label, color=(0,255,0)):

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

    cv2.putText(
        frame,
        label,
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_line(frame, y):

    height, width = frame.shape[:2]

    cv2.line(
        frame,
        (0, y),
        (width, y),
        (0,0,255),
        2
    )
