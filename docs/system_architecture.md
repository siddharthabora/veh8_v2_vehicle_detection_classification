# System Architecture: Vehicle Detection + Tracking + Line-Cross Counting

## High-level pipeline

```mermaid
flowchart TD
    A[Input Video / Camera Stream] --> B[Frame Decoder: OpenCV / CameraX]
    B --> C[Detector: YOLOv8 veh8_v2.1 (.pt / later TFLite)]
    C --> D[Detections per frame\n(bboxes, class_id, conf)]
    D --> E[Tracking Layer\n(ByteTrack / DeepSORT experiments)\nCurrent: Minimal Centroid Tracker]
    E --> F[Tracklets\n(track_id, centroid trajectory, class history)]
    F --> G[Counting Layer\nLine crossing event logic\n(top-to-bottom, configurable line_frac)]
    G --> H[Outputs]
    H --> H1[Class-wise counts]
    H --> H2[Event log CSV\n(frame, time, class, track_id)]
    H --> H3[Annotated video\n(bboxes, labels, line, counts)]
