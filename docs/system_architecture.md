# System Architecture: Vehicle Detection Tracking and Line Counting

## High level pipeline

```mermaid
flowchart TD

A[Input Video or Camera Stream] --> B[Frame Decoder OpenCV or CameraX]

B --> C[Vehicle Detector YOLOv8 veh8 model]

C --> D[Frame Detections Bounding Boxes Class ID Confidence]

D --> E[Tracking Layer Experiments ByteTrack DeepSORT Current Centroid Tracker]

E --> F[Tracklets Track ID Centroid Trajectory Class History]

F --> G[Counting Layer Line Crossing Event Logic]

G --> H[System Outputs]

H --> H1[Class Wise Vehicle Counts]

H --> H2[Event Log CSV Frame Time Class Track]

H --> H3[Annotated Video Bounding Boxes Labels Counting Line]
