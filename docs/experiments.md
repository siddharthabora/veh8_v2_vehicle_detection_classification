# Experiments Log: Tracking + Counting

## Tracking Experiments

```mermaid
flowchart TD

A[YOLO Detection Output] --> B[Experiment 1 ByteTrack]

A --> C[Experiment 2 DeepSORT]

A --> D[Experiment 3 Minimal Centroid Tracker]

B --> E[Observation ID fragmentation in dense traffic]

C --> F[Observation Too many track IDs heavy compute]

D --> G[Observation Stable near counting line lightweight]

E --> H[Rejected for counting use case]

F --> H[Rejected for counting use case]

G --> I[Selected for line crossing counting]

## Goal
Count vehicles by class using a horizontal line at 0.75 of frame height, robust to dense traffic and occlusion, targeting edge deployment.

---

## Week 1: Detection Training (Baseline)
### Model
- YOLOv8s
- Dataset: Veh8 + BDD100K + IDD
- Classes: 8 (auto, bus, car, light_motor_vehicle, motorcycle, multi-axle, tractor, truck)
- Input: 640x640

### What worked
- Good class accuracy on sparse to moderate traffic
- Exported to TFLite for Android deployment

### Known limitations
- Detector-only (no tracking, no counting)
- Occlusion causes missed boxes or jitter across frames

---

## Week 2: Tracking and Counting (System Design)

### Experiment 1: ByteTrack
**Why**
- Fast, commonly used for YOLO-based tracking

**What I tried**
- Tuned thresholds for 60 FPS videos
- Tested on dense traffic clips

**What I observed**
- Track fragmentation in dense traffic
- ID inflation when association was unstable
- Counting errors due to ID switching

**Decision**
- Not reliable enough for line-cross counting in dense traffic

---

### Experiment 2: DeepSORT
**Why**
- Appearance embeddings should reduce ID switches under occlusion

**What I tried**
- YOLO detections → DeepSORT update loop on first 5 seconds

**What I observed**
- Track explosion (too many IDs in dense scenes)
- Too heavy for edge deployment targets

**Decision**
- Rejected for Android 60 FPS target

---

### Experiment 3: Minimal Centroid Tracker + Event-Based Counting
**Why**
- Traffic counting does not need long-term identity, only stable event detection near a line

**Core idea**
- Track objects locally using centroid association + TTL
- Trigger count only when centroid crosses the line top → bottom
- Prevent double counting using stability rules (min hits, TTL, optional exit gating)

**What worked**
- Matched manual ground truth on test clips
- Resolution-agnostic by computing LINE_Y from frame height

**Known limitations**
- Track fragmentation still exists globally (acceptable)
- Needs extra event gating for very dense junction scenes

---

## Current Status
- Working pipeline: Detection → Centroid Tracking → Line Crossing Count
- Next: multi-direction counting, stronger event gating, Android real-time integration

## Test Videos Used
- count_test_1.mp4: manual GT validated
- count_test_2.mp4: manual GT validated
