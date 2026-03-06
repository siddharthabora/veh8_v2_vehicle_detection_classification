# Veh8-v2 Vehicle Detection Model Card

## Model Overview

**Model Name:** Veh8-v2\
**Architecture:** YOLOv8s\
**Task:** Multi-class vehicle detection\
**Classes:** 8\
**Deployment Target:** Android (TFLite)\
**Training Framework:** Ultralytics YOLOv8\
**Parameters:** 11.13M\
**GFLOPs:** 28.5

Veh8-v2 is a real-time vehicle detection model trained for mixed traffic
environments, with emphasis on Indian road conditions while preserving
general detection robustness.

------------------------------------------------------------------------

## Class Definitions

  ID   Class
  ---- ---------------------
  0    auto
  1    bus
  2    car
  3    light_motor_vehicle
  4    motorcycle
  5    multi-axle
  6    tractor
  7    truck

------------------------------------------------------------------------

## Training Data Composition

  Dataset          Images Used in Training
  ---------------- ----------------------------------------------
  Veh8             6,574
  IDD              7,000
  BDD100K          Previously used in pre-fine-tuned checkpoint
  **Total (v2)**   **13,574 images**

Validation set: - 821 images\
- 2,632 labeled instances

------------------------------------------------------------------------

## Validation Metrics

### Per-Class mAP@0.5

  Class                   AP@0.5
  --------------------- --------
  auto                     0.968
  bus                      0.897
  car                      0.942
  light_motor_vehicle      0.791
  motorcycle               0.884
  multi-axle               0.855
  tractor                  0.960
  truck                    0.787

------------------------------------------------------------------------

### Aggregate Metrics

  Metric           Value
  -------------- -------
  Precision        0.829
  Recall           0.833
  mAP@0.5          0.886
  mAP@0.5:0.95     0.684

Improvement from previous baseline: - mAP@0.5:0.95 increased from \~0.64
→ 0.684

------------------------------------------------------------------------

## Inference Performance

### GPU Benchmark (Tesla T4)

  Stage            Latency
  ------------- ----------
  Preprocess        0.2 ms
  Inference         5.4 ms
  Postprocess       1.4 ms
  **Total**       \~7.0 ms

Estimated throughput: \~142 FPS (GPU theoretical)

------------------------------------------------------------------------

# Model File Specifications

| File Name               | Format        | Approx Size | Intended Use |
|------------------------|--------------|------------:|--------------|
| veh8_v2_best.pt        | PyTorch      | ~22.5 MB    | Training, evaluation, further fine-tuning |
| veh8_v2_float16.tflite | TFLite FP16  | ~22 MB      | Android deployment (recommended) |
| veh8_v2_float32.tflite | TFLite FP32  | ~44 MB      | High-precision Android inference |
| labels.txt             | Text         | <1 KB       | Class label mapping for deployment |

------------------------------------------------------------------------

## Deployment Guidance

- Use float16 TFLite for most Android devices (better speed-size balance).
- Use float32 TFLite if numerical precision is critical.
- Use the .pt file for further research, training, or benchmarking.

------------------------------------------------------------------------

## Confidence Threshold Recommendation

- 0.25 for higher recall
- 0.30–0.35 for balanced detection
- 0.40+ for counting systems to reduce false positives

------------------------------------------------------------------------

## Key Improvements in v2

-   Significant improvement in tractor detection robustness\
-   Strong performance for auto and multi-axle categories\
-   Balanced class distribution after IDD merge\
-   Reduced domain gap between structured and chaotic traffic
    environments\
-   Stable car performance despite dataset rebalance

------------------------------------------------------------------------

## Intended Use

-   Real-time vehicle counting\
-   Traffic analytics\
-   Line-crossing detection\
-   Edge deployment on Android\
-   Smart city applications\
-   On-device video inference

------------------------------------------------------------------------

## Limitations

-   Motorcycle AP at strict IoU threshold remains moderate\
-   Tracking-based counting may double count in heavy occlusion\
-   Performance sensitive to low-light conditions not explicitly trained
    on\
-   Domain shift possible in non-Indian environments

------------------------------------------------------------------------

## Ethical & Deployment Considerations

-   Model detects vehicle categories only\
-   No facial recognition or identity inference\
-   Suitable for non-surveillance analytical applications\
-   Edge deployment reduces privacy risk by avoiding cloud uploads

------------------------------------------------------------------------

## Future Roadmap

-   Add directional line-crossing counting\
-   Improve motorcycle detection under occlusion\
-   Explore YOLOv8n or quantized variant for lower-end devices\
-   Conduct FPS benchmarking on real Android hardware\
-   Introduce semi-supervised augmentation for rare classes
