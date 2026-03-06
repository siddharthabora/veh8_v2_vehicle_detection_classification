# Validation Metrics & Benchmarks

## Validation Results

Evaluation performed on:
- Validation images: 821
- Validation instances: 2,632
- Image size: 640
- Confidence threshold during validation: default
- Model: YOLOv8s
- Parameters: 11,128,680
- GFLOPs: 28.5

Evaluation on the validation set yielded the following per‑class metrics (mAP@0.5):

| Class       | AP@0.5|
|-------------|-------|
| auto        | 0.968 |
| bus         | 0.897 |
| car         | 0.942 |
| lcv         | 0.791 |
| motorcycle  | 0.884 |
| multi-axle  | 0.855 |
| tractor     | 0.960 |
| truck       | 0.787 |

## Aggregate Metrics: 

| Metric        | Value |
|---------------|-------|
| Precision (P) | 0.829 |
| Recall (R)    | 0.833 |
| mAP@0.5       | 0.886 |
| mAP@0.5:0.95  | 0.684 |

## On‑device Benchmarks

- Inference timing measured on Tesla T4 (batch=1):
    - Preprocess: 0.2 ms
    - Inference: 5.4 ms
    - Postprocess: 1.4 ms

- Total per image ≈ 7.0 ms

- Estimated FPS (GPU T4):
~142 FPS theoretical (1 / 0.007)

Note: Android performance will significantly vary depending on device.