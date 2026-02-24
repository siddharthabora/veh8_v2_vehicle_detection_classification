
# Veh8-v2: Real-Time Multi-Class Vehicle Detection (YOLOv8)

High-performance 8-class vehicle detection model optimized for real-time inference and Android edge deployment.

---

## Overview

**Veh8-v2** is a YOLOv8s-based object detection model trained on a hybrid dataset combining structured and chaotic traffic environments. It is optimized for:

- Indian mixed-traffic conditions  
- Real-time vehicle analytics  
- On-device inference using TFLite  
- Vehicle counting and smart traffic applications  

---

## Model Architecture

- **Base Model:** YOLOv8s  
- **Parameters:** 11.13M  
- **GFLOPs:** 28.5  
- **Input Size:** 640×640  
- **Classes:** 8  
- **Framework:** Ultralytics YOLOv8  

---

## Class Definitions

| ID | Class |
|----|----------------------|
| 0  | auto |
| 1  | bus |
| 2  | car |
| 3  | light_motor_vehicle |
| 4  | motorcycle |
| 5  | multi-axle |
| 6  | tractor |
| 7  | truck |

---
## Dataset Sources

1. Vehicle Detection 8 Classes | Object Detection
  - Link: https://www.kaggle.com/datasets/sakshamjn/vehicle-detection-8-classes-object-detection
  - License: Unknown
  - Used: 6547 CCTV images filtered for vehicle classes

2. Indian Driving Dataset-Detections (YOLOv11)
  - Link: https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11
  - License: CC BY-NC-SA 4.0
  - Used: 7000 images filtered for vehicle classes

3. BDD100K Detection Dataset
  - Link: https://bdd-data.berkeley.edu
  - License: https://doc.bdd100k.com/license.html#license
  - Used: 602 images filtered for vehicle classes and used to train v1. So BDD knowledge exists in weights in this model, but was not actively retrained.

## Dataset Composition (Training)

| Dataset | Images Used |
|----------|------------|
| Veh8 (Primary) | 6,574 |
| IDD (Indian Driving Dataset) | 7,000 |
| BDD100K | Used in initial checkpoint |
| **Total (v2)** | **13,574 images** |

Validation:
- 821 images  
- 2,632 labeled instances  

---

## Validation Performance

### Per-Class mAP@0.5

| Class | AP@0.5 |
|-----------------------|-------:|
| auto | 0.968 |
| bus | 0.897 |
| car | 0.942 |
| light_motor_vehicle | 0.791 |
| motorcycle | 0.884 |
| multi-axle | 0.855 |
| tractor | 0.960 |
| truck | 0.787 |

---

### Aggregate Metrics

| Metric | Value |
|---------------|------:|
| Precision | 0.829 |
| Recall | 0.833 |
| mAP@0.5 | 0.886 |
| mAP@0.5:0.95 | 0.684 |

Improvement from previous baseline:
mAP@0.5:0.95 increased from ~0.64 → **0.684**

---

## Inference Performance

### GPU Benchmark (Tesla T4)

| Stage | Latency |
|-------------|--------:|
| Preprocess | 0.2 ms |
| Inference | 5.4 ms |
| Postprocess | 1.4 ms |
| **Total** | ~7.0 ms |

Estimated throughput: ~142 FPS (theoretical GPU batch=1)

---

## TFLite Deployment (Android)

| Model Variant | Approx Size | Est. CPU Latency | Est. FPS |
|-----------------------|------------|-----------------:|---------:|
| Float32 TFLite | ~44 MB | 45–70 ms | 14–22 FPS |
| Float16 TFLite | ~22 MB | 30–50 ms | 20–33 FPS |

Actual device performance depends on chipset, NNAPI/GPU delegate usage, thermal throttling, and camera pipeline overhead.

---

## Use Cases

- Real-time vehicle counting  
- Traffic flow analytics  
- Line-crossing vehicle tracking  
- Smart city deployments  
- Edge-based inference systems  
- Mobile AI applications  

---

## Inference Example

```python
from ultralytics import YOLO

model = YOLO("best_veh8+bdd100k+idd_v2.1.pt") 

results = model.predict(
    source="video.mp4",         #Replace with your actual test video
    imgsz=640,
    conf=0.30,                  #Choose a hgher conf for lower ghost boxes
    save=True
)
```

The exported `.pt` model can be loaded and executed in any Python notebook environment (Kaggle, Colab, or local Jupyter) with:

```bash
pip install ultralytics
```

---

## Limitations

- Motorcycle detection at strict IoU remains comparatively weaker  
- Tracking-based counting may double-count under occlusion  
- Low-light robustness not explicitly optimized  
- Domain shift possible in non-Indian traffic environments  

---

## Ethical Considerations

- Detects vehicle categories only  
- No facial recognition  
- Suitable for non-invasive traffic analytics  
- Edge deployment reduces privacy exposure  

---

## =Roadmap

- Direction-based line crossing analytics  
- Improved motorcycle performance  
- Low-light data augmentation  
- Quantized edge-optimized variants  
- Production Android integration  

---

## License

MIT
