# Dataset Notes

## Sources

The training data for this model comes from three primary sources:

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
  - Used: 602 images filtered for vehicle classes and used to train the first version. So BDD knowledge exists in weights in this model, but was not actively retrained.

## Preprocessing

- Images were resized to 640×640 pixels.
- Bounding box labels were converted to YOLO format `(x_center, y_center, width, height)` normalized between 0 and 1.
- Data augmentation included random flips, crops, color jitter and mosaic augmentation to improve robustness.

## Label Schema

The model uses the following class names in order:

```
auto
bus
car
lcv
motorcycle
multiaxle
tractor
truck
```

Ensure that your own dataset follows the same label order when retraining; the order determines the class index.