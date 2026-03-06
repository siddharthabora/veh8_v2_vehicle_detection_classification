# Veh8-v2 Version Information

## Model Version
Veh8-v2

## Release Date
2026-02

## Architecture
YOLOv8s

## Parameters
11,128,680

## GFLOPs
28.5

## Input Resolution
640 x 640

## Classes
8 (auto, bus, car, light_motor_vehicle, motorcycle, multi-axle, tractor, truck)

## Training Data
- Veh8: 6,574 training images
- IDD: 7,000 training images
- BDD100K: 600 in initialization checkpoint

Total training images (v2): 13,574

## Validation Performance
- Precision: 0.829
- Recall: 0.833
- mAP@0.5: 0.886
- mAP@0.5:0.95: 0.684

## Notes
Initialized from a Veh8 + BDD pre-trained checkpoint and fine-tuned with IDD for improved mixed-traffic robustness.
