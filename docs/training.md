# Reproducible Training & Export Instructions

This guide explains how to train your own YOLOv8 vehicle detection model and export it to TFLite for mobile deployment. It assumes familiarity with Python and basic machine‑learning concepts but does not require advanced programming skills.

## Prerequisites

- A GPU‑enabled environment (Google Colab or Kaggle) with CUDA support.
- Python 3.8 or higher.
- Ultralytics YOLOv8 package (`pip install ultralytics`).

## 1. Prepare the Dataset

1. Collect or download images of vehicles. 
2. Annotate the images with bounding boxes for each vehicle class using tools such as Roboflow or CVAT.
3. Export annotations in YOLO format, producing a directory structure such as:

    vehicle-data/
      images/
        train/
        val/
      labels/
        train/
        val/

4. Create a `data.yaml` file that specifies dataset paths and class names:

    path: /content/vehicle-data
    train: images/train
    val: images/val
    names:
      0: auto
      1: bus
      2: car
      3: lcv
      4: motorcycle
      5: multiaxle
      6: tractor
      7: truck

## 2. Train the Model

Install the Ultralytics package and start training:

    pip install ultralytics

    # train using pre‑trained YOLOv8n weights
    yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 batch=16 name=vehicle_yolov8n

- `epochs`: number of passes through the dataset (increase for better accuracy).
- `imgsz`: input resolution.
- `batch`: adjust based on GPU memory.
- `name`: identifier for the run; results are stored in `runs/detect/<name>`.

Training outputs a `.pt` checkpoint in `runs/detect/vehicle_yolov8n/weights/best.pt`.

## 3. Evaluate the Model

After training, evaluate on the validation set:

    yolo task=detect mode=val model=runs/detect/vehicle_yolov8n/weights/best.pt data=data.yaml imgsz=640

This prints mAP, precision and recall metrics and writes a results file in the run directory.

## 4. Export to TFLite

Ultralytics makes exporting straightforward:

    yolo mode=export model=runs/detect/vehicle_yolov8n/weights/best.pt format=tflite

By default this exports a full 32‑bit float model (`best.tflite`). To obtain a float16 model:

    yolo mode=export model=runs/detect/vehicle_yolov8n/weights/best.pt format=tflite int8=False half=True

The exported TFLite files will appear in `runs/detect/vehicle_yolov8n/weights`.

## 5. Benchmark on Device

Before deploying, benchmark your model on a representative device using the TensorFlow Lite Benchmark Tool to ensure acceptable latency and memory usage. See `docs/metrics.md` for baseline numbers.

## 6. Update the Repository

To integrate your new model:

1. Replace `models/best_float32.tflite` and `models/best_float16.tflite` with your exported files.
2. Update `docs/metrics.md` with new validation and benchmark metrics.
3. Commit and push the changes to GitHub.

Following these steps ensures that others can reproduce your training pipeline and understand how the model was obtained.