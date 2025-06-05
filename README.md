# YOLOv8 Face Mask Detection using Custom Pascal VOC Annotations

This repository presents an end-to-end pipeline for training a YOLOv8 object detection model to identify the following classes:

- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

The dataset used is the Face Mask Detection dataset provided by Andrew Mvd on Kaggle, originally annotated in Pascal VOC (XML) format. This implementation converts the annotations to YOLO format and trains a YOLOv8 model using the Ultralytics framework.

---

## Features

- XML annotation parsing and conversion to YOLO format
- Automatic data split into training and validation sets
- Directory structuring compliant with YOLOv8 training requirements
- Training using YOLOv8 with configurable parameters
- Evaluation and inference on the validation set
- Result visualization with training metrics

---

## Dataset

Dataset: [Face Mask Detection by Andrew Mvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

Annotation format: Pascal VOC (XML)

---

## Environment Setup

```bash
pip install ultralytics xmltodict matplotlib pandas
```

---

## Directory Structure

```
working/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── face-mask-yolov8/
│   └── results.csv
└── yolov8n-mask-detection.pt
```

---

## Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Use yolov8s.pt or larger for improved accuracy

model.train(
    data="/kaggle/working/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    project="/kaggle/working",
    name="face-mask-yolov8",
    exist_ok=True
)
```

---

## Evaluation and Inference

```python
metrics = model.val()

results = model.predict(
    source="/kaggle/working/images/val",
    save=True,
    conf=0.4
)

model.save("yolov8n-mask-detection.pt")
```

---

## Visualization of Training Metrics

```python
import pandas as pd
import matplotlib.pyplot as plt

results_dir = "/kaggle/working/face-mask-yolov8"
df = pd.read_csv(f"{results_dir}/results.csv")

df.plot(
    x="epoch",
    y=["train/box_loss", "train/cls_loss", "metrics/mAP_0.5"],
    figsize=(12, 6),
    title="YOLOv8 Training Metrics"
)
plt.grid(True)
plt.show()
```

---

## Inference Output

The model outputs prediction images with bounding boxes drawn and saves them in the specified directory. Results can be found in:

```
/kaggle/working/runs/detect/predict
```

---

## License

This project is released under the MIT License.
