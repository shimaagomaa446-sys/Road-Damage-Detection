# 🚧 Road Damage Detection using YOLO11s

## 📌 Project Overview

This project focuses on detecting road damage using Deep Learning and Object Detection techniques.

The system uses the YOLO11s model to identify road damage in images, draw bounding boxes around detected areas, and display confidence scores for each prediction.

A user-friendly web application was developed using Gradio to allow users to upload road images and visualize detection results.

## 🎯 Project Objectives

- Detect different types of road damage.
- Compare different YOLO architectures.
- Evaluate model performance using object detection metrics.
- Build an interactive application using Gradio.
- Deploy the trained model for practical use.

## 🧠 Models Compared

Three YOLO models were trained and evaluated:

- YOLOv8n
- YOLO11n
- YOLO11s

## 📊 Model Comparison

| Model | mAP50 | mAP50-95 | Precision | Recall | Size (MB) |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 0.4207 | 0.2338 | 0.6031 | 0.4297 | 6.25 |
| YOLO11s | 0.4325 | 0.2471 | 0.6293 | 0.4443 | 18.34 |
| YOLO11n | 0.4319 | 0.2464 | 0.6494 | 0.4116 | 5.29 |

### Best Model

YOLO11s achieved the highest mAP50-95 score of 0.2471 among the evaluated models.

YOLO11n achieved a very similar result with a smaller model size, making it an interesting lightweight alternative.

## 🛠️ Technologies Used

- Python
- YOLO11 (Ultralytics)
- YOLOv8
- PyTorch
- Gradio
- Pandas
- Object Detection
- Computer Vision

## 🚀 Application Features

- Upload road images.
- Detect road damage automatically.
- Display bounding boxes.
- Show damage classes.
- Display confidence scores.
- Adjust confidence threshold.
- View detection results through an interactive interface.



## 📈 Evaluation

The models were compared using:

- mAP50
- mAP50-95
- Precision
- Recall
- Model Size

The reported comparison metrics are validation results.

## 🔮 Future Improvements

- Improve detection accuracy and recall.
- Evaluate inference speed (FPS).
- Add detection statistics.
- Support video-based road damage detection.
- Optimize the model for deployment.
