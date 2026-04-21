# Real-Time Anomaly Detection System (Jetson Orin Nano)

## Overview

This project implements a real-time anomaly detection system using object detection. It is designed to run on NVIDIA Jetson devices and also supports execution on Windows systems.

The system monitors a live camera feed and flags anomalies based on predefined rules such as object count violations and forbidden object detection.

---

## Platform Support

### Jetson (Primary)

* Device: Jetson Orin Nano (15W)
* Inference Engine: jetson-inference (detectNet)
* Model: SSD-Mobilenet-v2
* Camera: USB (/dev/video0)

### Windows (Fallback Mode)

* Framework: OpenCV + Ultralytics YOLOv8
* Purpose: Development and testing without Jetson hardware
* Model: yolov8n (lightweight)

---

## Approach

### Object Detection

* Jetson: detectNet with SSD-Mobilenet-v2
* Windows: YOLOv8 via Ultralytics

### Anomaly Rules

* Count-based:

  * Trigger if number of people > 2
* Forbidden objects:

  * Detect "cell phone" or "backpack"

---

## System Pipeline

Camera → Object Detection → Rule Engine → Logging → Web UI

---

## Features

* Real-time object detection
* Rule-based anomaly detection
* Cyberpunk-style web dashboard
* Live MJPEG video streaming using Flask
* Real-time statistics:

  * total anomalies
  * last anomaly detected
  * time since last anomaly
* Event logging with cooldown mechanism

---

## Results

### Anomaly Detection Example

![Anomaly](screenshots/anomaly_detected.png)

### Full System UI

![UI](screenshots/UI.png)

---

## Output

The system generates a CSV log file:

anomaly_log.csv

Example:
Timestamp, Anomaly_Type, Details
2026-04-21 18:32:10, Forbidden Object, Detected cell phone

---

## How to Run

### On Jetson

Run inside the jetson-inference Docker container:

```bash
python3 app.py
```

Then open:
http://<jetson-ip>:5000

---

### On Windows

Install dependencies:

```bash
pip install flask ultralytics opencv-python numpy
```

Run:

```bash
python app.py
```

Then open:
http://localhost:5000

---

## Project Structure

```
project/
│
├── app.py
├── anomaly_log.csv
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── screenshots/
│   ├── anomaly_detected.png
│   ├── ui.png
│
└── README.md
```

---

## Key Highlights

* Works on both Jetson (GPU) and Windows (CPU fallback)
* Modular anomaly detection logic
* Real-time streaming and UI dashboard
* Logging with cooldown to prevent duplicate entries


