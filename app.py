import time
import csv
import os
import threading
from flask import Flask, render_template, Response, jsonify
from datetime import datetime

# Attempt to import Jetson libraries
try:
    import jetson.inference
    import jetson.utils
    HAS_JETSON = True
except ImportError:
    print("Warning: jetson.inference or jetson.utils not found. This script is intended to run on a Jetson device.")
    HAS_JETSON = False

app = Flask(__name__)

# --- Configuration ---
CAMERA_PATH = "/dev/video0" # Default USB camera path
MODEL_NAME = "ssd-mobilenet-v2"
LOG_FILE = "anomaly_log.csv"

# Global state for UI updates
state = {
    "anomaly_count": 0,
    "last_anomaly": "None",
    "time_since_last": "N/A",
    "current_log": [],
    "last_anomaly_time": 0
}

# Cooldown logic to prevent spamming
COOLDOWN_SECONDS = 5
last_logged_time = 0

def init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Anomaly_Type", "Details"])

def log_anomaly(anomaly_type, details):
    global last_logged_time, state
    current_time = time.time()
    
    if current_time - last_logged_time < COOLDOWN_SECONDS:
        return # Cooldown active
        
    last_logged_time = current_time
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_str, anomaly_type, details])
        
    state["anomaly_count"] += 1
    state["last_anomaly"] = f"{anomaly_type}: {details}"
    state["last_anomaly_time"] = current_time
    state["current_log"].insert(0, {"time": timestamp_str, "type": anomaly_type, "details": details})
    
    # Keep log small for UI
    if len(state["current_log"]) > 10:
        state["current_log"].pop()

def draw_cyberpunk_box(img, x1, y1, x2, y2, label, color=(232, 217, 5)):
    import cv2
    import math
    
    # Calculate center and dimensions
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    w = x2 - x1
    h = y2 - y1
    r = int(max(w, h) / 2 * 1.1)
    r = max(r, 20) # Minimum radius
    
    # 1. Draw central crosshair
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1) # Red center dot
    cv2.line(img, (cx - 15, cy), (cx - 5, cy), color, 1)
    cv2.line(img, (cx + 5, cy), (cx + 15, cy), color, 1)
    cv2.line(img, (cx, cy - 15), (cx, cy - 5), color, 1)
    cv2.line(img, (cx, cy + 5), (cx, cy + 15), color, 1)
    
    # 2. Draw outer segmented arcs
    axes = (r, r)
    angle = 0
    thickness = 2
    cv2.ellipse(img, (cx, cy), axes, angle, 15, 75, color, thickness)
    cv2.ellipse(img, (cx, cy), axes, angle, 105, 165, color, thickness)
    cv2.ellipse(img, (cx, cy), axes, angle, 195, 255, color, thickness)
    cv2.ellipse(img, (cx, cy), axes, angle, 285, 345, color, thickness)
    
    # 3. Draw Corner Brackets
    line_len = int(min(w, h) * 0.2)
    line_len = max(line_len, 10)
    # Top-left
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, 2)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, 2)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, 2)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, 2)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, 2)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, 2)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, 2)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, 2)
    
    # 4. Text Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
    tx = cx - int(text_size[0]/2)
    ty = cy + r + 15
    cv2.rectangle(img, (tx - 2, ty - text_size[1] - 4), (tx + text_size[0] + 2, ty + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty - 2), font, 0.5, color, 1)

def generate_frames():
    global state
    
    import cv2
    import numpy as np
    
    if not HAS_JETSON:
        # Windows fallback using OpenCV and Ultralytics YOLOv8
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt") # lightweight model
        except ImportError:
            print("Warning: ultralytics not installed. Please run 'pip install ultralytics'.")
            while True:
                time.sleep(1)
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n'
                       b'Ultralytics YOLO not found. Run pip install ultralytics.\r\n')
                       
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        while True:
            success, img_bgr = cap.read()
            if not success:
                break
                
            results = model(img_bgr, verbose=False)
            
            person_count = 0
            forbidden_detected = False
            forbidden_name = ""
            
            # results[0].boxes contains the detections
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # YOLOv8 COCO classes: 0: person, 24: backpack, 67: cell phone
                if class_name == "person":
                    person_count += 1
                elif class_name in ["cell phone", "backpack"]:
                    forbidden_detected = True
                    forbidden_name = class_name
                    
                # Draw custom HUD box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                color = (0, 0, 255) if class_name in ["cell phone", "backpack"] else (232, 217, 5) # BGR
                label = f"{class_name.upper()} {conf:.2f}"
                draw_cyberpunk_box(img_bgr, x1, y1, x2, y2, label, color)

            # Check Rules
            if forbidden_detected:
                log_anomaly("Forbidden Object", f"Detected {forbidden_name}")
                
            if person_count > 2:
                log_anomaly("Count Violation", f"Detected {person_count} people (> 2)")

            # Update time since last anomaly
            if state["last_anomaly_time"] > 0:
                diff = int(time.time() - state["last_anomaly_time"])
                state["time_since_last"] = f"{diff} seconds ago"
            else:
                state["time_since_last"] = "N/A"
                
            # Draw a custom Cyberpunk overlay alert on the frame
            if forbidden_detected or person_count > 2:
                cv2.putText(img_bgr, "THREAT DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', img_bgr)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    else:
        # Initialize Jetson Network and Camera
        try:
            net = jetson.inference.detectNet(MODEL_NAME, threshold=0.5)
            camera = jetson.utils.videoSource(CAMERA_PATH)
        except Exception as e:
            print(f"Error initializing camera or model: {e}")
            return

        while True:
            # Capture frame
            img = camera.Capture()
            
            if img is None:
                continue
                
            # Detect objects but DO NOT draw default overlay
            detections = net.Detect(img, overlay="none")
            
            # Convert cuda image to numpy array for custom OpenCV drawing
            img_np = jetson.utils.cudaToNumpy(img)
            if img_np.dtype == np.float32:
                img_np = img_np.astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # --- Anomaly Logic & Drawing ---
            person_count = 0
            forbidden_detected = False
            forbidden_name = ""
            
            for d in detections:
                class_name = net.GetClassDesc(d.ClassID)
                if class_name == "person":
                    person_count += 1
                elif class_name in ["cell phone", "backpack"]: # Cyberpunk forbidden items
                    forbidden_detected = True
                    forbidden_name = class_name
                
                # Draw custom HUD box
                x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                conf = float(d.Confidence)
                color = (0, 0, 255) if class_name in ["cell phone", "backpack"] else (232, 217, 5) # BGR
                label = f"{class_name.upper()} {conf:.2f}"
                draw_cyberpunk_box(img_bgr, x1, y1, x2, y2, label, color)
                    
            # Check Rules
            if forbidden_detected:
                log_anomaly("Forbidden Object", f"Detected {forbidden_name}")
                
            if person_count > 2:
                log_anomaly("Count Violation", f"Detected {person_count} people (> 2)")

            # Update time since last anomaly
            if state["last_anomaly_time"] > 0:
                diff = int(time.time() - state["last_anomaly_time"])
                state["time_since_last"] = f"{diff} seconds ago"
            else:
                state["time_since_last"] = "N/A"

            # Draw a custom Cyberpunk overlay alert on the frame
            if forbidden_detected or person_count > 2:
                cv2.putText(img_bgr, "THREAT DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', img_bgr)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(state)

if __name__ == '__main__':
    init_csv()
    # Run Flask server natively
    print("Starting NCPD Threat Detection System Server...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
