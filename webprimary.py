from flask import Flask, render_template, Response, jsonify, request
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import threading
import json
import re
import uuid
import os
from datetime import datetime

app = Flask(__name__)

# Hardcoded paths for the model and disease data
MODEL_PATH = "models/yolov8n.pt"
DISEASE_DATA_PATH = "disease_info/data.json"
CURE_PATH = "disease_info/cure.json"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load disease data
with open(DISEASE_DATA_PATH, "r") as f:
    disease_data = json.load(f)

with open(CURE_PATH, "r") as c:
    cure_data = json.load(c)

# Normalized disease names
disease_names = {re.sub(r'\s+', ' ', key.lower().strip()) for key in disease_data.keys()}

# Camera setup
cap = None
cap_lock = threading.Lock()
current_camera_index = 0
live_feed_frame = None
latest_processed_image_path = None
latest_detection_info = []


def init_camera(camera_index):
    """Initialize the camera with retry logic."""
    global cap
    if cap is not None:
        cap.release()
    for _ in range(3):  # Retry up to 3 times
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return True
    return False


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')


@app.route('/live_feed')
def live_feed():
    """Stream live video feed."""
    return Response(stream_live_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


def stream_live_feed():
    """Generate frames for the live feed."""
    global cap, live_feed_frame
    if not init_camera(current_camera_index):
        raise RuntimeError("Cannot open camera.")
    while True:
        with cap_lock:
            ret, frame = cap.read()
            if not ret:
                break
            live_feed_frame = frame.copy()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


@app.route('/capture', methods=['POST'])
def capture():
    """Capture a frame from the live feed and perform detection."""
    global live_feed_frame, latest_processed_image_path, latest_detection_info
    if live_feed_frame is not None:
        with cap_lock:
            captured_image = live_feed_frame.copy()

        # Generate a random filename for the captured image
        filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join("static", filename)
        cv2.imwrite(save_path, captured_image)

        # Convert BGR to RGB (YOLO expects RGB format)
        image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)

        # Perform detection with a lower confidence threshold
        results = model(image_rgb, conf=0.3)

        detected_diseases = []
        for result in results:
            if hasattr(result, 'boxes') and hasattr(result, 'names'):
                for box in result.boxes:
                    disease_label = result.names[int(box.cls)]
                    if re.sub(r'\s+', ' ', disease_label.lower().strip()) in disease_names:
                        # Draw bounding boxes and labels
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
                        cv2.rectangle(captured_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(captured_image, disease_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 2)
                        print("OK")
                        detected_diseases.append({
                            "disease": disease_label,
                            "coordinates": [x1, y1, x2, y2],
                            "info": disease_data.get(disease_label.lower(), "No data available"),
                            "cure": cure_data.get(disease_label.lower(), "No available data")
                        })

        # Save the processed image with detections
        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        processed_path = os.path.join("static", processed_filename)
        cv2.imwrite(processed_path, captured_image)

        # Update latest processed image and detection info
        latest_processed_image_path = processed_path
        latest_detection_info = detected_diseases

        return jsonify({
            "status": "Image captured and processed successfully.",
            "original_path": f"/static/{filename}",
            "processed_path": f"/static/{processed_filename}"
        })

    return jsonify({"error": "No live feed available"}), 400


@app.route('/latest_image_info', methods=['GET'])
def latest_image_info():
    """Return the latest processed image and detection results."""
    global latest_processed_image_path, latest_detection_info
    if latest_processed_image_path is None:
        return jsonify({"status": "error", "message": "No latest image available."})

    print(latest_detection_info)
    return jsonify({
        "status": "success",
        "image_path": f"/static/{os.path.basename(latest_processed_image_path)}",
        "detections": latest_detection_info
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
