from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the model
try:
    # Priority 1: Kaggle-trained fine-tuned model
    model_path = 'results/runs/detect/emotion_yolo11s_kaggle/weights/best.pt'
    if not os.path.exists(model_path):
        # Priority 2: Local trained model
        model_path = 'runs/detect/emotion_yolo11s/weights/best.pt'
    if not os.path.exists(model_path):
        # Priority 3: Root best.pt
        model_path = 'best.pt'
    if not os.path.exists(model_path):
        print("Trained model not found. Using fallback pre-trained yolo11s.pt")
        model_path = 'yolo11s.pt'
    
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference
        results = model(img)

        # Process results
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                detections.append({
                    'bbox': coords,
                    'label': label,
                    'confidence': conf
                })

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
