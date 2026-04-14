import os
import sys

print("--- Starting SentientAI Web Server...", flush=True)
print("--- Loading core libraries (this may take 15-30 seconds)...", flush=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    from utils.onnx_engine import YOLO11ONNX
    import cv2
    import numpy as np
    import base64
except ImportError as e:
    print(f"--- Failed to load libraries: {e}", flush=True)
    sys.exit(1)

print("--- Libraries loaded.", flush=True)

app = Flask(__name__)
CORS(app)

# Load the model
try:
    model_path = 'model/best.onnx'
    if not os.path.exists(model_path):
        # Fallback to current directory for edge cases
        model_path = 'best.onnx'
    
    print(f"--- Loading YOLO model (ONNX) from {model_path} ...", flush=True)
    model = YOLO11ONNX(model_path)
    print(f"--- Model successfully loaded!", flush=True)
except Exception as e:
    print(f"--- Error loading model: {e}", flush=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluation-report')
def evaluation_report():
    # Return the generated evaluation report
    report_path = 'evaluation/evaluation_report.html'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Report not generated yet. Please run evaluation script.", 404

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

        # Run inference using the new ONNX engine
        detections = model(img)

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5001)
