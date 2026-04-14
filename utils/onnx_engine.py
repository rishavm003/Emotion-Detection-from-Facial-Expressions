import onnxruntime as ort
import cv2
import numpy as np

class YOLO11ONNX:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load the model and create the session
        # We use CPU execution provider for serverless compatibility
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.width = self.input_shape[2]
        self.height = self.input_shape[3]
        
        # Hardcoded classes for SentientAI (based on emotion_data.yaml)
        self.names = [
            "angry", "contempt", "disgust", "fear", "happy", 
            "natural", "sad", "sleepy", "surprised"
        ]

    def preprocess(self, img):
        """Prepare image for YOLO11 ONNX."""
        img_h, img_w = img.shape[:2]
        
        # Resize to 640x640 with padding (letterbox)
        # Pad to square
        max_size = max(img_h, img_w)
        canvas = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        canvas[:img_h, :img_w] = img
        
        # Resize to model input size
        resized = cv2.resize(canvas, (self.width, self.height))
        
        # Convert to float and normalize
        input_data = resized.astype(np.float32) / 255.0
        
        # HWC to CHW
        input_data = input_data.transpose(2, 0, 1)
        
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data, (img_h, img_w, max_size)

    def postprocess(self, outputs, meta):
        """Convert raw ONNX output to detection results."""
        img_h, img_w, max_size = meta
        
        # Raw output is [1, 13, 8400]
        # Transpose to [1, 8400, 13]
        data = outputs[0].transpose(0, 2, 1)[0]
        
        boxes = []
        scores = []
        class_ids = []
        
        scale = max_size / self.width
        
        # YOLO11 output: [x_center, y_center, width, height, class0...class8]
        for row in data:
            # Score is the maximum class probability
            classes_scores = row[4:]
            max_score = np.max(classes_scores)
            
            if max_score >= self.conf_threshold:
                # Box coordinates are in local 640x640 space
                x_center, y_center, w, h = row[:4]
                
                # Convert to x1, y1, w, h for cv2.dnn.NMSBoxes
                x1 = (x_center - w / 2) * scale
                y1 = (y_center - h / 2) * scale
                w_scaled = w * scale
                h_scaled = h * scale
                
                boxes.append([float(x1), float(y1), float(w_scaled), float(h_scaled)])
                scores.append(float(max_score))
                class_ids.append(int(np.argmax(classes_scores)))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, 
            score_threshold=self.conf_threshold, 
            nms_threshold=self.iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes[i]
                detections.append({
                    'bbox': [x1, y1, x1 + w, y1 + h],
                    'label': self.names[class_ids[i]],
                    'confidence': scores[i]
                })
        
        return detections

    def __call__(self, img):
        """Full detection pipeline."""
        input_data, meta = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_data})
        return self.postprocess(outputs, meta)
