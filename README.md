# SentientAI — Emotion Detection from Facial Expressions



<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11s-FF6F00?style=for-the-badge&logo=opencv&logoColor=white)](https://docs.ultralytics.com)
[![Kaggle](https://img.shields.io/badge/Trained_on-Kaggle_GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![mAP50](https://img.shields.io/badge/mAP@50-87.3%25-22c55e?style=for-the-badge)](https://github.com/rishavm003/Emotion-Detection-from-Facial-Expressions)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)

**Real-time facial emotion detection powered by a fine-tuned YOLO11s model —  
trained from scratch on Kaggle GPU across 9 emotion classes.**

[Features](#-features) • [Demo](#-demo) • [Model Performance](#-model-performance) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure)

</div>

---

## ✨ Features

- 🎥 **Real-time webcam detection** — live bounding boxes with emotion labels and confidence scores
- ⚡ **Decoupled render/inference** — video runs at 60 FPS, inference fires independently without blocking
- 📊 **Model Results dashboard** — interactive training charts, confusion matrix, F1 curves, per-class accuracy bars
- 🌞🌙 **Light / Dark theme toggle** — preference saved across sessions via localStorage
- 📸 **Snapshot** — save the current frame with bounding boxes drawn
- 🔄 **Flip View** — mirror the webcam feed

---

## 🎬 Demo

### Live Detection Tab
The main interface starts the webcam and runs real-time emotion detection:

| Primary emotion display | Confidence history chart | Per-face detection feed |
|---|---|---|
| Large coloured label | 40-frame sparkline | Sorted confidence bars |

### Model Results Tab
A full analytics dashboard built from the Kaggle training run:

- **Training curves** — box loss, cls loss, dfl loss across all 20 epochs
- **Interactive mAP chart** — mAP@50, mAP@50-95, Precision, Recall (Chart.js)
- **Confusion matrix** — normalised per-class accuracy
- **F1-Confidence & PR curves**
- **Validation prediction samples** (pred vs ground truth side by side)
- **Training configuration table**

---

## 📈 Model Performance

Trained using **YOLO11s** (pretrained on COCO) fine-tuned for emotion detection on a labelled facial expression dataset.

| Metric | Score |
|---|---|
| **mAP@50** | **87.3%** |
| **mAP@50-95** | **69.1%** |
| **Precision** | **80.3%** |
| **Recall** | **80.5%** |

### Per-Class Accuracy (from Confusion Matrix)

| Emotion | Accuracy | Emotion | Accuracy |
|---|---|---|---|
| 😴 Sleepy | **95%** | 😲 Surprised | 80% |
| 😊 Happy | **91%** | 🤢 Disgust | 74% |
| 😠 Angry | **86%** | 😐 Natural | 69% |
| 😨 Fear | **84%** | 😒 Contempt | 67% |
| 😢 Sad | **83%** | | |

### Training Configuration

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Model | YOLO11s | Platform | Kaggle GPU |
| Epochs | 20 | Batch Size | 50 |
| Image Size | 640×640 | Optimizer | Auto (AdamW) |
| LR₀ | 0.01 | LRf | 0.01 |
| Warmup Epochs | 3 | IoU Threshold | 0.70 |
| Pretrained | Yes (COCO) | AMP | Enabled |

---

## 🛠 Installation

### Prerequisites
- Python 3.10+
- Webcam
- (Optional) NVIDIA GPU for faster inference

### 1. Clone the repository
```bash
git clone https://github.com/rishavm003/Emotion-Detection-from-Facial-Expressions.git
cd Emotion-Detection-from-Facial-Expressions
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the trained model weights

The model weights (`best.pt`) are not included in the repo due to file size.  
Download them and place at:
```
results/runs/detect/emotion_yolo11s_kaggle/weights/best.pt
```

> You can retrain the model using `train_on_kaggle.ipynb` on Kaggle, or `train_on_colab.ipynb` on Google Colab.

---

## 🚀 Usage

### Start the application
```bash
python app.py
```

Then open your browser at:
```
http://127.0.0.1:5000
```

### In the browser
1. Click **"INITIALISE NEURAL LINK"** to start the webcam
2. The model will detect faces and classify emotions in real time
3. Switch to **"Model Results"** tab to explore training analytics
4. Use the **☀️/🌙** button in the top right to toggle light/dark theme

---

## 📁 Project Structure

```
Emotion-Detection-from-Facial-Expressions/
│
├── app.py                          # Flask backend — serves UI and /detect API
├── requirements.txt                # Python dependencies
├── .gitignore                      # Excludes secrets, venv, weights
├── train_on_kaggle.ipynb           # Kaggle training notebook
├── train_on_colab.ipynb            # Google Colab training notebook
│
├── templates/
│   └── index.html                  # Full single-page UI (Live + Results tabs)
│
├── static/
│   ├── style.css                   # Light & Dark theme system
│   └── results/                    # Training output images (charts, batches)
│       ├── results.png             # All metrics curves
│       ├── confusion_matrix_normalized.png
│       ├── BoxF1_curve.png
│       ├── BoxPR_curve.png
│       ├── val_batch0_pred.jpg     # Validation predictions
│       └── ...
│
└── results/
    ├── emotion_data.yaml           # Dataset config (classes, paths)
    └── runs/detect/emotion_yolo11s_kaggle/
        ├── results.csv             # Per-epoch metrics log
        ├── args.yaml               # Full training config
        └── weights/
            ├── best.pt             # ← best model (not tracked by git)
            └── last.pt             # ← last epoch model (not tracked by git)
```

---

## 🔌 API Reference

### `POST /detect`

Accepts a base64-encoded JPEG image and returns detected emotion bounding boxes.

**Request body:**
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "detections": [
    {
      "label": "happy",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## 🏗 Tech Stack

| Layer | Technology |
|---|---|
| Model | [Ultralytics YOLO11s](https://docs.ultralytics.com) |
| Backend | [Flask](https://flask.palletsprojects.com) + Flask-CORS |
| Computer Vision | [OpenCV](https://opencv.org) + NumPy |
| Frontend | Vanilla HTML · CSS · JavaScript |
| Charts | [Chart.js](https://www.chartjs.org) |
| Training | Kaggle GPU P100 |

---

## ⚠️ Notes

- **Model weights are not included** in this repo. You must download or retrain them.
- A CUDA-compatible GPU is recommended for real-time inference. CPU runs at ~4-5 detections/sec.
- The `env` / `.env` files containing API tokens are excluded from the repo by `.gitignore`.

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/rishavm003">rishavm003</a>
</div>
