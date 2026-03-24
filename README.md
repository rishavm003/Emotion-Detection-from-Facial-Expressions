# 😊 Emotion Detection from Facial Expressions

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning-based system that detects and classifies human emotions in real-time from facial expressions using Convolutional Neural Networks (CNN).
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Emotions Detected](#-emotions-detected)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

This project implements a real-time **Facial Emotion Recognition (FER)** system using deep learning. It captures video from a webcam (or static images), detects faces using OpenCV's Haar Cascade or MTCNN, and classifies the detected face into one of 7 universal emotion categories using a trained CNN model.

The model is trained on the **FER-2013** dataset and achieves competitive accuracy on the test set.

---

## 🎥 Demo

```
Input: Webcam / Image / Video
  └─► Face Detection (OpenCV / MTCNN)
        └─► Preprocessing (Grayscale, Resize to 48x48)
              └─► CNN Model Inference
                    └─► Emotion Label Displayed on Frame
```

> **Example Output:** Bounding box around the detected face with the predicted emotion label (e.g., `Happy: 94.3%`) overlaid on the video feed.

---

## ✨ Features

- 🔴 **Real-time detection** via webcam feed
- 🖼️ **Static image** emotion analysis
- 🎬 **Video file** processing support
- 🧩 **Multi-face detection** in a single frame
- 📊 **Confidence scores** displayed for each prediction
- 💾 **Pre-trained model** included for quick inference
- 🧪 **Training pipeline** to retrain on custom data

---

## 😄 Emotions Detected

| Label | Emotion   |
|-------|-----------|
| 0     | Angry     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happy     |
| 4     | Neutral   |
| 5     | Sad       |
| 6     | Surprise  |

---

## 🛠️ Tech Stack

| Component         | Technology                        |
|------------------|-----------------------------------|
| Language          | Python 3.8+                      |
| Deep Learning     | TensorFlow / Keras                |
| Computer Vision   | OpenCV                           |
| Face Detection    | Haar Cascade / MTCNN             |
| Data Handling     | NumPy, Pandas                    |
| Visualization     | Matplotlib, Seaborn              |
| Environment       | Jupyter Notebook / Python Script |

---

## 📁 Project Structure

```
Emotion-Detection-from-Facial-Expressions/
│
├── dataset/                    # Dataset directory
│   ├── train/                  # Training images (organized by emotion)
│   └── test/                   # Testing images (organized by emotion)
│
├── models/                     # Saved model files
│   ├── emotion_model.h5        # Pre-trained Keras model
│   └── emotion_model.json      # Model architecture
│
├── haarcascades/               # OpenCV face detection classifiers
│   └── haarcascade_frontalface_default.xml
│
├── notebooks/                  # Jupyter notebooks
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── Model_Training.ipynb    # Training pipeline
│
├── src/                        # Source code
│   ├── train.py                # Model training script
│   ├── predict.py              # Inference on images/video
│   ├── realtime_detection.py   # Real-time webcam detection
│   └── utils.py                # Helper functions
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # License file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip
- A webcam (for real-time detection)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Emotion-Detection-from-Facial-Expressions.git
cd Emotion-Detection-from-Facial-Expressions
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the pre-trained model** *(if not included)*
```bash
# Place emotion_model.h5 inside the models/ directory
```

---

## 🚀 Usage

### 🔴 Real-Time Webcam Detection
```bash
python src/realtime_detection.py
```
Press `Q` to quit the webcam window.

---

### 🖼️ Predict on a Single Image
```bash
python src/predict.py --image path/to/image.jpg
```

---

### 🎬 Predict on a Video File
```bash
python src/predict.py --video path/to/video.mp4
```

---

### 🏋️ Train the Model from Scratch
```bash
python src/train.py --data_dir dataset/ --epochs 50 --batch_size 64
```

Optional arguments:

| Argument        | Default | Description                        |
|----------------|---------|------------------------------------|
| `--epochs`      | 50      | Number of training epochs          |
| `--batch_size`  | 64      | Batch size for training            |
| `--lr`          | 0.001   | Learning rate                      |
| `--data_dir`    | dataset/| Path to dataset directory          |
| `--save_path`   | models/ | Directory to save trained model    |

---

## 📦 Dataset

This project uses the **FER-2013** dataset, which contains **35,887 grayscale images** of size **48×48 pixels**, each labeled with one of 7 emotion categories.

### Download

You can download it from [Kaggle — FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).

After downloading, place it in the `dataset/` directory following this structure:

```
dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── ...
    └── surprise/
```

---

## 🧱 Model Architecture

The CNN model is designed with the following layers:

```
Input (48x48x1)
    │
    ▼
Conv2D (32 filters, 3x3) + BatchNorm + ReLU
    │
    ▼
Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
    │
    ▼
Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
    │
    ▼
Conv2D (256 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
    │
    ▼
Flatten
    │
    ▼
Dense (512) + ReLU + Dropout(0.5)
    │
    ▼
Dense (7) + Softmax
```

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Regularization:** Batch Normalization + Dropout

---

## 📊 Results

| Metric              | Value     |
|--------------------|-----------|
| Training Accuracy   | ~95%      |
| Validation Accuracy | ~66%      |
| Test Accuracy       | ~65%      |

> **Note:** FER-2013 is a challenging dataset with inherent label noise. Results may vary based on training configuration.

### Confusion Matrix

> *(Add your confusion matrix image here)*
```
![Confusion Matrix](results/confusion_matrix.png)
```

### Training Curves

> *(Add your training/validation accuracy and loss curves here)*
```
![Training Curves](results/training_curves.png)
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. **Fork** the repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a **Pull Request**

Please make sure your code follows PEP 8 style guidelines and includes relevant docstrings/comments.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FER-2013 Dataset — Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow / Keras Documentation](https://www.tensorflow.org/)
- Research inspiration: *"Challenges in Representation Learning: A report on three machine learning contests"* — Goodfellow et al.

---

<p align="center">
  Made with ❤️ | If you found this useful, consider giving it a ⭐
</p>
