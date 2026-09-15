# Helmet Detection using CNN & Transfer Learning

A deep learning project that classifies whether a person is wearing a helmet or not, trained progressively across multiple CNN architectures, and deployed for **real-time prediction via webcam** using OpenCV.

This project was built as a hands-on deep learning practice project, focused on learning the full pipeline: data preparation, CNN training, transfer learning, fine-tuning, debugging, and real-time deployment.

---

## Demo

Real-time helmet detection running on a live webcam feed, with bounding box and confidence score overlay.

- **Green box** → "Helmet" detected
- **Red box** → "No Helmet" detected

---

## Project Journey

This project evolved through several stages of experimentation and debugging — each step meaningfully improved accuracy:

| Stage | Approach | Validation Accuracy |
|---|---|---|
| 1 | Custom CNN trained from scratch | ~70% |
| 2 | Transfer learning with **MobileNetV2** | ~86% |
| 3 | Transfer learning with **EfficientNetB0** | ~90% |
| 4 | Transfer learning with **MobileNetV3** (final) | **~95%** |

Key lessons learned along the way (documented in detail during development):
- Cropping labeled objects from bounding-box annotations, rather than using whole uncropped images, was critical for a usable training signal.
- Each pretrained backbone requires its **own matching `preprocess_input` function** — mixing them (e.g. `Rescaling(1./255)` alongside a different model's `preprocess_input`) silently destroys accuracy.
- Fine-tuning a pretrained base requires a much smaller learning rate (`1e-5`) than default (`1e-3`) to avoid destructively overwriting pretrained weights.
- Freezing the base model and training the classification head first, before unfreezing for fine-tuning, produces more stable results than fine-tuning from step one.
- `EarlyStopping` and `ReduceLROnPlateau` callbacks were used to avoid overfitting and stabilize training once validation loss plateaued.

---

## Dataset

- [Helmet Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) (Pascal VOC XML annotations, 2 classes: With Helmet / Without Helmet)
- [Helmet Detection Dataset (rishabhzen)](https://www.kaggle.com/datasets/rishabhzen/helmet-detection) (pre-sorted `helmet` / `no helmet` folders)

---

## Tech Stack

- **Python**
- **TensorFlow / Keras** — model building and training
- **OpenCV** — image preprocessing and live webcam inference
- **NumPy / Matplotlib** — data handling and visualization
- **scikit-learn** — train/test split, evaluation metrics

---

## Model Architecture

Final model uses **MobileNetV3** as a frozen/fine-tuned feature extractor, with a custom classification head:

```
Input (224x224x3)
   → preprocess_input (matched to backbone)
   → Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
   → MobileNetV3 (pretrained on ImageNet, partially fine-tuned)
   → GlobalAveragePooling2D
   → Dense(128, relu)
   → Dropout(0.3)
   → Dense(1, sigmoid)
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (`1e-3` for head training, `1e-5` for fine-tuning)
- **Callbacks:** EarlyStopping, ReduceLROnPlateau

---

## How to Run

### 1. Train the model
Open the training notebook and run all cells top to bottom (recommended: restart runtime first for a clean run). This will:
- Download the dataset
- Build the training/validation pipeline
- Train the model in two stages (frozen head → fine-tuning)
- Save the trained model as `helmet_prediction.h5`

### 2. Run live webcam prediction
```bash
pip install opencv-python tensorflow numpy
python live_prediction.py
```
Press `q` to quit the live prediction window.

---

## Future Improvements

- Expand the dataset for better generalization
- Add multi-class detection (e.g. helmet type, or full object detection instead of classification)
- Convert to a lightweight format (TFLite) for mobile/edge deployment
- Add temporal smoothing across frames for steadier live predictions

---

## Author

Built as a personal deep learning practice project — feedback and suggestions welcome.
