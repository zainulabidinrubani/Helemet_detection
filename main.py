import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ------------------------------
# 1. Load model correctly (custom_objects passed IN the call)
# ------------------------------
model = tf.keras.models.load_model(
    'Helemt_prediction.keras',
    custom_objects={'preprocess_input': preprocess_input}
)

IMG_HEIGHT = 224
IMG_WIDTH = 224

# ------------------------------
# 2. Load OpenCV's built-in face detector (Haar Cascade)
# ------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ------------------------------
# 3. Start webcam
# ------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # Crop a slightly larger region around the face to include head/helmet area
        pad = int(0.4 * h)
        y1 = max(0, y - pad)
        y2 = min(frame.shape[0], y + h + int(0.2 * h))
        x1 = max(0, x - int(0.2 * w))
        x2 = min(frame.shape[1], x + w + int(0.2 * w))
        head_region = frame[y1:y2, x1:x2]

        if head_region.size == 0:
            continue

        # ------------------------------
        # 4. Preprocess EXACTLY like training (no manual /255 — model does it internally)
        # ------------------------------
        resized = cv2.resize(head_region, (IMG_WIDTH, IMG_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb.astype("float32"), axis=0)  # raw [0,255], no normalization here

        # ------------------------------
        # 5. Predict — sigmoid output is a SINGLE score, not argmax
        # ------------------------------
        score = model.predict(input_data, verbose=0)[0][0]

        # Match your training convention: score < 0.5 = "With Helmet"
        if score < 0.5:
            label = f"With Helmet ({(1 - score) * 100:.1f}%)"
            color = (0, 255, 0)  # green
        else:
            label = f"No Helmet ({score * 100:.1f}%)"
            color = (0, 0, 255)  # red

        # ------------------------------
        # 6. Draw face box + label
        # ------------------------------
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Helmet Detection Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()