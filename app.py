from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(".final_hand_gesture_model.keras")

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

IMG_SIZE = 300


def decode_label(label_idx):
    labels = [chr(i) for i in range(65, 91)] + ["SPC"]
    return labels[label_idx] if 0 <= label_idx < len(labels) else "Unknown"


def preprocess_image(image, mean=0.5, std=0.5, threshold=10):
    if image.ndim == 3:
        image = image.mean(axis=-1)
    mask = image > threshold
    masked = np.where(mask, image, 0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)
    enhanced = enhanced.astype(np.float32) / 255.0
    enhanced = (enhanced - mean) / (std + 1e-7)
    rgb_image = np.stack([enhanced] * 3, axis=-1)
    return rgb_image


def segment_and_crop_hand(image, box_size=IMG_SIZE, padding=20):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(
        image, cv2.COLOR_RGB2BGR
    )  # PIL loads RGB, MediaPipe expects RGB
    results = hands.process(image)
    if not results.multi_hand_landmarks:
        return None, None, None

    landmarks = results.multi_hand_landmarks[0]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
    cx, cy = np.mean(points, axis=0).astype(int)

    hand_w = max(p[0] for p in points) - min(p[0] for p in points)
    hand_h = max(p[1] for p in points) - min(p[1] for p in points)
    scale = 1.5
    half_w = int(hand_w * scale / 2)
    half_h = int(hand_h * scale / 2)

    x1 = max(cx - half_w - padding, 0)
    y1 = max(cy - half_h - padding, 0)
    x2 = min(cx + half_w + padding, w)
    y2 = min(cy + half_h + padding, h)

    expanded = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        dist = np.hypot(dx, dy)
        factor = (dist + padding) / dist if dist != 0 else 1
        ex, ey = int(cx + dx * factor), int(cy + dy * factor)
        expanded.append([ex, ey])

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(expanded))
    cv2.drawContours(mask, [hull], -1, 255, -1)

    hand_only = cv2.bitwise_and(image, image, mask=mask)
    cropped = hand_only[y1:y2, x1:x2]

    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None, None, None

    resized = cv2.resize(cropped, (box_size, box_size))

    landmark_list = []
    for lm in landmarks.landmark:
        landmark_list.extend([lm.x, lm.y, lm.z])

    return resized, landmark_list, (x1, y1, x2, y2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image (data:image/jpeg;base64,...)
    header, encoded = img_b64.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    cropped_img, landmarks, _ = segment_and_crop_hand(img_np)
    if cropped_img is None:
        return jsonify({"label": "No hand detected", "confidence": 0})

    processed = preprocess_image(cropped_img)
    landmarks = np.array(landmarks).flatten()
    preds = model.predict(
        [np.expand_dims(processed, axis=0), np.expand_dims(landmarks, axis=0)]
    )
    label_idx = np.argmax(preds)
    label = decode_label(label_idx)
    confidence = float(np.max(preds))

    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=("cert.pem", "key.pem"))
