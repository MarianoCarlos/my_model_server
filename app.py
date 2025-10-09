import os
import base64
import cv2
import numpy as np
from collections import deque, Counter
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from inference_classifier import GestureClassifier

# === Flask Setup ===
app = Flask(__name__)
CORS(app)  # ✅ allows both localhost and production frontend connections

# Environment detection
ENV = os.getenv("ENV", "development")

if ENV == "production":
    allowed_origins = [
        "https://www.insyncweb.site", "https://insync-omega.vercel.app",  # 🔹 replace with your actual Vercel domain
    ]
else:
    allowed_origins = ["*"]

socketio = SocketIO(app, cors_allowed_origins=allowed_origins, async_mode="eventlet")

# === Load Model ===
classifier = GestureClassifier()

# 🔹 Fine-tune Mediapipe Hands for better accuracy
classifier.hands = classifier.mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,  # slightly higher than default
    min_tracking_confidence=0.6,
)

print("✅ GestureClassifier model loaded successfully!")

# === Buffer for smoothing predictions ===
recent_preds = deque(maxlen=8)

# === Helper: Decode file (from FormData) ===
def decode_frame_from_file(file):
    arr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

# === Helper: Decode Base64 frame (from socket) ===
def decode_frame_base64(img_base64):
    try:
        img_bytes = base64.b64decode(img_base64.split(",")[1])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("⚠️ Frame decode error:", e)
        return None

# === Root Route ===
@app.route("/")
def home():
    return jsonify({"message": "🖐 ASL Gesture Recognition Server is running!"})

# === REST Endpoint: /predict ===
@app.route("/predict", methods=["POST"])
def predict():
    """Predict ASL gesture from uploaded frame"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    frame = decode_frame_from_file(file)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # === Extract hand landmarks (same preprocessing as training) ===
    results = classifier.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return jsonify({"prediction": None, "confidence": 0})

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks[:2]:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        zs = [lm.z for lm in hand_landmarks.landmark]

        # Normalize
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        for x, y, z in zip(xs, ys, zs):
            data_aux.extend([
                (x - min_x) / (max_x - min_x + 1e-6),
                (y - min_y) / (max_y - min_y + 1e-6),
                (z - min_z) / (max_z - min_z + 1e-6)
            ])

    # Pad missing hands
    if len(data_aux) < 126:
        data_aux += [0] * (126 - len(data_aux))

    # === Predict ===
    features = np.asarray(data_aux).reshape(1, -1)
    if classifier.scaler:
        features = classifier.scaler.transform(features)

    probs = classifier.model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    confidence = float(np.max(probs))
    pred_label = classifier.encoder.inverse_transform([pred_idx])[0]

    # === Apply smoothing ===
    recent_preds.append(pred_label)
    smooth_label = Counter(recent_preds).most_common(1)[0][0]

    # 🟢 Log for debugging
    print(f"🖐 Prediction: {smooth_label} ({confidence*100:.1f}%)")

    return jsonify({
        "prediction": smooth_label,
        "confidence": confidence
    })

# === Socket.IO: Real-time Frame Handling ===
@socketio.on("video_frame")
def handle_video_frame(data):
    frame_b64 = data.get("frame")
    if not frame_b64:
        emit("prediction", {"error": "No frame"})
        return

    frame = decode_frame_base64(frame_b64)
    if frame is None:
        emit("prediction", {"error": "Invalid frame"})
        return

    # Same preprocessing logic
    results = classifier.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        emit("prediction", {"label": None, "confidence": 0})
        return

    data_aux = []
    for hand_landmarks in results.multi_hand_landmarks[:2]:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        zs = [lm.z for lm in hand_landmarks.landmark]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        for x, y, z in zip(xs, ys, zs):
            data_aux.extend([
                (x - min_x) / (max_x - min_x + 1e-6),
                (y - min_y) / (max_y - min_y + 1e-6),
                (z - min_z) / (max_z - min_z + 1e-6)
            ])

    if len(data_aux) < 126:
        data_aux += [0] * (126 - len(data_aux))

    features = np.asarray(data_aux).reshape(1, -1)
    if classifier.scaler:
        features = classifier.scaler.transform(features)

    probs = classifier.model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    confidence = float(np.max(probs))
    pred_label = classifier.encoder.inverse_transform([pred_idx])[0]

    recent_preds.append(pred_label)
    smooth_label = Counter(recent_preds).most_common(1)[0][0]

    print(f"🖐 [Socket] Prediction: {smooth_label} ({confidence*100:.1f}%)")

    emit("prediction", {"label": smooth_label, "confidence": confidence})

# === Run Server ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if ENV == "production" else "127.0.0.1"
    print(f"🚀 Server running at http://{host}:{port}")
    socketio.run(app, host=host, port=port)
