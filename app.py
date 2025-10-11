import os
import base64
import cv2
import numpy as np
from collections import deque, Counter
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

ENV = os.getenv("ENV", "development")
ALLOWED_ORIGINS = (
    ["https://www.insyncweb.site", "https://insync-omega.vercel.app"]
    if ENV == "production"
    else ["*"]
)

# --- Lazy import of SocketIO to save RAM ---
try:
    from flask_socketio import SocketIO, emit
    import eventlet

    socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS, async_mode="eventlet")
except Exception:
    socketio = None

# --- Lazy import of classifier (loads Mediapipe later) ---
from inference_classifier import GestureClassifier

classifier = GestureClassifier(confidence=0.6)
recent_preds = deque(maxlen=8)


def decode_upload(file):
    arr = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def home():
    return jsonify({"message": "🖐 ASL backend running."})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    frame = decode_upload(request.files["file"])
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    label, conf = classifier.predict_single(frame)
    recent_preds.append(label or "")
    smooth = Counter(recent_preds).most_common(1)[0][0]

    print(f"🖐 Prediction: {smooth} ({conf*100:.1f}%)")
    return jsonify({"prediction": smooth, "confidence": conf})


# --- Optional Socket.IO real-time prediction ---
if socketio:

    @socketio.on("video_frame")
    def handle_video(data):
        frame64 = data.get("frame")
        if not frame64:
            emit("prediction", {"error": "no frame"})
            return
        try:
            img_bytes = base64.b64decode(frame64.split(",")[1])
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            emit("prediction", {"error": "decode fail"})
            return

        label, conf = classifier.predict_single(frame)
        recent_preds.append(label or "")
        smooth = Counter(recent_preds).most_common(1)[0][0]
        emit("prediction", {"label": smooth, "confidence": conf})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = "0.0.0.0" if ENV == "production" else "127.0.0.1"
    print(f"🚀 Server running at http://{host}:{port}")
    if socketio:
        socketio.run(app, host=host, port=port)
    else:
        app.run(host=host, port=port)
