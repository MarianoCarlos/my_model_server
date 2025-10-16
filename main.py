import os
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load as joblib_load
import pickle

# ========= Config =========
MODEL_PATHS = [
    os.getenv("MODEL_PATH", "model_enhanced_compressed.joblib"),
    "model_enhanced.p",  # fallback
]
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "https://www.insyncweb.site,https://insync-omega.vercel.app,http://localhost:3000"
    ).split(",")
    if o.strip()
]

# ========= Flask Setup =========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ========= Load model bundle =========
bundle = None
last_err = None
for p in MODEL_PATHS:
    try:
        if p.endswith(".joblib"):
            bundle = joblib_load(p)
        else:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
        print(f"✅ Loaded model bundle from: {p}")
        break
    except Exception as e:
        last_err = e
        print(f"⚠️ Could not load {p}: {e}")

if bundle is None:
    raise RuntimeError(f"Failed to load any model bundle. Last error: {last_err}")

model = bundle["model"]
scaler = bundle["scaler"]
encoder = bundle["encoder"]

# ========= Feature Extraction =========
def features_from_landmarks_array(hands):
    """
    hands: [{handedness: 'Left'|'Right'|..., points: [{x,y,z}, ...21]}, ...]
    Returns a 126-length feature vector (2 hands × 21 × xyz) with per-hand normalization and padding.
    """
    sample_features = []

    for hand in hands[:2]:
        pts = hand.get("points", [])
        if not pts:
            continue

        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        zs = [p["z"] for p in pts]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        scale_x = max(1e-5, x_max - x_min)
        scale_y = max(1e-5, y_max - y_min)
        scale_z = max(1e-5, z_max - z_min)

        for p in pts:
            x = (p["x"] - x_min) / scale_x
            y = (p["y"] - y_min) / scale_y
            z = (p["z"] - z_min) / scale_z
            sample_features.extend([x, y, z])

    # Pad to 126 features (for 2 hands)
    while len(sample_features) < 126:
        sample_features.append(0.0)

    return np.asarray(sample_features, dtype=np.float32)

# ========= Routes =========
@app.get("/")
def health():
    return jsonify({
        "ok": True,
        "message": "🖐 ASL Landmark Model Server running",
        "labels": list(encoder.classes_)
    })

@app.post("/predict")
def predict():
    t0 = time.time()
    try:
        data = request.get_json(force=True, silent=False) or {}
        hands = data.get("hands", [])

        if not hands:
            return jsonify({"error": "No landmark data provided"}), 400

        feats = features_from_landmarks_array(hands)
        X = scaler.transform(feats.reshape(1, -1))
        pred_idx = model.predict(X)[0]
        label = encoder.inverse_transform([pred_idx])[0]

        return jsonify({
            "label": label,
            "latency_ms": int((time.time() - t0) * 1000)
        })

    except Exception as e:
        print("❌ Prediction error:", repr(e))
        return jsonify({"error": str(e)}), 500

# ========= Entrypoint =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
