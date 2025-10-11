import os, gzip, joblib, cv2, numpy as np
from collections import deque, Counter

# Disable matplotlib backend and font cache to save memory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["MPLBACKEND"] = "Agg"


class GestureClassifier:
    def __init__(self, confidence=0.6):
        self.conf_threshold = confidence
        self.model = self.scaler = self.encoder = None
        self.hands = None
        self.lazy_loaded = False
        self._load_model()

    def _load_model(self):
        print("📦 Loading compressed model...")
        with gzip.open("./model_compressed.p.gz", "rb") as f:
            model_dict = joblib.load(f)
        self.model = model_dict["model"]
        self.scaler = model_dict.get("scaler")
        self.encoder = model_dict.get("encoder")
        print("✅ Model loaded successfully.")

    def _load_mediapipe(self):
        if not self.lazy_loaded:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.conf_threshold,
                min_tracking_confidence=self.conf_threshold,
            )
            self.lazy_loaded = True
            print("🤖 Mediapipe Hands loaded (lazy).")

    def preprocess(self, hand_lm):
        xs = [lm.x for lm in hand_lm.landmark]
        ys = [lm.y for lm in hand_lm.landmark]
        zs = [lm.z for lm in hand_lm.landmark]
        mnx, mxx = min(xs), max(xs)
        mny, mxy = min(ys), max(ys)
        mnz, mxz = min(zs), max(zs)
        norm = []
        for x, y, z in zip(xs, ys, zs):
            norm += [
                (x - mnx) / (mxx - mnx + 1e-6),
                (y - mny) / (mxy - mny + 1e-6),
                (z - mnz) / (mxz - mnz + 1e-6),
            ]
        return norm

    def predict_single(self, frame):
        self._load_mediapipe()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None, 0.0

        data = []
        for hand in res.multi_hand_landmarks[:2]:
            data += self.preprocess(hand)
        if len(data) < 126:
            data += [0] * (126 - len(data))

        feat = np.asarray(data).reshape(1, -1)
        if self.scaler:
            feat = self.scaler.transform(feat)

        probs = self.model.predict_proba(feat)[0]
        idx = np.argmax(probs)
        conf = float(np.max(probs))
        label = self.encoder.inverse_transform([idx])[0]
        return label, conf
