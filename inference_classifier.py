import gzip
import joblib
import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter


class GestureClassifier:
    def __init__(self):
        # === Load compressed model ===
        print("📦 Loading compressed model...")
        with gzip.open("./model_compressed.p.gz", "rb") as f:
            model_dict = joblib.load(f)

        self.model = model_dict["model"]
        self.scaler = model_dict.get("scaler", None)
        self.encoder = model_dict.get("encoder", None)
        print("✅ Model loaded successfully from model_compressed.p.gz")

        # === Initialize MediaPipe Hands ===
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # For temporal smoothing
        self.recent_preds = deque(maxlen=5)

    def preprocess_landmarks(self, hand_landmarks):
        """Normalize x, y, z coordinates for one hand."""
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        zs = [lm.z for lm in hand_landmarks.landmark]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        norm = []
        for x, y, z in zip(xs, ys, zs):
            norm.extend([
                (x - min_x) / (max_x - min_x + 1e-6),
                (y - min_y) / (max_y - min_y + 1e-6),
                (z - min_z) / (max_z - min_z + 1e-6),
            ])
        return norm

    def predict(self, frame):
        """Run inference on a frame and return smoothed prediction."""
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        predicted_label = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            data_aux = []

            # ✅ Limit to 2 hands max (matches training)
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                # Extract normalized xyz
                data_aux.extend(self.preprocess_landmarks(hand_landmarks))

            # ✅ Ensure fixed feature length (126)
            if len(data_aux) < 126:
                data_aux += [0] * (126 - len(data_aux))
            elif len(data_aux) > 126:
                data_aux = data_aux[:126]

            # Apply scaler if available
            features = np.asarray(data_aux).reshape(1, -1)
            if self.scaler:
                features = self.scaler.transform(features)

            # Predict class & confidence
            probs = self.model.predict_proba(features)[0]
            confidence = np.max(probs)
            pred_idx = np.argmax(probs)

            # Decode label
            if self.encoder:
                predicted_label = self.encoder.inverse_transform([pred_idx])[0]
            else:
                predicted_label = pred_idx

            # Smoothing
            self.recent_preds.append(predicted_label)
            smooth_label = Counter(self.recent_preds).most_common(1)[0][0]

            # Draw result
            color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
            cv2.putText(
                frame,
                f"{smooth_label} ({confidence*100:.1f}%)",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
                cv2.LINE_AA,
            )

            return smooth_label, frame

        return None, frame
