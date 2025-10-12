import pickle
import cv2
import mediapipe as mp
import numpy as np

class GestureClassifier:
    def __init__(self, model_path="./model_enhanced.p"):
        # ✅ Load trained model bundle
        print("📦 Loading model:", model_path)
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)

        self.model = model_bundle["model"]
        self.scaler = model_bundle["scaler"]
        self.encoder = model_bundle["encoder"]

        # ✅ Setup MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("✅ Model, scaler, and encoder loaded successfully!")

    def predict(self, frame):
        """
        Processes a frame and predicts the ASL gesture.
        Returns: (predicted_word, annotated_frame)
        """
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        predicted_word = None

        if results.multi_hand_landmarks:
            # Draw hands
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

            # Sort by handedness for consistency
            if results.multi_handedness:
                handedness = [h.classification[0].label for h in results.multi_handedness]
                pairs = sorted(zip(handedness, results.multi_hand_landmarks), key=lambda x: x[0])
            else:
                pairs = [("Unknown", results.multi_hand_landmarks[0])]

            sample_features = []
            x_, y_ = [], []

            for label, hand_landmarks in pairs:
                xs, ys, zs = [], [], []
                for lm in hand_landmarks.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
                    zs.append(lm.z)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                z_min, z_max = min(zs), max(zs)

                scale_x = max(1e-5, x_max - x_min)
                scale_y = max(1e-5, y_max - y_min)
                scale_z = max(1e-5, z_max - z_min)

                for lm in hand_landmarks.landmark:
                    x = (lm.x - x_min) / scale_x
                    y = (lm.y - y_min) / scale_y
                    z = (lm.z - z_min) / scale_z
                    sample_features.extend([x, y, z])

                x_.extend(xs)
                y_.extend(ys)

            # Pad for consistent length (2 hands = 126 features)
            while len(sample_features) < 126:
                sample_features.append(0.0)

            # ✅ Normalize with the same scaler used in training
            sample_features = np.asarray(sample_features).reshape(1, -1)
            sample_scaled = self.scaler.transform(sample_features)

            # ✅ Predict gesture
            prediction = self.model.predict(sample_scaled)
            predicted_word = self.encoder.inverse_transform(prediction)[0]

            # Draw box and prediction
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                predicted_word.upper(),
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        return predicted_word, frame
