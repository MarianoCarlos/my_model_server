from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from inference_classifier import GestureClassifier

app = FastAPI(title="ASL Interpreter API", version="1.1")

# ✅ CORS for frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load classifier
classifier = GestureClassifier("./model_enhanced.p")

# 🧠 Stable detection memory
last_prediction = None
repeat_count = 0
STABILITY_THRESHOLD = 3  # Require 3 identical detections before confirming

@app.get("/")
def root():
    return {"message": "ASL Interpreter API is running!"}

# ✅ Predict from image upload (stable mode)
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global last_prediction, repeat_count

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result, _ = classifier.predict(img)
    if not result:
        return {"prediction": None, "message": "No hands detected."}

    # 🧠 Stable gesture filter
    if result == last_prediction:
        repeat_count += 1
        if repeat_count < STABILITY_THRESHOLD:
            return {"prediction": None, "message": "Holding same gesture"}
    else:
        last_prediction = result
        repeat_count = 0

    return {"prediction": result, "message": "New gesture detected"}

# ✅ WebSocket for real-time prediction (also stable)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_prediction, repeat_count
    await websocket.accept()
    print("🟢 WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            result, _ = classifier.predict(img)

            # 🧠 Stable gesture logic
            if not result:
                await websocket.send_json({"prediction": None, "message": "No hands detected"})
                continue

            if result == last_prediction:
                repeat_count += 1
                if repeat_count < STABILITY_THRESHOLD:
                    await websocket.send_json({"prediction": None, "message": "Holding same gesture"})
                    continue
            else:
                last_prediction = result
                repeat_count = 0

            await websocket.send_json({"prediction": result, "message": "New gesture detected"})
    except Exception as e:
        print("🔴 WebSocket disconnected:", e)
        await websocket.close()
