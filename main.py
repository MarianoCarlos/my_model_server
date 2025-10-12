from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from inference_classifier import GestureClassifier

app = FastAPI(title="ASL Interpreter API", version="1.2")

# ✅ Enable CORS for frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.insyncweb.site","https://insync-omega.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load classifier
classifier = GestureClassifier("./model_enhanced.p")

# 🧠 Global memory for stability tracking
last_prediction = None
repeat_count = 0
STABILITY_THRESHOLD = 3  # Require 3 identical frames before confirming


@app.get("/")
def root():
    return {"message": "ASL Interpreter API is running!"}


# ✅ Predict from image upload (stable mode)
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    global last_prediction, repeat_count

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    result, _ = classifier.predict(img)
    if not result:
        # Reset repeat count when no hands
        repeat_count = 0
        last_prediction = None
        return {"prediction": None, "message": "No hands detected."}

    # 🧩 Stability logic
    if result == last_prediction:
        repeat_count += 1
    else:
        last_prediction = result
        repeat_count = 1  # new gesture starts counting

    # Return only after N stable detections
    if repeat_count >= STABILITY_THRESHOLD:
        return {"prediction": result, "message": "Stable gesture detected"}
    else:
        return {"prediction": None, "message": "Waiting for stability"}


# ✅ WebSocket endpoint (optional real-time use)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_prediction, repeat_count
    await websocket.accept()
    print("🟢 WebSocket client connected")

    try:
        while True:
            # Receive Base64-encoded frame
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run inference
            result, _ = classifier.predict(img)

            if not result:
                repeat_count = 0
                last_prediction = None
                await websocket.send_json({"prediction": None, "message": "No hands detected"})
                continue

            if result == last_prediction:
                repeat_count += 1
            else:
                last_prediction = result
                repeat_count = 1

            if repeat_count >= STABILITY_THRESHOLD:
                await websocket.send_json({"prediction": result, "message": "Stable gesture detected"})
            else:
                await websocket.send_json({"prediction": None, "message": "Waiting for stability"})

    except Exception as e:
        print("🔴 WebSocket disconnected:", e)
        await websocket.close()
