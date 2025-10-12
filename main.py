from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from inference_classifier import GestureClassifier
import base64

app = FastAPI(title="ASL Interpreter API", version="1.0")

# ✅ CORS for frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = GestureClassifier("./model_enhanced.p")

@app.get("/")
def root():
    return {"message": "ASL Interpreter API is running!"}

# ✅ Predict from image upload
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result, _ = classifier.predict(img)
    if not result:
        return {"prediction": None, "message": "No hands detected."}
    return {"prediction": result}

# ✅ Optional WebSocket endpoint (for real-time streaming)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            result, _ = classifier.predict(img)
            await websocket.send_json({"prediction": result})
    except Exception as e:
        print("🔴 WebSocket disconnected:", e)
        await websocket.close()
