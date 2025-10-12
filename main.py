from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import traceback
from inference_classifier import GestureClassifier

# -------------------------------------------------------------
# 🚀 FASTAPI APP CONFIG
# -------------------------------------------------------------
app = FastAPI(title="ASL Interpreter API", version="1.4")

# ✅ Allow frontend origins
origins = [
    "https://www.insyncweb.site",
    "https://insync-omega.vercel.app",
    "https://insyncweb.site",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 🧠 LOAD CLASSIFIER MODEL
# -------------------------------------------------------------
try:
    classifier = GestureClassifier("./model_enhanced.p")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    classifier = None

# -------------------------------------------------------------
# ⚙️ GLOBAL VARIABLES
# -------------------------------------------------------------
last_prediction = None
repeat_count = 0
STABILITY_THRESHOLD = 3  # Require 3 identical frames before confirming


# -------------------------------------------------------------
# 🩺 HEALTH CHECK
# -------------------------------------------------------------
@app.get("/health")
def health_check():
    """Check if the API and model are loaded correctly."""
    if classifier is None:
        return {"status": "error", "message": "Model not loaded"}
    try:
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = classifier.predict(test_img)
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------
# 🏠 ROOT ROUTE
# -------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "ASL Interpreter API is running!"}


# -------------------------------------------------------------
# 🧠 IMAGE PREDICTION ENDPOINT
# -------------------------------------------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Handles single-frame image inference for ASL gestures."""
    global last_prediction, repeat_count

    if classifier is None:
        return {"error": "Model not loaded", "message": "Server error"}

    try:
        # Read and decode uploaded frame
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid or unreadable image input")

        # Run inference
        result, _ = classifier.predict(img)

        # Handle no-hand detection
        if not result:
            repeat_count = 0
            last_prediction = None
            return {"prediction": None, "message": "No hands detected."}

        # Apply stability logic
        if result == last_prediction:
            repeat_count += 1
        else:
            last_prediction = result
            repeat_count = 1

        # Return stable detection result
        if repeat_count >= STABILITY_THRESHOLD:
            return {"prediction": result, "message": "Stable gesture detected"}
        else:
            return {"prediction": None, "message": "Waiting for stability"}

    except Exception as e:
        print("❌ Error during /predict:", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Internal Server Error"},
        )


# -------------------------------------------------------------
# ⚠️ GLOBAL EXCEPTION HANDLER
# -------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("⚠️ Global error caught:", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "message": "Unexpected server error"},
    )
