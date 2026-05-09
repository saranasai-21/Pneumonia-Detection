from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import cv2
import base64

from tensorflow.keras.applications.densenet import preprocess_input

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia_densenet.keras")

# =====================================================
# STATIC + TEMPLATE PATHS
# =====================================================

STATIC_DIR = os.path.join(BASE_DIR, "api", "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "api", "templates")

# =====================================================
# DEBUG LOGS
# =====================================================

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))
print("STATIC_DIR EXISTS:", os.path.exists(STATIC_DIR))
print("TEMPLATE_DIR EXISTS:", os.path.exists(TEMPLATE_DIR))

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(title="Pneumonia Detection System")

# =====================================================
# MOUNT STATIC FILES
# =====================================================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =====================================================
# TEMPLATES
# =====================================================

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMG_SIZE = 256

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def image_to_base64_bgr(image_bgr):
    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        return ""
    return base64.b64encode(encoded).decode("utf-8")

# =====================================================
# HOME PAGE
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Original RGB image as array
        rgb_array = np.array(image)

        # Preprocess for model
        processed_image = preprocess_image(image)

        prediction = float(model.predict(processed_image, verbose=0)[0][0])

        if prediction > 0.5:
            result = "PNEUMONIA"
            confidence = prediction
        else:
            result = "NORMAL"
            confidence = 1 - prediction

        # Convert RGB -> BGR for OpenCV drawing
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        # Very simple region highlighting for visual effect
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        highlighted = bgr_image.copy()

        found_region = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(highlighted, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.circle(highlighted, (x + w // 2, y + h // 2), max(w, h) // 2, (0, 0, 255), 2)
                found_region = True

        # If no region found, show a subtle label on the image
        if not found_region:
            cv2.putText(
                highlighted,
                "No clear region detected",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        rgb_image_base64 = image_to_base64_bgr(bgr_image)
        highlighted_base64 = image_to_base64_bgr(highlighted)

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": round(confidence * 100, 2),
                "rgb_image": rgb_image_base64,
                "highlighted_image": highlighted_base64
            }
        )

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# =====================================================
# FAVICON FIX
# =====================================================

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})
