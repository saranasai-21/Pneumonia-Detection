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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

IMG_SIZE = 256

# =====================================================
# HELPERS
# =====================================================

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def image_to_base64_bgr(image_bgr):
    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        return ""
    return base64.b64encode(encoded).decode("utf-8")

def make_pseudocolor_image(rgb_array):
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def create_lightweight_region_image(original_bgr, result):
    """
    Lightweight visual highlight:
    - Creates a pseudo-heatmap style image
    - For pneumonia, draws a rough red suspicious area
    - For normal, returns only the colored image
    """
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

    if result == "PNEUMONIA":
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                center = (int(x), int(y))
                radius = int(radius)

                cv2.circle(overlay, center, radius, (0, 0, 255), 4)
                cv2.circle(overlay, center, 6, (0, 0, 255), -1)

                x1, y1, w, h = cv2.boundingRect(largest)
                cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    return overlay

# =====================================================
# ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        rgb_array = np.array(image)
        original_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        processed_image = preprocess_image(image)
        prediction = float(model.predict(processed_image, verbose=0)[0][0])

        if prediction > 0.6:
            result = "PNEUMONIA"
            confidence = prediction
        else:
            result = "NORMAL"
            confidence = 1 - prediction

        # Pseudo-colored image for display
        colored_bgr = make_pseudocolor_image(rgb_array)

        # Lightweight region visualization
        highlighted_bgr = create_lightweight_region_image(original_bgr, result)

        original_base64 = image_to_base64_bgr(original_bgr)
        colored_base64 = image_to_base64_bgr(colored_bgr)
        highlighted_base64 = image_to_base64_bgr(highlighted_bgr)

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": round(confidence * 100, 2),
                "original_image": original_base64,
                "colored_image": colored_base64,
                "highlighted_image": highlighted_base64
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})
