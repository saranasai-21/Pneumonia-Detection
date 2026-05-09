from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

from tensorflow.keras.applications.densenet import preprocess_input

# =====================================================
# BASE DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pneumonia_densenet.keras"
)

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

app = FastAPI(
    title="Pneumonia Detection System"
)

# =====================================================
# MOUNT STATIC FILES
# =====================================================

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

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

# =====================================================
# HOME PAGE
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy"
    }

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read image
        contents = await file.read()

        # Open image
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Preprocess
        processed_image = preprocess_image(image)

        # Prediction
        prediction = model.predict(processed_image)[0][0]

        # Classification
        if prediction > 0.5:

            result = "PNEUMONIA"

            confidence = float(prediction)

        else:

            result = "NORMAL"

            confidence = float(1 - prediction)

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": round(confidence * 100, 2)
            }
        )

    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

# =====================================================
# FAVICON FIX
# =====================================================

@app.get("/favicon.ico")
async def favicon():

    return JSONResponse(content={})
