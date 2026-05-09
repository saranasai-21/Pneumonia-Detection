from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.densenet import preprocess_input

# =====================================================
# LOAD MODEL
# =====================================================

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "pneumonia_densenet.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

# =====================================================
# STATIC + TEMPLATES
# =====================================================

STATIC_DIR = os.path.join(BASE_DIR, "api", "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "api", "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

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
        {"request": request}
    )

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    if prediction > 0.5:
        result = "PNEUMONIA"
        confidence = float(prediction)
    else:
        result = "NORMAL"
        confidence = float(1 - prediction)

    return {
        "prediction": result,
        "confidence": round(confidence * 100, 2)
    }
