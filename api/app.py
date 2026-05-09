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
    # X-rays are grayscale, so this makes them visually like your example
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def find_last_conv_layer(model):
    # Search backward for a convolution layer
    for layer in reversed(model.layers):
        layer_name = layer.name.lower()
        if "conv" in layer_name:
            try:
                _ = layer.output
                return layer.name
            except Exception:
                pass
    return None

def make_gradcam_heatmap(img_array, model, pred_index=0):
    last_conv_layer_name = find_last_conv_layer(model)

    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolution layer for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + tf.keras.backend.epsilon())

    return heatmap.numpy()

def overlay_heatmap_on_image(original_bgr, heatmap, alpha=0.45):
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, color_map, alpha, 0)
    return overlay

def draw_rounded_region(overlay_bgr, heatmap, threshold=0.55):
    h, w = overlay_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    mask = np.uint8(heatmap_resized >= threshold) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = overlay_bgr.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 50:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(output, center, radius, (0, 0, 255), 4)
            cv2.circle(output, center, 6, (0, 0, 255), -1)

            x1, y1, ww, hh = cv2.boundingRect(largest)
            cv2.rectangle(output, (x1, y1), (x1 + ww, y1 + hh), (0, 0, 255), 2)

    return output

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

        if prediction > 0.5:
            result = "PNEUMONIA"
            confidence = prediction
        else:
            result = "NORMAL"
            confidence = 1 - prediction

        # Pseudo-colored image for display
        colored_bgr = make_pseudocolor_image(rgb_array)

        # Grad-CAM only for pneumonia
        if result == "PNEUMONIA":
            heatmap = make_gradcam_heatmap(processed_image, model)
            overlay = overlay_heatmap_on_image(original_bgr, heatmap, alpha=0.45)
            rounded = draw_rounded_region(overlay, heatmap, threshold=0.55)
        else:
            overlay = colored_bgr.copy()
            rounded = colored_bgr.copy()

        original_base64 = image_to_base64_bgr(original_bgr)
        colored_base64 = image_to_base64_bgr(colored_bgr)
        overlay_base64 = image_to_base64_bgr(overlay)
        rounded_base64 = image_to_base64_bgr(rounded)

        return JSONResponse(
            content={
                "prediction": result,
                "confidence": round(confidence * 100, 2),
                "original_image": original_base64,
                "colored_image": colored_base64,
                "heatmap_image": overlay_base64,
                "rounded_image": rounded_base64
            }
        )

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})
