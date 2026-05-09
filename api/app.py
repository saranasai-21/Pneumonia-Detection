from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from tensorflow.keras.applications.densenet import preprocess_input

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = r"models/pneumonia_densenet.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Pneumonia Detection API"
)

IMG_SIZE = 256

# =====================================================
# PREPROCESS FUNCTION
# =====================================================

def preprocess_image(image):

    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy
    image = np.array(image)

    # Ensure float32
    image = image.astype(np.float32)

    # DenseNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Pneumonia Detection API Running"
    }

# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        processed_image = preprocess_image(image)

        # Prediction
        prediction = model.predict(processed_image)[0][0]

        # Result
        if prediction > 0.64:
            result = "PNEUMONIA"
            confidence = float(prediction)
        else:
            result = "NORMAL"
            confidence = float(1 - prediction)

        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }

    except Exception as e:

        return {
            "error": str(e)
        }