# main.py
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file

from utils import (
    preprocess_for_cnn, 
    preprocess_for_mobilenet, 
    preprocess_for_siamese
)

# --- CUSTOM FUNCTIONS (needed for loading) ---
def euclidean_distance(vectors):
    s = tf.keras.backend.sum(tf.keras.backend.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# --- MODEL LOADING ON STARTUP (The Production Strategy) ---
print("Server starting up... This may take a few minutes as models are downloaded.")

MODEL_URLS = {
    "simple_cnn": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_no_func.keras",
    "mobilenet": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_mobilenet_streamed.keras",
    "siamese": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_siamese_model_final.keras"
}
MODELS_DIR = "models_cache"

# Helper function to download and load models
def download_and_load_model(model_name, url, custom_objects=None):
    print(f"Downloading and loading {model_name} model...")
    try:
        model_path = get_file(f"{model_name}.keras", url, cache_dir=".", cache_subdir=MODELS_DIR)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_name}: {e}")
        return None

# Load all models now
simple_cnn_model = download_and_load_model("simple_cnn", MODEL_URLS["simple_cnn"])
mobilenet_model = download_and_load_model("mobilenet", MODEL_URLS["mobilenet"])
siamese_model = download_and_load_model("siamese", MODEL_URLS["siamese"], custom_objects={
    'contrastive_loss': contrastive_loss,
    'euclidean_distance': euclidean_distance
})

# --- FastAPI App Setup ---
app = FastAPI(title="SignatureAI API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://signature-ai.netlify.app" # Your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the SignatureAI Verification API. Models are loaded."}

@app.post("/verify/")
async def verify_signature_endpoint(
    model_id: str = Form(...), 
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    # Check if models were loaded correctly on startup
    if not all([simple_cnn_model, mobilenet_model, siamese_model]):
        raise HTTPException(status_code=503, detail="One or more models are not available. Please check server logs.")

    start_time = time.time()
    image_bytes = await image.read()
    response_data = {}

    if model_id == 'simplecnn':
        model = simple_cnn_model
        processed_image = preprocess_for_cnn(image_bytes)
        prediction = model.predict(processed_image)[0][0]
        is_original = bool(prediction < 0.5)
        confidence = (1 - prediction) * 100 if is_original else prediction * 100
        response_data = {"isOriginal": is_original, "confidence": float(confidence)}
            
    elif model_id == 'mobilenetv2':
        model = mobilenet_model
        processed_image = preprocess_for_mobilenet(image_bytes)
        prediction = model.predict(processed_image)[0][0]
        is_original = bool(prediction < 0.5)
        confidence = (1 - prediction) * 100 if is_original else prediction * 100
        response_data = {"isOriginal": is_original, "confidence": float(confidence)}

    elif model_id == 'siamesenet':
        if not reference_image:
            raise HTTPException(status_code=400, detail="Reference image is required for the Siamese model.")
        
        anchor_bytes = await reference_image.read()
        model = siamese_model
        processed_anchor = preprocess_for_siamese(anchor_bytes)
        processed_candidate = preprocess_for_siamese(image_bytes)
        distance = model.predict([processed_anchor, processed_candidate])[0][0]
        threshold = 0.9 
        is_original = bool(distance < threshold)
        confidence = max(0, (1 - (distance / (threshold * 1.5)))) * 100
        response_data = { 
            "isOriginal": is_original, 
            "confidence": float(confidence),
            "distance": float(distance)
        }
            
    else:
        raise HTTPException(status_code=400, detail="Invalid model ID provided.")

    end_time = time.time()
    processing_time = int((end_time - start_time) * 1000)

    response_data.update({
        "model": model_id,
        "processingTime": processing_time
    })
    return response_data