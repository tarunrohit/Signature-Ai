# main.py
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file
import threading

from utils import (
    preprocess_for_cnn, 
    preprocess_for_mobilenet, 
    preprocess_for_siamese
)

# --- MODEL PLACEHOLDERS & URLs ---
models = {
    "simple_cnn": None,
    "mobilenet": None,
    "siamese": None
}
MODEL_URLS = {
    "simple_cnn": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_no_func.keras",
    "mobilenet": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_mobilenet_streamed.keras",
    "siamese": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_siamese_model_final.keras"
}
MODELS_DIR = "models_cache"

# --- CUSTOM FUNCTIONS for loading the Siamese model ---
def euclidean_distance(vectors):
    s = tf.keras.backend.sum(tf.keras.backend.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# --- MODEL LOADING LOGIC (to be run in background) ---
def load_models_in_background():
    """Downloads and loads all models into the global 'models' dictionary."""
    print("Background thread started for model loading.")
    custom_objects_siamese = {'contrastive_loss': contrastive_loss, 'euclidean_distance': euclidean_distance}
    
    try:
        # Load SimpleCNN
        print("Downloading and loading simple_cnn model...")
        model_path = get_file("simple_cnn.keras", MODEL_URLS["simple_cnn"], cache_dir=".", cache_subdir=MODELS_DIR)
        models["simple_cnn"] = tf.keras.models.load_model(model_path)
        print("SimpleCNN model loaded successfully.")

        # Load MobileNetV2
        print("Downloading and loading mobilenet model...")
        model_path = get_file("mobilenet.keras", MODEL_URLS["mobilenet"], cache_dir=".", cache_subdir=MODELS_DIR)
        models["mobilenet"] = tf.keras.models.load_model(model_path)
        print("MobileNetV2 model loaded successfully.")
        
        # Load Siamese
        print("Downloading and loading siamese model...")
        model_path = get_file("siamese.keras", MODEL_URLS["siamese"], cache_dir=".", cache_subdir=MODELS_DIR)
        models["siamese"] = tf.keras.models.load_model(model_path, custom_objects=custom_objects_siamese)
        print("Siamese model loaded successfully.")

        print("All models have been loaded in the background.")
    except Exception as e:
        print(f"FATAL: A model failed to load in the background thread: {e}")

# --- FastAPI APP SETUP ---
# NOTE: We do NOT use the 'lifespan' manager here, as it conflicts with Gunicorn's behavior
app = FastAPI(title="SignatureAI API")

origins = [
    "http://localhost:5173",
    "https://signature-ai.netlify.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---
@app.get("/healthz", status_code=200)
def health_check():
    """This endpoint is used by Render to check if the server is alive."""
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the SignatureAI Verification API."}

@app.post("/verify/")
async def verify_signature_endpoint(
    model_id: str = Form(...), 
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    model_key = model_id.replace('v2', '')
    if models.get(model_key) is None:
        raise HTTPException(status_code=503, detail=f"The '{model_id}' model is not ready. It may still be loading. Please try again in a moment.")

    start_time = time.time()
    image_bytes = await image.read()
    response_data = {}
    
    model = models[model_key]

    if model_id == 'simplecnn':
        processed_image = preprocess_for_cnn(image_bytes)
        prediction = model.predict(processed_image)[0][0]
        is_original = bool(prediction < 0.5)
        confidence = (1 - prediction) * 100 if is_original else prediction * 100
        response_data = {"isOriginal": is_original, "confidence": float(confidence)}
            
    elif model_id == 'mobilenetv2':
        processed_image = preprocess_for_mobilenet(image_bytes)
        prediction = model.predict(processed_image)[0][0]
        is_original = bool(prediction < 0.5)
        confidence = (1 - prediction) * 100 if is_original else prediction * 100
        response_data = {"isOriginal": is_original, "confidence": float(confidence)}

    elif model_id == 'siamesenet':
        if not reference_image:
            raise HTTPException(status_code=400, detail="Reference image is required for the Siamese model.")
        
        anchor_bytes = await reference_image.read()
        processed_anchor = preprocess_for_siamese(anchor_bytes)
        processed_candidate = preprocess_for_siamese(image_bytes)
        distance = model.predict([processed_anchor, processed_candidate])[0][0]
        threshold = 0.9 
        is_original = bool(distance < threshold)
        confidence = max(0, (1 - (distance / (threshold * 1.5)))) * 100
        response_data = { "isOriginal": is_original, "confidence": float(confidence), "distance": float(distance) }
            
    else:
        raise HTTPException(status_code=400, detail="Invalid model ID provided.")

    end_time = time.time()
    processing_time = int((end_time - start_time) * 1000)

    response_data.update({"model": model_id, "processingTime": processing_time})
    return response_data

# --- START BACKGROUND LOADING THREAD ---
# This code runs only once when the script is first executed.
print("Starting background thread for model loading.")
thread = threading.Thread(target=load_models_in_background)
thread.daemon = True  # Allows main thread to exit even if this one is running
thread.start()