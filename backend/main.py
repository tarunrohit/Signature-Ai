# main.py
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file
from contextlib import asynccontextmanager

from utils import (
    preprocess_for_cnn, 
    preprocess_for_mobilenet, 
    preprocess_for_siamese
)

# --- MODEL PLACEHOLDERS ---
models = {}

# --- CUSTOM FUNCTIONS (needed for loading) ---
def euclidean_distance(vectors):
    s = tf.keras.backend.sum(tf.keras.backend.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# --- MODEL LOADING & LIFESPAN MANAGEMENT ---
MODEL_URLS = {
    "simple_cnn": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_no_func.keras",
    "mobilenet": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_mobilenet_streamed.keras",
    "siamese": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_siamese_model_final.keras"
}
MODELS_DIR = "models_cache"

def download_and_load_model(model_name, url, custom_objects=None):
    print(f"Downloading and loading {model_name} model...")
    try:
        model_path = get_file(f"{model_name}.keras", url, cache_dir=".", cache_subdir=MODELS_DIR)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR loading {model_name}: {e}")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan startup: Beginning model loading...")
    
    custom_objects_siamese = {'contrastive_loss': contrastive_loss, 'euclidean_distance': euclidean_distance}
    
    models["simple_cnn"] = download_and_load_model("simple_cnn", MODEL_URLS["simple_cnn"])
    models["mobilenet"] = download_and_load_model("mobilenet", MODEL_URLS["mobilenet"])
    models["siamese"] = download_and_load_model("siamese", MODEL_URLS["siamese"], custom_objects=custom_objects_siamese)
    
    print("All models loaded. Application is ready.")
    yield
    print("Server shutting down.")


# Pass the lifespan manager to the app
app = FastAPI(title="SignatureAI API", lifespan=lifespan)

# --- CORS Middleware ---
origins = [
    "https://signature-ai.netlify.app",  # Your deployed frontend URL
    # Add localhost for local testing
    "http://localhost:5173",
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
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the SignatureAI Verification API. Models are loaded."}

@app.post("/verify/")
async def verify_signature_endpoint(
    model_id: str = Form(...), 
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    model_key = model_id.replace('v2', '')
    if models.get(model_key) is None:
        raise HTTPException(status_code=503, detail=f"The '{model_id}' model is not available. Please check server logs.")

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