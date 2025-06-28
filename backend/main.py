# main.py
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import get_file # Import get_file utility

# Use an absolute import because we are running main.py as the top-level script
from utils import (
    preprocess_for_cnn, 
    preprocess_for_mobilenet, 
    preprocess_for_siamese
)

# --- LAZY LOADING & REMOTE MODELS STRATEGY ---
simple_cnn_model = None
mobilenet_model = None
siamese_model = None
print("Server started. Models will be loaded on first use (lazy loading).")

# --- MODIFIED: Replace these with your actual Hugging Face model URLs ---
MODEL_URLS = {
    "simple_cnn": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_no_func.keras",
    "mobilenet": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_mobilenet_streamed.keras",
    "siamese": "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_siamese_model_final.keras"
}

# --- Custom Functions for Siamese Model ---
def euclidean_distance(vectors):
    s = tf.keras.backend.sum(tf.keras.backend.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

custom_objects = {
    'contrastive_loss': contrastive_loss,
    'euclidean_distance': euclidean_distance
}

# --- FastAPI App Setup ---
app = FastAPI(title="SignatureAI API")

# MODIFIED: Prepare origins for deployment. Add your future Netlify URL here.
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://signature-ai.netlify.app"  # <-- IMPORTANT: Add your deployed frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to download and cache models
def download_and_load_model(model_name, url, custom_obj=None):
    model_path = get_file(f"{model_name}.keras", url, cache_dir=".", cache_subdir="models")
    return tf.keras.models.load_model(model_path, custom_objects=custom_obj)

# --- Modified loading functions ---
def get_simple_cnn_model():
    global simple_cnn_model
    if simple_cnn_model is None:
        print("Loading SimpleCNN model for the first time from URL...")
        try:
            simple_cnn_model = download_and_load_model("simple_cnn", MODEL_URLS["simple_cnn"])
            print("SimpleCNN model loaded successfully!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading SimpleCNN model: {e}")
    return simple_cnn_model

def get_mobilenet_model():
    global mobilenet_model
    if mobilenet_model is None:
        print("Loading MobileNetV2 model for the first time from URL...")
        try:
            mobilenet_model = download_and_load_model("mobilenet", MODEL_URLS["mobilenet"])
            print("MobileNetV2 model loaded successfully!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading MobileNetV2 model: {e}")
    return mobilenet_model
    
def get_siamese_model():
    global siamese_model
    if siamese_model is None:
        print("Loading Siamese model for the first time from URL...")
        try:
            siamese_model = download_and_load_model("siamese", MODEL_URLS["siamese"], custom_obj=custom_objects)
            print("Siamese model loaded successfully!")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading Siamese model: {e}")
    return siamese_model


@app.get("/")
def read_root():
    return {"message": "Welcome to the SignatureAI Verification API"}

# --- The /verify/ endpoint remains the same as the previous version ---
@app.post("/verify/")
async def verify_signature_endpoint(
    model_id: str = Form(...), 
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    
    image_bytes = await image.read()
    response_data = {}

    try:
        if model_id == 'simplecnn':
            model = get_simple_cnn_model()
            processed_image = preprocess_for_cnn(image_bytes)
            prediction = model.predict(processed_image)[0][0]
            is_original = bool(prediction < 0.5)
            confidence = (1 - prediction) * 100 if is_original else prediction * 100
            response_data = { "isOriginal": is_original, "confidence": float(confidence) }
            
        elif model_id == 'mobilenetv2':
            model = get_mobilenet_model()
            processed_image = preprocess_for_mobilenet(image_bytes)
            prediction = model.predict(processed_image)[0][0]
            is_original = bool(prediction < 0.5)
            confidence = (1 - prediction) * 100 if is_original else prediction * 100
            response_data = { "isOriginal": is_original, "confidence": float(confidence) }

        elif model_id == 'siamesenet':
            if not reference_image:
                raise HTTPException(status_code=400, detail="Reference image is required for the Siamese model.")
            
            anchor_bytes = await reference_image.read()
            model = get_siamese_model()

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
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


    end_time = time.time()
    processing_time = int((end_time - start_time) * 1000)

    response_data.update({
        "model": model_id,
        "processingTime": processing_time
    })
    return response_data