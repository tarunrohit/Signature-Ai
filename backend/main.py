# main.py
import os
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np

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

# --- MODEL LOADING ON STARTUP ---
print("Server starting up... Loading models.")
try:
    simple_cnn_model = tf.keras.models.load_model('best_signature_model_no_func.keras')
    print("SimpleCNN model loaded.")

    mobilenet_model = tf.keras.models.load_model('best_signature_model_mobilenet_streamed.keras')
    print("MobileNetV2 model loaded.")

    siamese_model = tf.keras.models.load_model('best_signature_siamese_model_final.keras', custom_objects={
        'contrastive_loss': contrastive_loss,
        'euclidean_distance': euclidean_distance
    })
    print("Siamese model loaded.")
    print("All models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading models: {e}")
    simple_cnn_model = mobilenet_model = siamese_model = None

# --- FastAPI App Setup ---
app = FastAPI(title="SignatureAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the SignatureAI Verification API."}

@app.post("/verify/")
async def verify_signature_endpoint(
    model_id: str = Form(...), 
    image: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None)
):
    # This check remains as a safeguard
    if model_id == 'simplecnn' and simple_cnn_model is None or \
       model_id == 'mobilenetv2' and mobilenet_model is None or \
       model_id == 'siamesenet' and siamese_model is None:
        raise HTTPException(status_code=503, detail="A required model failed to load on startup. Please check the server logs.")

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

    response_data.update({"model": model_id, "processingTime": processing_time})
    return response_data