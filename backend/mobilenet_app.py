import os
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.utils import get_file
from utils import preprocess_for_mobilenet

# --- FastAPI App Setup ---
app = FastAPI(title="MobileNetV2 Service")
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Model Loading from URL ---
print("MobileNetV2 Service: Initializing...")
MODEL_URL = "https://huggingface.co/Tarun5098/signature-ai-models/resolve/main/best_signature_model_mobilenet_streamed.keras"
model_path = get_file("mobilenet.keras", MODEL_URL, cache_dir=".", cache_subdir="models")
model = tf.keras.models.load_model(model_path)
print("MobileNetV2 model loaded successfully.")

@app.post("/verify/")
async def verify(image: UploadFile = File(...)):
    image_bytes = await image.read()
    processed_image = preprocess_for_mobilenet(image_bytes)
    prediction = model.predict(processed_image)[0][0]
    is_original = bool(prediction < 0.5)
    confidence = (1 - prediction) * 100 if is_original else prediction * 100
    return {"isOriginal": is_original, "confidence": float(confidence)}

@app.get("/")
def health_check():
    return {"status": "ok", "model": "MobileNetV2"}