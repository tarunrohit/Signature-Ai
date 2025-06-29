from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from utils import preprocess_for_cnn

app = FastAPI()
origins = ["*"] # Allow all for simplicity
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = tf.keras.models.load_model('models/best_signature_model_no_func.keras')

@app.post("/verify/")
async def verify(image: UploadFile = File(...)):
    image_bytes = await image.read()
    processed_image = preprocess_for_cnn(image_bytes)
    prediction = model.predict(processed_image)[0][0]
    is_original = bool(prediction < 0.5)
    confidence = (1 - prediction) * 100 if is_original else prediction * 100
    return {"isOriginal": is_original, "confidence": float(confidence)}