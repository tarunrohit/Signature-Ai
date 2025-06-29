from fastapi import FastAPI, Form, File, UploadFile, Optional
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from utils import preprocess_for_siamese

# Custom functions must be defined to load the model
def euclidean_distance(vectors):
    s = tf.keras.backend.sum(tf.keras.backend.square(vectors[0] - vectors[1]), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

app = FastAPI()
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

custom_objects = {'contrastive_loss': contrastive_loss, 'euclidean_distance': euclidean_distance}
model = tf.keras.models.load_model('models/best_signature_siamese_model_final.keras', custom_objects=custom_objects)

@app.post("/verify/")
async def verify(image: UploadFile = File(...), reference_image: UploadFile = File(...)):
    anchor_bytes = await reference_image.read()
    candidate_bytes = await image.read()

    processed_anchor = preprocess_for_siamese(anchor_bytes)
    processed_candidate = preprocess_for_siamese(candidate_bytes)
    
    distance = model.predict([processed_anchor, processed_candidate])[0][0]
    threshold = 0.9 
    is_original = bool(distance < threshold)
    confidence = max(0, (1 - (distance / (threshold * 1.5)))) * 100
    return {"isOriginal": is_original, "confidence": float(confidence), "distance": float(distance)}