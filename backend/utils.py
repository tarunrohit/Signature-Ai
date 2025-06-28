# utils.py
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

IMG_HEIGHT = 224
IMG_WIDTH = 224

def resize_and_pad(image_bytes: bytes, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Decodes image bytes and preprocesses for Siamese/CNN models."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (max(1, new_w), max(1, new_h)), cv2.INTER_AREA)

    canvas = np.ones((target_h, target_w, 3), np.uint8) * 255
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas

def preprocess_for_mobilenet(image_bytes: bytes):
    """Prepares image for MobileNetV2 classification."""
    img_array = resize_and_pad(image_bytes, target_size=(224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    return mobilenet_preprocess_input(img_array)

def preprocess_for_cnn(image_bytes: bytes):
    """Prepares image for your custom CNN."""
    img_array = resize_and_pad(image_bytes, target_size=(150, 220)) # Use your CNN's dimensions
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32) / 255.0

def preprocess_for_siamese(image_bytes: bytes):
    """Prepares a single image for Siamese network prediction."""
    img_array = resize_and_pad(image_bytes, target_size=(224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    return mobilenet_preprocess_input(img_array)