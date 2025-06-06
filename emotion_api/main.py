from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = FastAPI(title="Emotion Detection API")

# Constants
MAX_WORDS = 10000
MAX_LEN = 100

# Define the base path for model files relative to the emotion_api directory
MODEL_BASE_PATH = "../emotion_model"

# Load model and tokenizer
try:
    model = tf.keras.models.load_model(os.path.join(MODEL_BASE_PATH, 'emotion_model.h5'))
    
    # Load tokenizer
    with open(os.path.join(MODEL_BASE_PATH, 'tokenizer.json'), 'r') as f:
        word_index = json.load(f)
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.word_index = word_index

    # Load emotions
    with open(os.path.join(MODEL_BASE_PATH, 'emotions.json'), 'r') as f:
        EMOTIONS = json.load(f)

    print("Model, tokenizer, and emotions loaded successfully!")
except Exception as e:
    print(f"Error loading model or related files: {e}")
    # Exit or handle the error appropriately if model loading fails
    # For now, we'll let the app start but predictions will fail
    model = None
    tokenizer = None
    EMOTIONS = []

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotion(input_data: TextInput):
    if model is None or tokenizer is None or not EMOTIONS:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess the input text
        sequence = tokenizer.texts_to_sequences([input_data.text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)
        
        # Make prediction
        prediction = model.predict(padded)
        emotion_idx = np.argmax(prediction[0])
        emotion = EMOTIONS[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": {
                emotion: float(conf) for emotion, conf in zip(EMOTIONS, prediction[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Emotion Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 