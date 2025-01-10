# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import tensorflow as tf
import numpy as np
import librosa
import shutil
import os
from typing import Dict, Any
import json
from audio_augmentation import process_audio

app = FastAPI()

# Middleware to allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your model
MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def get_class_message(predicted_class):
    """
    Gets detailed message for predicted class.
    """
    class_messages = {
        0: "Audio is real",
        1: "Audio is fake and forgery is Speech Synthesis",
        2: "Audio is fake and forgery is Speech Synthesis",
        3: "Audio is fake and forgery is Voice Conversion",
        4: "Audio is fake and forgery is Voice Conversion",
        5: "Audio is fake and forgery is Replay Attack",
        6: "Audio is fake and forgery is Replay Attack with Distant Microphone",
    }
    return class_messages.get(predicted_class, "Unknown class")

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return ("hello")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process audio with augmentation
        spectrogram = process_audio(file_location, apply_augmentation=True)
        
        if spectrogram is None:
            raise HTTPException(status_code=400, detail="Failed to process audio file")

        spectrogram = np.expand_dims(spectrogram, axis=0)
        predictions = model.predict(spectrogram)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        probabilities = (predictions * 100).astype(int).tolist()

        message = get_class_message(predicted_class)
        
        result = {
            "file_name": file.filename,
            "predicted_class": int(predicted_class),
            "probabilities": probabilities,
            "message": message,
            "status": "success"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)