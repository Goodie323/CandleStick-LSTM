from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

from inference.pipeline import InferencePipeline

# Define request schema
class CandleRequest(BaseModel):
    features: list  # OHLCV + indicators for one candle

# Initialize FastAPI app
app = FastAPI(title="Candlestick LSTM Predictor")

# Load pipeline
pipeline = InferencePipeline(
    model_path="best_lstm_model.pth",
    input_dim=10,   # adjust to your feature count
    lookback=60
)

@app.post("/predict")
def predict_next(candle: CandleRequest):
    # Update buffer with new candle
    pipeline.update_buffer(np.array(candle.features))

    # Only predict if buffer is full
    if len(pipeline.buffer) < pipeline.lookback:
        return {"status": "waiting", "message": "Not enough data yet"}

    pred = pipeline.predict_next()
    return {"status": "ok", "prediction": float(pred)}

@app.get("/health")
def health_check():
    return {"status": "running"}
