import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.lstm_model import LSTMPredictor

class InferencePipeline:
    def __init__(self, model_path, input_dim, lookback=60, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = LSTMPredictor(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.lookback = lookback
        self.scaler = StandardScaler()  # fit on training data beforehand

        # rolling buffer for live candles
        self.buffer = []

    def preprocess(self, df):
        """
        df: pandas DataFrame with OHLCV + indicators
        """
        values = df.values
        scaled = self.scaler.transform(values)  # use fitted scaler
        return scaled

    def update_buffer(self, new_candle):
        """
        new_candle: np.array of features for one candle
        """
        self.buffer.append(new_candle)
        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)

    def predict_next(self):
        if len(self.buffer) < self.lookback:
            raise ValueError("Not enough data in buffer for prediction")

        X = np.array(self.buffer[-self.lookback:])  # shape (lookback, features)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(X).cpu().numpy().flatten()[0]
        return pred
