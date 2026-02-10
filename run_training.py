import pandas as pd
from torch.utils.data import DataLoader
import torch

from src.data_ingestion.fetch_api import fetch_ohlcv
from src.features.indicators import add_indicators
from src.dataset.windowing import TimeSeriesDataset
from src.models.train import train_model
from src.utils import config

def main():
    # 1. Fetch data
    df = fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=1000)

    # 2. Feature engineering
    df = add_indicators(df)

    # 3. Dataset construction
    dataset = TimeSeriesDataset(df, lookback=config.LOOKBACK, target_col=config.TARGET)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 4. Train model
    model = train_model(train_dataset, val_dataset, input_dim=df.shape[1], device="cpu")

    print("Training complete. Best model saved as best_lstm_model.pth")

if __name__ == "__main__":
    main()
