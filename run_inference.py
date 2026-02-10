import time
import numpy as np
from src.data_ingestion.fetch_api import fetch_ohlcv
from src.features.indicators import add_indicators
from src.inference.pipeline import InferencePipeline
from src.utils import config

def main():
    # Initialize pipeline
    df = fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=1000)
    df = add_indicators(df)

    pipeline = InferencePipeline(
        model_path="best_lstm_model.pth",
        input_dim=df.shape[1],
        lookback=config.LOOKBACK
    )

    # Warm up buffer with historical data
    scaled = pipeline.preprocess(df)
    for row in scaled[-config.LOOKBACK:]:
        pipeline.update_buffer(row)

    # Simulate live stream
    while True:
        # Fetch latest candle (replace with API polling)
        new_df = fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=1)
        new_df = add_indicators(new_df)
        features = pipeline.preprocess(new_df)[-1]

        pipeline.update_buffer(features)
        pred = pipeline.predict_next()
        print(f"Predicted next close/log return: {pred:.5f}")

        time.sleep(60 * 60)  # wait for next 1h candle

if __name__ == "__main__":
    main()
