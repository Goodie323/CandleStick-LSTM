# CandleStick-LSTM
An end‑to‑end system for predicting the next candlestick close price (or log return) using an LSTM neural network.
Designed for liquid markets (crypto, forex, stocks) with configurable timeframes (default: 1h).

🚀 Features
Data ingestion from exchange APIs or CSVs

Feature engineering: OHLCV + EMA, RSI, MACD, ATR, volatility, lag features

Dataset construction: sliding windows, time‑aware splits, leakage prevention

Model: multi‑layer LSTM with dropout + dense output

Training: early stopping, checkpoints, GPU acceleration

Evaluation: RMSE, MAE, directional accuracy, volatility regime analysis

Inference pipeline: rolling predictions, real‑time or batch mode

Deployment ready: FastAPI backend, Streamlit dashboard (optional)

Unit tests: features, dataset, model forward pass

⚡ Quickstart
1. Install dependencies
    pip install -r requirements.txt
2. Train model
    python run_training.py


📊 Evaluation Metrics
RMSE / MAE — regression accuracy

Directional Accuracy — % correct up/down moves

Backtesting — strategy simulation with PnL, Sharpe ratio

Limitations
Market non‑stationarity & concept drift

Black swan events not predictable

Predictions are probabilistic, not guarantees