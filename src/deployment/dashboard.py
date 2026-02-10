import streamlit as st
import requests
import pandas as pd
import time

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Candlestick LSTM Dashboard", layout="wide")

st.title("📈 Candlestick LSTM Predictor")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "close", "prediction"])

# Sidebar controls
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 15)

# Simulated live candle input (replace with exchange API)
def get_live_candle():
    # Example: random OHLCV + indicators
    import numpy as np
    features = np.random.rand(10).tolist()
    return features

# Main loop
placeholder = st.empty()

while True:
    # Get new candle features
    features = get_live_candle()

    # Call FastAPI backend
    response = requests.post(API_URL, json={"features": features})
    result = response.json()

    # Append to session data
    ts = pd.Timestamp.now()
    st.session_state.data.loc[len(st.session_state.data)] = [ts, features[3], result.get("prediction", None)]

    # Plot chart
    with placeholder.container():
        st.line_chart(st.session_state.data.set_index("timestamp")[["close", "prediction"]])

    time.sleep(refresh_rate)
