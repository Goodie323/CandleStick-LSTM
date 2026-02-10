import requests
import pandas as pd

def fetch_ohlcv(symbol="BTCUSDT", interval="1h", limit=1000):
    """
    Example: Binance OHLCV fetch
    """
    url = f"https://api1.binance.com/api/v3/klines"  # ← only this line changed (endpoint + base for better reliability)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params)
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df = df[["open_time","open","high","low","close","volume"]]
    df = df.astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("timestamp")