import pandas as pd
import numpy as np
from datetime import datetime
from db import engine

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_signal(row):
    score = 0

    # RSI
    if row["rsi"] < 30:
        score += 2
    elif row["rsi"] < 50:
        score += 1

    # Trend
    if row["ema_trend"] == "Uptrend":
        score += 2

    # Volume spike
    if row["volume"] > row["volume_avg"]:
        score += 2

    # Final decision
    if score >= 5:
        return "STRONG BUY"
    elif score >= 3:
        return "BUY"
    else:
        return "HOLD"

def fetch_data():
    # 🔁 Replace with Sharesansar API
    data = [
        {"symbol": "NABIL", "sector": "Banking", "ltp": 500, "close": 490, "volume": 15000},
        {"symbol": "NTC", "sector": "Telecom", "ltp": 800, "close": 780, "volume": 9000},
    ]
    return pd.DataFrame(data)

def process_data(df):
    df["date"] = datetime.now().date()

    # RSI
    df["rsi"] = calculate_rsi(df["close"])

    # Volume avg
    df["volume_avg"] = df["volume"].rolling(3).mean()

    # EMA trend
    df["ema"] = df["close"].ewm(span=50).mean()
    df["ema_trend"] = np.where(df["close"] > df["ema"], "Uptrend", "Downtrend")

    # Signals
    df["signal"] = df.apply(generate_signal, axis=1)

    # Trade levels
    df["entry"] = df["ltp"]
    df["target"] = df["ltp"] * 1.05
    df["stoploss"] = df["ltp"] * 0.97

    return df

def save_to_db(df):
    df.to_sql("stock_data", engine, if_exists="append", index=False)

def run():
    df = fetch_data()
    df = process_data(df)
    save_to_db(df)
    print("✅ Data saved")