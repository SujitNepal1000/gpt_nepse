import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from db import engine

log = logging.getLogger('nepse.scraper')

def fetch_data():
    """
    Mock fetch for historical/snapshot data.
    Generates 20 days of history for NABIL to ensure RSI etc. are computable.
    """
    base_date = datetime.now().date()
    data = []
    
    # 20 days of history for NABIL
    for i in range(25, -1, -1):
        dt = base_date - timedelta(days=i)
        # Sequence forcing RSI down
        price = 600 - (25 - i) * 10 if i > 5 else 350 + (5 - i) * 5
        data.append({
            "symbol": "NABIL", "sector": "Banking", "ltp": price, "close": price, 
            "volume": 15000, "date": dt, "open": price, "high": price+2, "low": price-2
        })
        
    # NTC with high RSI sequence
    for i in range(25, -1, -1):
        dt = base_date - timedelta(days=i)
        price = 800 + (25 - i) * 15
        data.append({
            "symbol": "NTC", "sector": "Telecom", "ltp": price, "close": price, 
            "volume": 9000, "date": dt, "open": price, "high": price+5, "low": price-5
        })
        
    return pd.DataFrame(data)

def process_data(df):
    """
    Basic cleaning of raw data. 
    Technical indicators are now handled by indicators.py
    """
    if 'date' not in df.columns:
        df["date"] = pd.to_datetime(datetime.now().date())
    else:
        df['date'] = pd.to_datetime(df['date'])
        
    for col in ['ltp', 'close', 'volume', 'open', 'high', 'low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Placeholder for diff_pct if missing
    if 'diff_pct' not in df.columns and 'close' in df.columns:
        df['diff_pct'] = 0.0 # Will be calculated by pipeline with history
        
    return df

def save_to_db(df):
    df.to_sql("stock_data", engine, if_exists="append", index=False)

def fetch_live():
    # 🔁 Real implementation would scrape Sharesansar Live Trading page
    log.info("Fetching live market data...")
    # Return NABIL and NTC today
    data = [
        {"symbol": "NABIL", "ltp": 340, "close": 340, "volume": 2000, "diff_pct": -2.0, "open": 345, "high": 345, "low": 338},
        {"symbol": "NTC", "ltp": 1200, "close": 1200, "volume": 1200, "diff_pct": 3.5, "open": 1150, "high": 1210, "low": 1150},
    ]
    return data

def run():
    df = fetch_data()
    df = process_data(df)
    save_to_db(df)
    print("✅ Data saved")