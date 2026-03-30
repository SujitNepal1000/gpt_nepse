import logging
import pandas as pd
from datetime import datetime, date
from db import engine, get_latest_date
from scraper import fetch_data, process_data, save_to_db, fetch_live
from ml_engine import train_ensemble
import schedule
import time
from apscheduler.schedulers.background import BackgroundScheduler

import indicators
from ml_engine import compute_signal
from alerts import process_alerts

log = logging.getLogger('nepse.pipeline')

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_today, 'cron', hour=16, minute=5) # 4:05 PM NPT
    scheduler.add_job(retrain_model, 'cron', day_of_week='sat', hour=10) # Saturday 10 AM
    scheduler.start()
    return scheduler

def process_with_history(new_df):
    """
    For each stock in new_df, fetch history from DB,
    compute all indicators, and return the updated latest rows.
    """
    log.info(f"Processing indicators for {len(new_df)} stocks")
    processed_rows = []
    
    for _, row in new_df.iterrows():
        sym = row['symbol']
        try:
            # Fetch last 250 days for this symbol to compute indicators accurately
            hist = pd.read_sql(f"SELECT * FROM stock_data WHERE symbol = '{sym}' ORDER BY date ASC LIMIT 250", engine)
            
            # Combine
            new_row_df = pd.DataFrame([row])
            full_df = pd.concat([hist, new_row_df], ignore_index=True).drop_duplicates('date', keep='last')
            
            # Compute all indicators
            full_df = indicators.compute_all(full_df)
            
            # Compute ML signals
            full_df = compute_signal(full_df)
            
            # Get only the latest row
            latest = full_df.iloc[-1:].copy()
            processed_rows.append(latest)
        except Exception as e:
            log.error(f"Error processing {sym}: {e}")
            processed_rows.append(pd.DataFrame([row]))
            
    if not processed_rows: return pd.DataFrame()
    return pd.concat(processed_rows, ignore_index=True)

def run_today():
    log.info("Running daily fetch and analysis...")
    try:
        # 1. Fetch live snapshot
        raw_data = fetch_live()
        if not raw_data: 
            log.warning("No live data fetched")
            return {"status": "no_data"}
            
        df = clean_batch(raw_data, datetime.now().date())
        
        # 2. Process with history and indicators
        df = process_with_history(df)
        
        # 3. Save to DB
        if not df.empty:
            save_to_db(df)
            
            # 4. Process Alerts
            process_alerts(df)
            
            return {"status": "success", "stored": len(df)}
    except Exception as e:
        log.error(f"Daily fetch failed: {e}")
        return {"status": "error", "message": str(e)}
    return {"status": "no_data"}

def run_single_date(date_str):
    log.info(f"Fetching data for {date_str}")
    # Placeholder: In a real app, this would query historical API
    return {"status": "mock_success", "date": date_str}

def run_backfill(days=90):
    log.info(f"Starting backfill for {days} days")
    # Placeholder: Simulation of backfill
    return f"Backfilled mock data for {days} days"

def retrain_model():
    log.info("Retraining ML ensemble...")
    try:
        # Load enough data for training
        df = pd.read_sql("SELECT * FROM stock_data ORDER BY date ASC", engine)
        result = train_ensemble(df)
        return result
    except Exception as e:
        log.error(f"Retrain failed: {e}")
        return {"status": "error", "message": str(e)}

def clean_batch(raw_data, batch_date):
    # Helper to clean raw scraped data
    df = pd.DataFrame(raw_data)
    if df.empty: return df
    df['date'] = batch_date
    # Basic cleaning similar to app.py / upload logic
    for col in ['close', 'volume', 'diff_pct', 'turnover', 'rsi']:

        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df
