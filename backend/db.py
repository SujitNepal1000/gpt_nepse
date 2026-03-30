from sqlalchemy import create_engine, text
import logging
import pandas as pd
from datetime import datetime, timedelta

from models import Base

DB_URL = "postgresql://postgres:admin@localhost:5432/nepsegpt"
engine = create_engine(DB_URL)
Base.metadata.create_all(engine)

log = logging.getLogger('nepse.db')

def get_latest_date():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT MAX(date) FROM stock_data")).scalar()
        return res

def get_all_symbols():
    df = pd.read_sql("SELECT DISTINCT symbol FROM stock_data ORDER BY symbol", engine)
    return df['symbol'].tolist()

def get_latest_snapshot():
    latest_date = get_latest_date()
    if not latest_date:
        return pd.DataFrame()
    query = text("SELECT * FROM stock_data WHERE date = :d")
    return pd.read_sql(query, engine, params={"d": latest_date})

def get_symbol_history(symbol, limit=300):
    query = text("SELECT * FROM stock_data WHERE symbol = :s ORDER BY date DESC LIMIT :l")
    df = pd.read_sql(query, engine, params={"s": symbol, "l": limit})
    return df.sort_values("date")

def get_market_summary_history(days=60):
    query = """
    SELECT date, 
           COUNT(*) as total_stocks,
           SUM(CASE WHEN diff_pct > 0 THEN 1 ELSE 0 END) as gainers,
           SUM(CASE WHEN diff_pct < 0 THEN 1 ELSE 0 END) as losers,
           SUM(volume) as total_vol
    FROM stock_data
    GROUP BY date
    ORDER BY date DESC
    LIMIT %s
    """ % days
    return pd.read_sql(query, engine).sort_values("date")

def get_recent_alerts(days=7, unread=False):
    since = datetime.now().date() - timedelta(days=days)
    query = "SELECT * FROM alerts WHERE date >= '%s'" % since
    if unread:
        query += " AND is_read = False"
    query += " ORDER BY date DESC"
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        log.error(f"Error fetching alerts: {e}")
        return pd.DataFrame()

def insert_alerts(alerts: list[dict]):
    """
    Batch insert alerts into the database.
    """
    if not alerts: return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO alerts (symbol, alert_type, message, severity, date, is_read)
                VALUES (:symbol, :alert_type, :message, :severity, :date, False)
            """), alerts)
            conn.commit()
            log.info(f"Inserted {len(alerts)} alerts into DB")
    except Exception as e:
        log.error(f"Failed to insert alerts: {e}")

def mark_alerts_read():
    with engine.connect() as conn:
        conn.execute(text("UPDATE alerts SET is_read = True"))
        conn.commit()

def get_watchlist():
    try:
        return pd.read_sql("SELECT * FROM watchlist", engine)
    except:
        return pd.DataFrame()

def add_watchlist(symbol, note="", target_buy=None):
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO watchlist (symbol, note, target_buy) VALUES (:s, :n, :t) ON CONFLICT (symbol) DO UPDATE SET note = :n, target_buy = :t"),
                     {"s": symbol, "n": note, "t": target_buy})
        conn.commit()

def remove_watchlist(symbol):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM watchlist WHERE symbol = :s"), {"s": symbol})
        conn.commit()

def get_latest_signals(sig_type=None, limit=50):
    latest_date = get_latest_date()
    query = "SELECT * FROM stock_data WHERE date = '%s'" % latest_date
    if sig_type:
        query += " AND signal = '%s'" % sig_type
    query += " ORDER BY date DESC LIMIT %s" % limit
    return pd.read_sql(query, engine)