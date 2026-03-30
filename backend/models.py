from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, Boolean, Text

Base = declarative_base()

class Stock(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    sector = Column(String)
    ltp = Column(Float)
    close = Column(Float)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    turnover = Column(Float)
    diff_pct = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    ema20 = Column(Float)
    ema50 = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    atr = Column(Float)
    adx = Column(Float)
    di_plus = Column(Float)
    di_minus = Column(Float)
    mfi = Column(Float)
    stoch_k = Column(Float)
    williams_r = Column(Float)
    cci = Column(Float)
    smart_money = Column(String)
    trend_struct = Column(String)
    signal = Column(String)
    entry = Column(Float)
    target = Column(Float)
    stoploss = Column(Float)
    date = Column(Date)

class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True)
    note = Column(Text)
    target_buy = Column(Float)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    alert_type = Column(String)
    message = Column(Text)
    severity = Column(String)
    is_read = Column(Boolean, default=False)
    date = Column(Date)
    close_price = Column(Float)
    trigger_val = Column(Float)