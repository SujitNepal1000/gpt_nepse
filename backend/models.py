from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date

Base = declarative_base()

class Stock(Base):
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    sector = Column(String)
    ltp = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    rsi = Column(Float)
    ema_trend = Column(String)
    signal = Column(String)
    entry = Column(Float)
    target = Column(Float)
    stoploss = Column(Float)
    date = Column(Date)