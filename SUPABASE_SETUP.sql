-- NEPSE AI Trading System - Database Schema Setup
-- Run this in your Supabase SQL Editor

-- 1. Create stock_data table
CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(255),
    sector VARCHAR(255),
    ltp FLOAT8,
    close FLOAT8,
    open FLOAT8,
    high FLOAT8,
    low FLOAT8,
    volume FLOAT8,
    turnover FLOAT8,
    diff_pct FLOAT8,
    rsi FLOAT8,
    macd FLOAT8,
    macd_signal FLOAT8,
    macd_hist FLOAT8,
    ema20 FLOAT8,
    ema50 FLOAT8,
    bb_upper FLOAT8,
    bb_lower FLOAT8,
    atr FLOAT8,
    adx FLOAT8,
    di_plus FLOAT8,
    di_minus FLOAT8,
    mfi FLOAT8,
    stoch_k FLOAT8,
    williams_r FLOAT8,
    cci FLOAT8,
    smart_money VARCHAR(255),
    trend_struct VARCHAR(255),
    signal VARCHAR(255),
    entry FLOAT8,
    target FLOAT8,
    stoploss FLOAT8,
    date DATE
);

-- 2. Create watchlist table
CREATE TABLE IF NOT EXISTS watchlist (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(255) UNIQUE,
    note TEXT,
    target_buy FLOAT8
);

-- 3. Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(255),
    alert_type VARCHAR(255),
    message TEXT,
    severity VARCHAR(255),
    is_read BOOLEAN DEFAULT FALSE,
    date DATE,
    close_price FLOAT8,
    trigger_val FLOAT8
);

-- 4. Set up useful indexes
CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_data_date ON stock_data(date);
CREATE INDEX IF NOT EXISTS idx_alerts_date ON alerts(date);
