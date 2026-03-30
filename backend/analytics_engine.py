"""
NEPSE Analytics Pro v3 — Analytics Engine
Hedge-fund grade data engineering, feature generation, and alpha signal pipeline.
Handles: multi-sheet Excel ingestion, cleaning, derived features, cross-sectional analysis.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

log = logging.getLogger('nepse.analytics')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA ENGINEERING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_multi_sheet_excel(file_or_path) -> pd.DataFrame:
    """Load all sheets from an Excel file, merge into unified dataset."""
    try:
        sheets = pd.read_excel(file_or_path, sheet_name=None)
    except Exception:
        # Fallback: single CSV
        return pd.read_csv(file_or_path)

    frames = []
    for name, df in sheets.items():
        df['_sheet'] = name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(sheets)} sheets → {len(merged)} rows")
    return merged


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production-grade cleaning:
      - Column normalisation
      - Numeric coercion
      - Duplicate removal
      - Outlier capping (IQR)
      - Missing value handling
    """
    # ── Normalise column names
    rename_map = {
        'Symbol': 'symbol', 'Ticker': 'symbol', 'SYMBOL': 'symbol',
        'LTP': 'ltp', 'Close': 'close', 'CLOSE': 'close', 'Close Price': 'close',
        'Open': 'open', 'OPEN': 'open', 'Open Price': 'open',
        'High': 'high', 'HIGH': 'high', 'Low': 'low', 'LOW': 'low',
        'Volume': 'volume', 'Vol': 'volume', 'VOLUME': 'volume', 'Qty': 'volume',
        'Turnover': 'turnover', 'TURNOVER': 'turnover', 'Amount': 'turnover',
        'Diff %': 'diff_pct', 'Diff Pct': 'diff_pct', 'Change%': 'diff_pct',
        '% Change': 'diff_pct', 'Change': 'diff_pct',
        'No. of Transactions': 'transactions', 'Txn': 'transactions',
        'Transactions': 'transactions',
        'Date': 'date', 'DATE': 'date', 'Business Date': 'date',
        'Sector': 'sector', 'SECTOR': 'sector',
        'Previous Close': 'prev_close', 'Prev Close': 'prev_close',
    }
    df = df.rename(columns=rename_map)

    # ── Numeric coercion
    num_cols = ['ltp', 'close', 'open', 'high', 'low', 'volume',
                'turnover', 'diff_pct', 'transactions', 'prev_close']
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill close from ltp if missing
    if 'close' not in df.columns and 'ltp' in df.columns:
        df['close'] = df['ltp']
    if 'close' in df.columns and 'ltp' in df.columns:
        df['close'] = df['close'].fillna(df['ltp'])

    # ── Date handling
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.Timestamp(datetime.now().date())

    # ── Drop duplicates
    if 'symbol' in df.columns:
        df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')

    # ── Outlier capping (IQR method) for price columns
    for col in ['close', 'volume']:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q99)

    # ── Fill missing OHLC
    for col in ['open', 'high', 'low']:
        if col in df.columns:
            df[col] = df[col].fillna(df.get('close', 0))

    df['volume'] = df.get('volume', pd.Series(0)).fillna(0)
    df['turnover'] = df.get('turnover', pd.Series(0)).fillna(0)

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ADVANCED DERIVED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-symbol derived features.
    Must be called on a single symbol's time-series (sorted by date).
    """
    df = df.copy().sort_values('date').reset_index(drop=True)
    c = df['close'].astype(float)
    v = df['volume'].astype(float).fillna(0)
    h = df['high'].astype(float).fillna(c)
    l = df['low'].astype(float).fillna(c)
    o = df['open'].astype(float).fillna(c)

    # ── Returns
    df['ret_1d'] = c.pct_change(1) * 100
    df['ret_5d'] = c.pct_change(5) * 100
    df['ret_10d'] = c.pct_change(10) * 100
    df['ret_20d'] = c.pct_change(20) * 100  # ~monthly

    # ── Volatility (rolling std of daily returns)
    daily_ret = c.pct_change()
    df['volatility_5d'] = daily_ret.rolling(5, min_periods=2).std() * 100
    df['volatility_20d'] = daily_ret.rolling(20, min_periods=5).std() * 100
    df['volatility_60d'] = daily_ret.rolling(60, min_periods=10).std() * 100

    # ── Liquidity
    df['avg_volume_20d'] = v.rolling(20, min_periods=1).mean()
    df['volume_ratio'] = (v / df['avg_volume_20d'].replace(0, np.nan)).round(4)
    df['turnover_ratio'] = (df.get('turnover', pd.Series(0)).astype(float) /
                            df.get('turnover', pd.Series(1)).astype(float).rolling(20, min_periods=1).mean().replace(0, np.nan)).round(4)

    # ── Price Momentum
    df['momentum_5d'] = (c - c.shift(5)).round(4)
    df['momentum_10d'] = (c - c.shift(10)).round(4)
    df['momentum_20d'] = (c - c.shift(20)).round(4)

    # ── VWAP (Volume Weighted Average Price)
    cum_vol = v.cumsum()
    cum_vp = (c * v).cumsum()
    df['vwap'] = (cum_vp / cum_vol.replace(0, np.nan)).round(4)
    df['vwap_pct'] = ((c - df['vwap']) / df['vwap'].replace(0, np.nan) * 100).round(4)

    # ── SMA (20, 50, 200)
    df['sma20'] = c.rolling(20, min_periods=1).mean().round(4)
    df['sma50'] = c.rolling(50, min_periods=1).mean().round(4)
    df['sma200'] = c.rolling(200, min_periods=1).mean().round(4)

    # ── EMA crossover signals
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['ema_cross_signal'] = np.where(
        (ema12 > ema26) & (ema12.shift(1) <= ema26.shift(1)), 'BULL_CROSS',
        np.where((ema12 < ema26) & (ema12.shift(1) >= ema26.shift(1)), 'BEAR_CROSS', 'NONE')
    )

    # ── Gap detection
    df['gap_pct'] = ((o - c.shift(1)) / c.shift(1).replace(0, np.nan) * 100).round(4)
    df['gap_up'] = df['gap_pct'] > 2.0
    df['gap_down'] = df['gap_pct'] < -2.0

    # ── Breakout detection (close above 20-day high)
    rolling_high_20 = h.rolling(20, min_periods=5).max()
    rolling_low_20 = l.rolling(20, min_periods=5).min()
    df['breakout_up'] = c > rolling_high_20.shift(1)
    df['breakout_down'] = c < rolling_low_20.shift(1)

    # ── Volume spike detection
    vol_mean = v.rolling(20, min_periods=5).mean()
    vol_std = v.rolling(20, min_periods=5).std()
    df['vol_spike'] = v > (vol_mean + 2 * vol_std.fillna(0))

    # ── Unusual transaction activity
    if 'transactions' in df.columns:
        txn = df['transactions'].astype(float).fillna(0)
        txn_mean = txn.rolling(20, min_periods=5).mean()
        txn_std = txn.rolling(20, min_periods=5).std().fillna(0)
        df['unusual_txn'] = txn > (txn_mean + 2 * txn_std)
    else:
        df['unusual_txn'] = False

    # ── Drawdown from peak
    cummax = c.cummax()
    df['drawdown_pct'] = ((c - cummax) / cummax.replace(0, np.nan) * 100).round(4)
    df['max_drawdown_20d'] = df['drawdown_pct'].rolling(20, min_periods=1).min().round(4)

    return df


def compute_cross_sectional(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional features across all stocks for a given date.
    Rank stocks by performance within sector and vs market.
    """
    if all_data.empty:
        return all_data

    df = all_data.copy()

    # ── Market-level stats
    market_ret = df.groupby('date')['ret_1d'].transform('mean')
    df['excess_return'] = (df['ret_1d'] - market_ret).round(4)

    # ── Relative Strength vs Market
    df['rel_strength_market'] = (df['ret_20d'] - df.groupby('date')['ret_20d'].transform('mean')).round(4)

    # ── Sector-relative performance
    if 'sector' in df.columns:
        sector_ret = df.groupby(['date', 'sector'])['ret_1d'].transform('mean')
        df['sector_rel_perf'] = (df['ret_1d'] - sector_ret).round(4)

        # Rank within sector
        df['sector_rank'] = df.groupby(['date', 'sector'])['ret_1d'].rank(ascending=False, method='min')
    else:
        df['sector_rel_perf'] = 0
        df['sector_rank'] = 0

    # ── Overall performance rank
    df['market_rank'] = df.groupby('date')['ret_1d'].rank(ascending=False, method='min')

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FULL ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end analysis pipeline:
    1. Clean data
    2. Compute per-symbol derived features + technical indicators
    3. Compute cross-sectional features
    """
    import indicators

    df = clean_dataframe(df)
    log.info(f"Cleaned: {len(df)} rows, {df['symbol'].nunique() if 'symbol' in df.columns else 0} symbols")

    # Per-symbol processing
    processed = []
    symbols = df['symbol'].unique() if 'symbol' in df.columns else ['UNKNOWN']

    for sym in symbols:
        sym_df = df[df['symbol'] == sym].copy() if 'symbol' in df.columns else df.copy()
        sym_df = sym_df.sort_values('date')

        # Technical indicators
        if len(sym_df) >= 3:
            sym_df = indicators.compute_all(sym_df)

        # Derived features
        sym_df = compute_derived_features(sym_df)
        processed.append(sym_df)

    if not processed:
        return pd.DataFrame()

    result = pd.concat(processed, ignore_index=True)

    # Cross-sectional features
    result = compute_cross_sectional(result)

    log.info(f"Analysis complete: {len(result)} rows, {result.shape[1]} features")
    return result


def get_latest_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent row per symbol from the analyzed data."""
    if df.empty:
        return df
    latest = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)
    return latest
