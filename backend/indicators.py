"""
NEPSE Analytics Pro v3 — Technical Indicators
Full suite: RSI, MACD, EMA, ATR, Bollinger, Stochastic, Williams R,
CCI, MFI, ADX, OBV, Momentum, ROC, Support/Resistance, Smart Money.
"""
import logging
import numpy as np
import pandas as pd

from config import (RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                    EMA_SHORT, EMA_LONG, ATR_PERIOD, BB_PERIOD, BB_STD,
                    STOCH_K, STOCH_D, WILLIAMS_R, CCI_PERIOD, MFI_PERIOD)

log = logging.getLogger('nepse.indicators')

# ── Core helpers ─────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()

def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (used in RSI, ATR)."""
    result = pd.Series(index=s.index, dtype=float)
    result.iloc[:period] = np.nan
    result.iloc[period-1] = s.iloc[:period].mean()
    for i in range(period, len(s)):
        result.iloc[i] = (result.iloc[i-1] * (period-1) + s.iloc[i]) / period
    return result


# ── Individual Indicators ────────────────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.rolling(period, min_periods=period).mean()
    avg_l = loss.rolling(period, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    rsi[avg_l == 0] = 100
    return rsi.round(4)


def compute_macd(close: pd.Series):
    fast = _ema(close, MACD_FAST)
    slow = _ema(close, MACD_SLOW)
    macd = fast - slow
    sig  = _ema(macd, MACD_SIGNAL)
    hist = macd - sig
    return macd.round(6), sig.round(6), hist.round(6)


def compute_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    pc   = close.shift(1)
    tr   = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().round(4)


def compute_bollinger(close: pd.Series, period=BB_PERIOD, n_std=BB_STD):
    mid   = close.rolling(period, min_periods=1).mean()
    std   = close.rolling(period, min_periods=1).std()
    upper = (mid + n_std * std).round(4)
    lower = (mid - n_std * std).round(4)
    width = ((upper - lower) / mid * 100).round(4)   # band width %
    pct_b = ((close - lower) / (upper - lower)).round(4)   # %B
    return upper, mid.round(4), lower, width, pct_b


def compute_stochastic(high: pd.Series, low: pd.Series,
                        close: pd.Series, k=STOCH_K, d=STOCH_D):
    low_k  = low.rolling(k, min_periods=1).min()
    high_k = high.rolling(k, min_periods=1).max()
    denom  = (high_k - low_k).replace(0, np.nan)
    stk    = ((close - low_k) / denom * 100).round(4)
    std    = stk.rolling(d, min_periods=1).mean().round(4)
    return stk, std


def compute_williams_r(high: pd.Series, low: pd.Series,
                        close: pd.Series, period=WILLIAMS_R) -> pd.Series:
    hh = high.rolling(period, min_periods=1).max()
    ll = low.rolling(period, min_periods=1).min()
    wr = ((hh - close) / (hh - ll).replace(0, np.nan) * -100).round(4)
    return wr


def compute_cci(high: pd.Series, low: pd.Series,
                close: pd.Series, period=CCI_PERIOD) -> pd.Series:
    tp  = (high + low + close) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = ((tp - sma) / (0.015 * mad.replace(0, np.nan))).round(4)
    return cci


def compute_mfi(high: pd.Series, low: pd.Series,
                close: pd.Series, vol: pd.Series,
                period=MFI_PERIOD) -> pd.Series:
    tp   = (high + low + close) / 3
    rmf  = tp * vol
    pos  = rmf.where(tp > tp.shift(1), 0)
    neg  = rmf.where(tp < tp.shift(1), 0)
    pos_r = pos.rolling(period, min_periods=1).sum()
    neg_r = neg.rolling(period, min_periods=1).sum()
    mfi   = (100 - 100 / (1 + pos_r / neg_r.replace(0, np.nan))).round(4)
    return mfi


def compute_adx(high: pd.Series, low: pd.Series,
                close: pd.Series, period=14):
    tr    = compute_atr(high, low, close, 1)
    dm_p  = (high - high.shift(1)).clip(lower=0)
    dm_m  = (low.shift(1) - low).clip(lower=0)
    dm_p  = dm_p.where(dm_p > dm_m, 0)
    dm_m  = dm_m.where(dm_m > dm_p, 0)

    atr_s = tr.rolling(period, min_periods=1).sum()
    di_p  = (dm_p.rolling(period, min_periods=1).sum() / atr_s.replace(0, np.nan) * 100).round(4)
    di_m  = (dm_m.rolling(period, min_periods=1).sum() / atr_s.replace(0, np.nan) * 100).round(4)
    dx    = ((di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan) * 100)
    adx   = dx.rolling(period, min_periods=1).mean().round(4)
    return adx, di_p, di_m


def compute_obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * vol).cumsum().round(2)


def compute_momentum(close: pd.Series, period=10) -> pd.Series:
    return (close - close.shift(period)).round(4)


def compute_roc(close: pd.Series, period=10) -> pd.Series:
    return ((close - close.shift(period)) / close.shift(period) * 100).round(4)


# ── Support & Resistance ─────────────────────────────────────────────────────

def find_sr_zones(closes: list, highs: list = None, lows: list = None,
                  window: int = 5, cluster_pct: float = 0.015):
    prices  = np.array(closes)
    h_arr   = np.array(highs) if highs else prices
    l_arr   = np.array(lows)  if lows  else prices

    res_levels, sup_levels = [], []
    for i in range(window, len(prices) - window):
        if (h_arr[i] >= h_arr[i-window:i].max() and
                h_arr[i] >= h_arr[i+1:i+window+1].max()):
            res_levels.append(round(float(h_arr[i]), 2))
        if (l_arr[i] <= l_arr[i-window:i].min() and
                l_arr[i] <= l_arr[i+1:i+window+1].min()):
            sup_levels.append(round(float(l_arr[i]), 2))

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(levels))
        cls    = [[levels[0]]]
        for l in levels[1:]:
            if (l - cls[-1][-1]) / max(abs(cls[-1][-1]), 1) < cluster_pct:
                cls[-1].append(l)
            else:
                cls.append([l])
        return [round(sum(c)/len(c), 2) for c in cls]

    return cluster(res_levels)[-5:], cluster(sup_levels)[-5:]


def detect_trend_structure(highs: list, lows: list, window: int = 10) -> str:
    if len(highs) < window * 2 or len(lows) < window * 2:
        return 'Insufficient data'
    h1, h2 = max(highs[-window:]),      max(highs[-window*2:-window])
    l1, l2 = min(lows[-window:]),       min(lows[-window*2:-window])
    if h1 > h2 and l1 > l2:
        return 'Bullish (HH/HL)'
    if h1 < h2 and l1 < l2:
        return 'Bearish (LH/LL)'
    return 'Sideways'


# ── Smart Money Detection ────────────────────────────────────────────────────

def detect_smart_money(df: pd.DataFrame) -> tuple[str, float]:
    """
    Returns (label, vol_ratio) where label is Accumulation/Distribution/Neutral.
    Considers: up-vs-down day volume, OBV direction, close position, vol spike.
    """
    if len(df) < 5:
        return 'Neutral', 1.0

    score     = 0
    avg_vol   = df['volume'].replace(0, np.nan).mean()
    vol_ratio = round(float(df['volume'].iloc[-1] / avg_vol), 4) if avg_vol else 1.0

    # 1. Volume on up vs down days (last 10)
    recent = df.tail(10)
    up_v   = recent[recent['diff_pct'] > 0]['volume'].mean() if len(recent[recent['diff_pct'] > 0]) else 0
    dn_v   = recent[recent['diff_pct'] < 0]['volume'].mean() if len(recent[recent['diff_pct'] < 0]) else 0
    if up_v > dn_v * 1.3:  score += 2
    elif dn_v > up_v * 1.3: score -= 2

    # 2. OBV direction (last 10 days)
    if 'obv' in df.columns and len(df) >= 10:
        obv_chg   = df['obv'].iloc[-1] - df['obv'].iloc[-10]
        price_chg = df['close'].iloc[-1] - df['close'].iloc[-10]
        if obv_chg > 0 and price_chg > 0:   score += 2
        elif obv_chg > 0 and price_chg < 0: score += 1   # bullish divergence
        elif obv_chg < 0:                    score -= 1

    # 3. Closing position in last 5 candles
    for _, row in df.tail(5).iterrows():
        rng = (row.get('high',0) or 0) - (row.get('low',0) or 0)
        if rng > 0:
            pos = ((row.get('close',0) or 0) - (row.get('low',0) or 0)) / rng
            score += (1 if pos > 0.7 else -1 if pos < 0.3 else 0)

    # 4. Volume spike today
    if vol_ratio > 2.0:
        score += (2 if (df['diff_pct'].iloc[-1] or 0) > 0 else -2)

    label = 'Accumulation' if score >= 3 else 'Distribution' if score <= -3 else 'Neutral'
    return label, vol_ratio


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all indicators to a symbol's sorted history DataFrame.
    Requires columns: date, open, high, low, close, volume (at minimum).
    """
    if df.empty or len(df) < 3:
        return df

    df = df.copy().sort_values('date').reset_index(drop=True)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    close = df['close'].ffill()
    high  = df['high'].ffill()
    low   = df['low'].ffill()
    vol   = df['volume'].fillna(0)

    # ── Trend
    df['rsi']            = compute_rsi(close)
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(close)
    df['ema20']          = _ema(close, EMA_SHORT).round(4)
    df['ema50']          = _ema(close, EMA_LONG).round(4)
    df['bb_upper'], df['bb_mid'], df['bb_lower'], df['bb_width'], df['bb_pct_b'] \
                         = compute_bollinger(close)
    df['atr']            = compute_atr(high, low, close)
    df['atr_pct']        = (df['atr'] / close * 100).round(4)
    df['stoch_k'], df['stoch_d']  = compute_stochastic(high, low, close)
    df['williams_r']     = compute_williams_r(high, low, close)
    df['cci']            = compute_cci(high, low, close)
    df['mfi']            = compute_mfi(high, low, close, vol)
    df['adx'], df['di_plus'], df['di_minus'] = compute_adx(high, low, close)
    df['obv']            = compute_obv(close, vol)
    df['momentum_10']    = compute_momentum(close, 10)
    df['roc_10']         = compute_roc(close, 10)

    # ── 52-week High/Low (rolling 252)
    if 'high_52w' not in df.columns or df['high_52w'].isna().all():
        df['high_52w'] = high.rolling(252, min_periods=1).max().round(4)
    if 'low_52w' not in df.columns or df['low_52w'].isna().all():
        df['low_52w']  = low.rolling(252, min_periods=1).min().round(4)

    # ── EMA trend
    df['ema_trend'] = np.where(close > df['ema50'], 'Uptrend', 'Downtrend')

    # ── Trend structure
    df['trend_struct'] = detect_trend_structure(high.tolist(), low.tolist())

    # ── Support & Resistance
    res_z, sup_z = find_sr_zones(close.tolist(), high.tolist(), low.tolist())
    cur = float(close.iloc[-1]) if not close.empty else 0
    sups = sorted([s for s in sup_z if s < cur], reverse=True)
    ress = sorted([r for r in res_z if r > cur])
    df['support']     = sups[0] if sups else (float(low.tail(20).min()) if len(low)>=20 else None)
    df['resistance']  = ress[0] if ress else (float(high.tail(20).max()) if len(high)>=20 else None)
    df['support2']    = sups[1] if len(sups)>1 else None
    df['resistance2'] = ress[1] if len(ress)>1 else None

    # ── Smart money
    sm, vr = detect_smart_money(df)
    df['smart_money'] = sm
    df['volume_ratio']   = vr
    # Fix for provided code: use volume_spike in signals logic if needed, but indicators.py uses vol_spike
    df['vol_spike']   = vr > 2.0

    # ── vwap_pct (if missing)
    if 'vwap' in df.columns and 'vwap_pct' not in df.columns:
        vwap_s = pd.to_numeric(df['vwap'], errors='coerce')
        df['vwap_pct'] = np.where(vwap_s > 0, (close - vwap_s)/vwap_s*100, np.nan)

    return df
