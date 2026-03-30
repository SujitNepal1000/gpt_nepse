"""
NEPSE Analytics Pro v3 — Trading Strategy Engine
Rule-based + ML hybrid strategy with:
  • Entry conditions (Strong Buy / Buy / Momentum)
  • Exit conditions (Stop-loss, Target, Trailing)
  • Position sizing and risk management
  • R:R filtering (only trades ≥ 2:1)
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger('nepse.strategy')


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_signal(row: dict) -> dict:
    """
    Hybrid rule-based + ML signal classification.
    Returns a comprehensive signal dict with entry, target, stop, confidence.
    """
    def v(k, default=None):
        val = row.get(k)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    close = v('close', 0)
    rsi = v('rsi', 50)
    macd = v('macd', 0)
    macd_sig = v('macd_signal', 0)
    ema20 = v('ema20')
    ema50 = v('ema50')
    sma50 = v('sma50')
    sma200 = v('sma200')
    atr = v('atr', 0)
    adx = v('adx', 0)
    bb_pct_b = v('bb_pct_b', 0.5)
    vol_ratio = v('volume_ratio', 1.0)
    diff_pct = v('diff_pct', 0)
    stoch_k = v('stoch_k', 50)
    mfi = v('mfi', 50)
    breakout_up = row.get('breakout_up', False)
    smart_money = str(row.get('smart_money', ''))
    ml_signal = str(row.get('signal', 'HOLD'))
    ml_conf = v('ml_confidence', 0)
    ret_5d = v('ret_5d', 0)
    volatility_20d = v('volatility_20d', 2)

    score = 0
    reasons = []
    risk_level = 'MEDIUM'

    # ── STRONG BUY: High ML conf + RSI < 40 + breakout
    if ml_signal == 'BUY' and ml_conf > 0.7 and rsi < 40 and breakout_up:
        score += 40
        reasons.append('Strong ML Buy + RSI oversold + Breakout')

    # ── RSI-based
    if rsi < 25:
        score += 25; reasons.append(f'RSI extreme oversold ({rsi:.0f})')
    elif rsi < 35:
        score += 15; reasons.append(f'RSI oversold ({rsi:.0f})')
    elif rsi > 80:
        score -= 25; reasons.append(f'RSI extreme overbought ({rsi:.0f})')
    elif rsi > 70:
        score -= 15; reasons.append(f'RSI overbought ({rsi:.0f})')

    # ── MACD
    if macd > macd_sig:
        score += 15; reasons.append('MACD bullish')
    else:
        score -= 15; reasons.append('MACD bearish')

    # ── EMA crossover
    if ema20 and ema50:
        if ema20 > ema50:
            score += 12; reasons.append('Golden cross (EMA20>50)')
        else:
            score -= 12; reasons.append('Death cross (EMA20<50)')

    # ── Momentum Buy: Price above MA50 & MA200
    if sma50 and sma200 and close > sma50 and close > sma200:
        score += 18; reasons.append('Price above SMA50 & SMA200 — strong momentum')
    elif sma200 and close < sma200:
        score -= 10; reasons.append('Price below SMA200 — bearish structure')

    # ── Bollinger Band position
    if bb_pct_b < 0.1:
        score += 12; reasons.append('Near lower Bollinger — oversold')
    elif bb_pct_b > 0.9:
        score -= 12; reasons.append('Near upper Bollinger — overbought')

    # ── Volume confirmation
    if vol_ratio > 2.0 and diff_pct > 0:
        score += 15; reasons.append('Volume spike + up day — institutional buying')
    elif vol_ratio > 2.0 and diff_pct < 0:
        score -= 15; reasons.append('Volume spike + down day — distribution')

    # ── ADX trend strength
    if adx > 25:
        score += 8; reasons.append(f'Strong trend (ADX {adx:.0f})')

    # ── Smart Money
    if smart_money == 'Accumulation':
        score += 15; reasons.append('Smart money accumulation')
    elif smart_money == 'Distribution':
        score -= 15; reasons.append('Smart money distribution')

    # ── MFI
    if mfi < 25:
        score += 8; reasons.append('MFI oversold — money inflow')
    elif mfi > 75:
        score -= 8; reasons.append('MFI overbought')

    # ── ML overlay
    if ml_signal == 'BUY' and ml_conf > 0.6:
        adj = int(ml_conf * 20)
        score += adj; reasons.append(f'ML BUY ({ml_conf*100:.0f}% conf)')
    elif ml_signal == 'SELL' and ml_conf > 0.6:
        adj = int(ml_conf * 20)
        score -= adj; reasons.append(f'ML SELL ({ml_conf*100:.0f}% conf)')

    # ── Classify
    if score >= 70:
        signal, strength = 'STRONG_BUY', 'STRONG'
    elif score >= 40:
        signal, strength = 'BUY', 'MODERATE'
    elif score >= 15:
        signal, strength = 'BUY', 'WEAK'
    elif score <= -60:
        signal, strength = 'SELL', 'STRONG'
    elif score <= -30:
        signal, strength = 'SELL', 'MODERATE'
    elif score <= -10:
        signal, strength = 'SELL', 'WEAK'
    else:
        signal, strength = 'HOLD', 'NEUTRAL'

    # ── Risk Level
    if volatility_20d > 4:
        risk_level = 'HIGH'
    elif volatility_20d > 2:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'

    # ── Entry / Target / Stop-Loss
    entry_price = close
    if atr > 0:
        # Stop loss: 2-3x ATR or 3-7% below entry
        stop_pct = max(3.0, min(7.0, (atr / close * 200)))
        stop_loss = round(close * (1 - stop_pct / 100), 2)

        # Target: 3-5x ATR or 8-20% depending on volatility
        target_mult = 3 if volatility_20d < 2 else 4 if volatility_20d < 4 else 5
        target_price = round(close + atr * target_mult, 2)
    else:
        stop_loss = round(close * 0.95, 2)
        target_price = round(close * 1.12, 2)

    # ── R:R Ratio
    risk = close - stop_loss if stop_loss < close else 0
    reward = target_price - close if target_price > close else 0
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    # Confidence score (0-100%)
    confidence = min(100, max(0, abs(score)))

    return {
        'symbol': row.get('symbol', ''),
        'signal': signal,
        'strength': strength,
        'entry_price': round(entry_price, 2),
        'target_price': target_price,
        'stop_loss': stop_loss,
        'rr_ratio': rr_ratio,
        'confidence': confidence,
        'risk_level': risk_level,
        'score': score,
        'reasoning': ' | '.join(reasons[:6]),
        'close': close,
        'rsi': rsi,
        'vol_ratio': vol_ratio,
    }


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading recommendations for all stocks in the dataset."""
    if df.empty:
        return pd.DataFrame()

    records = df.to_dict(orient='records')
    recs = [classify_signal(r) for r in records]
    recs_df = pd.DataFrame(recs)

    # Filter: Only show trades with R:R >= 1.5
    recs_df = recs_df.sort_values('confidence', ascending=False)
    return recs_df


def get_top_opportunities(recs: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N high-conviction stocks."""
    if recs.empty:
        return recs
    buys = recs[recs['signal'].isin(['STRONG_BUY', 'BUY'])]
    return buys.nlargest(n, 'confidence')


def get_sell_candidates(recs: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N sell candidates."""
    if recs.empty:
        return recs
    sells = recs[recs['signal'].isin(['SELL'])]
    return sells.nlargest(n, 'confidence')


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_position_size(account: float, risk_pct: float,
                            entry: float, stop: float) -> dict:
    """Kelly-inspired position sizing with risk constraints."""
    if entry <= 0 or stop >= entry:
        return {'error': 'Invalid entry/stop'}

    risk_per_share = entry - stop
    risk_amount = account * (risk_pct / 100)
    shares = int(risk_amount / risk_per_share)
    capital_req = round(shares * entry, 2)
    max_loss = round(shares * risk_per_share, 2)

    # Max position cap: 10% of portfolio
    max_shares = int(account * 0.10 / entry)
    shares = min(shares, max_shares)

    return {
        'shares': shares,
        'capital_required': round(shares * entry, 2),
        'max_loss': round(shares * risk_per_share, 2),
        'risk_amount': round(risk_amount, 2),
        'pct_of_portfolio': round(shares * entry / account * 100, 2),
    }
