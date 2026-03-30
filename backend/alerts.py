"""
NEPSE Analytics Pro v3 — Alerts Engine
"""
import logging
from datetime import date
import numpy as np
from sqlalchemy import text

log = logging.getLogger('nepse.alerts')

ALERT_TYPES = {
    'RSI_EXTREME_OS':   ('HIGH',   'RSI {val:.1f} — Extreme oversold. Strong bounce probability.'),
    'RSI_OVERSOLD':     ('MEDIUM', 'RSI {val:.1f} — Oversold zone. Watch for reversal.'),
    'RSI_EXTREME_OB':   ('HIGH',   'RSI {val:.1f} — Extreme overbought. Consider exiting.'),
    'RSI_OVERBOUGHT':   ('MEDIUM', 'RSI {val:.1f} — Overbought. Avoid new buys.'),
    'MACD_BULL_CROSS':  ('HIGH',   'MACD bullish crossover — Strong buy signal!'),
    'MACD_BEAR_CROSS':  ('HIGH',   'MACD bearish crossover — Exit or avoid.'),
    'GOLDEN_CROSS':     ('HIGH',   'GOLDEN CROSS: EMA20 crossed above EMA50 — Powerful uptrend!'),
    'DEATH_CROSS':      ('HIGH',   'DEATH CROSS: EMA20 crossed below EMA50 — Bearish warning!'),
    'VOL_SPIKE_UP':     ('MEDIUM', 'Volume spike on up move — Institutional buying detected.'),
    'VOL_SPIKE_DOWN':   ('MEDIUM', 'Volume spike on down move — Institutional selling detected.'),
    '52W_HIGH_BREAK':   ('HIGH',   '52-week HIGH breakout! Major bullish signal.'),
    '52W_LOW_BREAK':    ('HIGH',   'Near 52-week LOW. Potential support test.'),
    'ACCUM':            ('MEDIUM', 'Smart Money Accumulation — Institutions buying quietly.'),
    'DISTR':            ('MEDIUM', 'Smart Money Distribution — Institutions exiting.'),
    'STRONG_BUY_ML':    ('HIGH',   'ML Ensemble: STRONG BUY signal ({conf:.0f}% confidence).'),
    'STRONG_SELL_ML':   ('HIGH',   'ML Ensemble: STRONG SELL signal ({conf:.0f}% confidence).'),
}


def _a(sym, dt, atype, close, trigger=None, conf=0):
    sev, msg_tpl = ALERT_TYPES.get(atype, ('LOW', '{val}'))
    msg = msg_tpl.format(val=trigger or 0, conf=(conf or 0)*100)
    return {
        'symbol':      sym,
        'date':        str(dt),
        'alert_type':  atype,
        'severity':    sev,
        'message':     f'{sym}: {msg}',
        'close_price': close,
        'trigger_val': float(trigger) if trigger is not None else None,
    }


def generate_alerts(rows: list[dict]) -> list[dict]:
    """
    Check each enriched row against all alert rules.
    Returns list of alert dicts ready for DB insertion.
    """
    alerts = []

    for row in rows:
        sym   = row.get('symbol', '')
        close = row.get('close')
        dt    = date.fromisoformat(str(row.get('date', date.today())))

        def v(k, default=None):
            val = row.get(k)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        rsi   = v('rsi', 50)
        macd  = v('macd', 0)
        msig  = v('macd_signal', 0)
        e20   = v('ema20')
        e50   = v('ema50')
        vr    = v('volume_ratio', 1.0) # Corrected from vol_ratio to match indicators.py
        dp    = v('diff_pct', 0)
        h52   = v('high_52w')
        l52   = v('low_52w')
        sm    = str(row.get('smart_money') or '')
        ml_s  = str(row.get('signal') or 'HOLD') # Corrected from ml_signal to match indicators/pipeline
        ml_c  = v('ml_confidence', 0)
        ml_sc = v('ml_score', 0)

        # RSI
        if rsi <= 20:   alerts.append(_a(sym, dt, 'RSI_EXTREME_OS', close, rsi))
        elif rsi <= 30: alerts.append(_a(sym, dt, 'RSI_OVERSOLD',   close, rsi))
        elif rsi >= 80: alerts.append(_a(sym, dt, 'RSI_EXTREME_OB', close, rsi))
        elif rsi >= 70: alerts.append(_a(sym, dt, 'RSI_OVERBOUGHT', close, rsi))

        # MACD crossover (need prev row — skip for now, done in pipeline)

        # EMA cross
        if e20 and e50:
            if e20 > e50: alerts.append(_a(sym, dt, 'GOLDEN_CROSS', close, e20))

        # Volume spike
        if vr > 2.5:
            atype = 'VOL_SPIKE_UP' if dp > 0 else 'VOL_SPIKE_DOWN'
            alerts.append(_a(sym, dt, atype, close, vr))

        # 52W breaks
        if h52 and close and close >= h52 * 0.998:
            alerts.append(_a(sym, dt, '52W_HIGH_BREAK', close, h52))
        if l52 and close and close <= l52 * 1.002:
            alerts.append(_a(sym, dt, '52W_LOW_BREAK',  close, l52))

        # Smart money
        if sm == 'Accumulation':
            alerts.append(_a(sym, dt, 'ACCUM', close))
        elif sm == 'Distribution':
            alerts.append(_a(sym, dt, 'DISTR', close))

        # ML strong signals (Score and confidence)
        # Note: ML signals in this system use 'signal' (BUY/SELL) and 'score' (confidence value)
        if ml_s == 'BUY':
             # Here we map ml_score or confidence if available
             alerts.append(_a(sym, dt, 'STRONG_BUY_ML',  close, ml_sc, ml_c))
        elif ml_s == 'SELL':
             alerts.append(_a(sym, dt, 'STRONG_SELL_ML', close, ml_sc, ml_c))

    return alerts


def store_alerts_db(alerts: list[dict]):
    from db import insert_alerts
    if alerts:
        insert_alerts(alerts)
        log.info(f'Stored {len(alerts)} alerts')

def process_alerts(df):
    """
    Convenience wrapper for the pipeline.
    """
    if df.empty: return
    records = df.to_dict(orient='records')
    alerts = generate_alerts(records)
    if alerts:
        store_alerts_db(alerts)
        log.info(f"Generated {len(alerts)} alerts")
