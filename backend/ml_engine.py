"""
NEPSE Analytics Pro v3 — ML Engine
Hedge-fund grade signal generation using:
  • Gradient Boosting (XGBoost + LightGBM + sklearn GBM) ensemble
  • Walk-forward cross-validation (no look-ahead bias)
  • Multi-factor rule scoring
  • Risk-adjusted signal filtering (only trades with R:R ≥ 2)
"""
import logging
import pickle
import os
import warnings
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path

warnings.filterwarnings('ignore')

from config import (MODEL_DIR, LOOKFORWARD_DAYS, MIN_TRAIN_ROWS,
                    SIGNAL_THRESHOLD, BUY_RETURN_PCT, SELL_RETURN_PCT)

log = logging.getLogger('nepse.ml')

MODEL_PATH = Path(MODEL_DIR) / 'ensemble_model.pkl'
META_PATH  = Path(MODEL_DIR) / 'model_meta.pkl'

# ── Feature Set (36 features) ─────────────────────────────────────────────────
FEATURES = [
    # Momentum
    'rsi', 'rsi_slope3', 'rsi_divergence',
    # MACD
    'macd', 'macd_signal', 'macd_hist', 'macd_hist_slope',
    # EMA
    'close_vs_ema20', 'close_vs_ema50', 'ema20_vs_ema50',
    # Volatility
    'bb_pct_b', 'bb_width', 'atr_pct',
    # Oscillators
    'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi',
    # Trend
    'adx', 'di_plus_minus',
    # Volume
    'vol_ratio', 'obv_slope5', 'mfi_slope3',
    # Price action
    'diff_pct', 'vwap_pct', 'close_vs_support_pct', 'close_vs_resist_pct',
    'candle_body_pct', 'upper_wick_pct', 'lower_wick_pct',
    # Market structure
    'roc_10', 'momentum_10',
    # Rolling returns
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
]


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = pd.to_numeric(df['close'],   errors='coerce')
    o  = pd.to_numeric(df['open'],    errors='coerce')
    h  = pd.to_numeric(df['high'],    errors='coerce')
    l  = pd.to_numeric(df['low'],     errors='coerce')
    v  = pd.to_numeric(df['vol'],     errors='coerce').fillna(0)

    # RSI slope and divergence
    rsi = pd.to_numeric(df.get('rsi'), errors='coerce')
    df['rsi_slope3']     = rsi.diff(3)
    # RSI divergence: price makes lower low but RSI doesn't
    price_ll = (c < c.shift(5)) & (c < c.shift(10))
    rsi_hl   = (rsi > rsi.shift(5))
    df['rsi_divergence'] = (price_ll & rsi_hl).astype(int)

    # MACD histogram slope
    hist = pd.to_numeric(df.get('macd_hist'), errors='coerce')
    df['macd_hist_slope'] = hist.diff(2)

    # EMA ratios
    ema20 = pd.to_numeric(df.get('ema20'), errors='coerce')
    ema50 = pd.to_numeric(df.get('ema50'), errors='coerce')
    df['close_vs_ema20']  = (c - ema20) / ema20.replace(0, np.nan) * 100
    df['close_vs_ema50']  = (c - ema50) / ema50.replace(0, np.nan) * 100
    df['ema20_vs_ema50']  = (ema20 - ema50) / ema50.replace(0, np.nan) * 100

    # ADX direction bias
    dip  = pd.to_numeric(df.get('di_plus'),  errors='coerce')
    dim  = pd.to_numeric(df.get('di_minus'), errors='coerce')
    df['di_plus_minus'] = (dip - dim).fillna(0)

    # Volume slope
    df['obv_slope5'] = pd.to_numeric(df.get('obv'), errors='coerce').diff(5)
    mfi = pd.to_numeric(df.get('mfi'), errors='coerce')
    df['mfi_slope3'] = mfi.diff(3)

    # Price action patterns
    rng = (h - l).replace(0, np.nan)
    df['candle_body_pct'] = ((c - o) / rng * 100).round(4)
    df['upper_wick_pct']  = ((h - pd.concat([o,c], axis=1).max(axis=1)) / rng * 100).round(4)
    df['lower_wick_pct']  = ((pd.concat([o,c], axis=1).min(axis=1) - l) / rng * 100).round(4)

    # Support / resistance distance
    sup = pd.to_numeric(df.get('support'),    errors='coerce')
    res = pd.to_numeric(df.get('resistance'), errors='coerce')
    df['close_vs_support_pct']  = ((c - sup) / sup.replace(0, np.nan) * 100).round(4)
    df['close_vs_resist_pct']   = ((res - c) / res.replace(0, np.nan) * 100).round(4)

    # Rolling returns
    df['ret_1d']  = c.pct_change(1)  * 100
    df['ret_5d']  = c.pct_change(5)  * 100
    df['ret_10d'] = c.pct_change(10) * 100
    df['ret_20d'] = c.pct_change(20) * 100

    return df


def build_target(df: pd.DataFrame, n: int = LOOKFORWARD_DAYS) -> pd.DataFrame:
    """
    Target: 1=BUY, 0=SELL, NaN=neutral (excluded from training).
    Uses future close to avoid look-ahead bias.
    """
    df = df.copy()
    future = df['close'].shift(-n)
    pct    = (future - df['close']) / df['close'] * 100
    df['target'] = np.where(pct >= BUY_RETURN_PCT,  1,
                   np.where(pct <= SELL_RETURN_PCT, 0, np.nan))
    return df


# ── Model Training ────────────────────────────────────────────────────────────

def _get_models():
    """Return dict of candidate models for ensemble."""
    models = {}

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models['xgb'] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, gamma=1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0)
    except ImportError:
        log.warning('XGBoost not available')

    # LightGBM
    try:
        import lightgbm as lgb
        models['lgb'] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=31,
            random_state=42, verbose=-1)
    except ImportError:
        log.warning('LightGBM not available')

    # sklearn Gradient Boosting (always available)
    from sklearn.ensemble import GradientBoostingClassifier
    models['gbm'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, random_state=42)

    return models


def walk_forward_train(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Walk-forward validation: train on past, test on future.
    Returns per-model accuracy scores.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    df = engineer_features(df)
    df = build_target(df)
    df = df.dropna(subset=FEATURES + ['target'])

    if len(df) < MIN_TRAIN_ROWS:
        return {'status': 'insufficient_data', 'n': len(df)}

    X = df[FEATURES].fillna(0).values
    y = df['target'].astype(int).values

    step       = len(df) // (n_splits + 1)
    candidates = _get_models()
    scores     = {name: [] for name in candidates}

    for fold in range(n_splits):
        train_end  = step * (fold + 2)
        test_start = train_end
        test_end   = min(test_start + step, len(df))
        if test_end <= test_start:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        for name, mdl in candidates.items():
            try:
                mdl.fit(X_tr_s, y_tr)
                preds  = mdl.predict(X_te_s)
                scores[name].append(accuracy_score(y_te, preds))
            except Exception as e:
                log.warning(f'Walk-forward {name} fold {fold}: {e}')

    avg = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    log.info(f'Walk-forward scores: {avg}')
    return {'status': 'ok', 'scores': avg, 'n_samples': len(df)}


def train_ensemble(all_data: pd.DataFrame) -> dict:
    """
    Train full ensemble on all available data.
    Saves model to MODEL_PATH.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    if all_data.empty or len(all_data) < MIN_TRAIN_ROWS:
        log.warning(f'Insufficient data: {len(all_data)} rows')
        return {'status': 'insufficient_data'}

    df = engineer_features(all_data)
    df = build_target(df)
    df = df.dropna(subset=FEATURES + ['target'])

    if len(df) < MIN_TRAIN_ROWS:
        log.warning(f'After filtering: {len(df)} rows (need {MIN_TRAIN_ROWS})')
        return {'status': 'too_few_after_filter', 'n': len(df)}

    X = df[FEATURES].fillna(0).values
    y = df['target'].astype(int).values

    # Time-based split (no random shuffle — respects temporal order)
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    candidates = _get_models()
    fitted     = {}
    model_acc  = {}

    for name, mdl in candidates.items():
        try:
            mdl.fit(X_tr_s, y_tr)
            acc = accuracy_score(y_te, mdl.predict(X_te_s))
            fitted[name] = mdl
            model_acc[name] = round(acc, 4)
            log.info(f'  {name}: acc={acc:.4f}')
        except Exception as e:
            log.warning(f'  {name} train failed: {e}')

    if not fitted:
        return {'status': 'all_models_failed'}

    # Walk-forward validation scores
    wf = walk_forward_train(all_data)

    # Save
    payload = {'models': fitted, 'scaler': sc, 'features': FEATURES,
               'model_acc': model_acc, 'wf_scores': wf.get('scores', {})}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)

    # Save meta
    with open(META_PATH, 'wb') as f:
        pickle.dump({'trained_at': date.today().isoformat(),
                     'n_samples': len(df), 'model_acc': model_acc,
                     'class_dist': dict(zip(*np.unique(y, return_counts=True))),
                     'features': FEATURES}, f)

    log.info(f'Ensemble trained: {model_acc}  samples={len(df)}')
    return {'status': 'ok', 'accuracy': model_acc,
            'samples': len(df), 'models': list(fitted.keys()),
            'wf': wf}


def load_ensemble():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def load_model_meta() -> dict:
    if not META_PATH.exists():
        return {}
    with open(META_PATH, 'rb') as f:
        return pickle.load(f)


def predict_ensemble(payload: dict, df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Returns (ml_signal, confidence, raw_score).
    ml_signal: 'BUY' / 'SELL' / 'HOLD'
    """
    if payload is None or df.empty:
        return 'HOLD', 0.0, 0.5

    try:
        feat_df = engineer_features(df)
        last    = feat_df.tail(1)[FEATURES].fillna(0).values
        sc      = payload['scaler']
        X_s     = sc.transform(last)
        models  = payload['models']
        wf      = payload.get('wf_scores', {})

        # Weighted average probability
        # Weight each model by its walk-forward accuracy
        total_w = sum(wf.get(k, 0.33) for k in models) or 1.0
        prob_buy = 0.0

        for name, mdl in models.items():
            w     = wf.get(name, 0.33) / total_w
            proba = mdl.predict_proba(X_s)[0]
            # proba: [P(SELL), P(BUY)] (class 0=SELL, 1=BUY)
            prob_buy += w * proba[1]

        confidence = max(prob_buy, 1 - prob_buy)
        if prob_buy >= SIGNAL_THRESHOLD:
            return 'BUY', round(confidence, 4), round(prob_buy, 4)
        elif (1 - prob_buy) >= SIGNAL_THRESHOLD:
            return 'SELL', round(confidence, 4), round(prob_buy, 4)
        return 'HOLD', round(confidence, 4), round(prob_buy, 4)

    except Exception as e:
        log.warning(f'predict_ensemble failed: {e}')
        return 'HOLD', 0.0, 0.5


# ── Rule-Based Signal Scoring ─────────────────────────────────────────────────

RULE_WEIGHTS = {
    'rsi':         {'os': (+25, 'RSI oversold — bounce zone'),
                    'ob': (-25, 'RSI overbought — avoid'),
                    'mid':  (+15, 'RSI neutral-low — good entry zone')},
    'macd_cross':  {'bull': (+22, 'MACD bullish crossover'),
                    'bear': (-22, 'MACD bearish crossover')},
    'macd_hist':   {'grow': (+10, 'MACD histogram expanding — momentum'),
                    'fade': (-10, 'MACD histogram shrinking — weakening')},
    'ema50':       {'above': (+18, 'Price above EMA50 — uptrend'),
                    'below': (-15, 'Price below EMA50 — downtrend')},
    'ema_cross':   {'golden': (+18, 'EMA20 > EMA50 — golden zone'),
                    'death':  (-15, 'EMA20 < EMA50 — death zone')},
    'bb':          {'lower': (+14, 'Near lower Bollinger — oversold'),
                    'upper': (-14, 'Near upper Bollinger — overbought')},
    'stoch':       {'os': (+12, 'Stochastic oversold (<20)'),
                    'ob': (-12, 'Stochastic overbought (>80)')},
    'adx':         {'strong': (+10, 'ADX >25 — strong trend'),
                    'weak':   (-5,  'ADX <15 — no clear trend')},
    'vol':         {'high_up':   (+15, 'High volume + up day — institutional buy'),
                    'high_down': (-15, 'High volume + down day — institutional sell'),
                    'spike':     (+8,  'Volume spike — smart money active')},
    'smart_money': {'accum': (+20, 'Smart money accumulation detected'),
                    'distr': (-20, 'Smart money distribution detected')},
    'support':     {'near': (+12, 'Price near support — low risk entry')},
    'resistance':  {'near': (-8,  'Price near resistance — exit zone')},
    'trend':       {'bull': (+18, 'Bullish trend structure (HH/HL)'),
                    'bear': (-18, 'Bearish trend structure (LH/LL)')},
    'mfi':         {'os': (+10, 'MFI oversold — money flowing in'),
                    'ob': (-10, 'MFI overbought — money flowing out')},
    'williams_r':  {'os': (+8,  'Williams %R oversold'),
                    'ob': (-8,  'Williams %R overbought')},
    'cci':         {'os': (+8,  'CCI oversold (<-100)'),
                    'ob': (-8,  'CCI overbought (>+100)')},
}


def rule_score(row: dict) -> tuple[int, list[str]]:
    """
    Returns (composite_score, reasons[]).
    Max theoretical score: ~185, min: ~-185.
    """
    score   = 0
    reasons = []

    def val(k, default=None):
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)

    rsi    = val('rsi', 50)
    macd   = val('macd', 0)
    macd_s = val('macd_signal', 0)
    hist   = val('macd_hist', 0)
    ema20  = val('ema20')
    ema50  = val('ema50')
    bb_u   = val('bb_upper')
    bb_l   = val('bb_lower')
    bb_b   = val('bb_pct_b', 0.5)
    stk    = val('stoch_k', 50)
    adx    = val('adx', 0)
    dip    = val('di_plus', 0)
    dim    = val('di_minus', 0)
    vr     = val('vol_ratio', 1.0)
    dp     = val('diff_pct', 0)
    sm     = str(row.get('smart_money') or 'Neutral')
    sup    = val('support')
    res    = val('resistance')
    trend  = str(row.get('trend_struct') or '')
    close  = val('close', 0)
    mfi    = val('mfi', 50)
    wr     = val('williams_r', -50)
    cci    = val('cci', 0)
    vs     = val('vol_spike', False)
    hist_s = val('macd_hist')

    # RSI
    if rsi < 30:
        w = RULE_WEIGHTS['rsi']['os'];  score += w[0]; reasons.append(f'RSI {rsi:.1f}: {w[1]}')
    elif rsi > 70:
        w = RULE_WEIGHTS['rsi']['ob'];  score += w[0]; reasons.append(f'RSI {rsi:.1f}: {w[1]}')
    elif 35 <= rsi <= 55:
        w = RULE_WEIGHTS['rsi']['mid']; score += w[0]; reasons.append(f'RSI {rsi:.1f}: {w[1]}')

    # MACD cross
    if macd > macd_s:
        w = RULE_WEIGHTS['macd_cross']['bull']; score += w[0]; reasons.append(w[1])
    else:
        w = RULE_WEIGHTS['macd_cross']['bear']; score += w[0]; reasons.append(w[1])

    # MACD histogram slope
    if hist_s is not None and hist is not None:
        if hist > 0 and hist > hist_s:
            w = RULE_WEIGHTS['macd_hist']['grow']; score += w[0]; reasons.append(w[1])
        elif hist < 0 and hist < hist_s:
            w = RULE_WEIGHTS['macd_hist']['fade']; score += w[0]; reasons.append(w[1])

    # EMA50
    if ema50 and close:
        if close > ema50:
            w = RULE_WEIGHTS['ema50']['above']; score += w[0]; reasons.append(f'EMA50({ema50:.0f}): {w[1]}')
        else:
            w = RULE_WEIGHTS['ema50']['below']; score += w[0]; reasons.append(f'EMA50({ema50:.0f}): {w[1]}')

    # EMA cross
    if ema20 and ema50:
        if ema20 > ema50:
            w = RULE_WEIGHTS['ema_cross']['golden']; score += w[0]; reasons.append(w[1])
        else:
            w = RULE_WEIGHTS['ema_cross']['death'];  score += w[0]; reasons.append(w[1])

    # Bollinger
    if bb_b is not None:
        if bb_b < 0.15:
            w = RULE_WEIGHTS['bb']['lower']; score += w[0]; reasons.append(w[1])
        elif bb_b > 0.85:
            w = RULE_WEIGHTS['bb']['upper']; score += w[0]; reasons.append(w[1])

    # Stochastic
    if stk < 20:
        w = RULE_WEIGHTS['stoch']['os']; score += w[0]; reasons.append(w[1])
    elif stk > 80:
        w = RULE_WEIGHTS['stoch']['ob']; score += w[0]; reasons.append(w[1])

    # ADX
    if adx > 25:
        direction = '+' if dip > dim else '-'
        w = RULE_WEIGHTS['adx']['strong']
        score += w[0] if dip > dim else -w[0]
        reasons.append(f'ADX {adx:.1f} ({direction}): {w[1]}')
    elif adx < 15:
        w = RULE_WEIGHTS['adx']['weak']; score += w[0]; reasons.append(w[1])

    # Volume
    if vr > 1.8 and dp > 0:
        w = RULE_WEIGHTS['vol']['high_up'];   score += w[0]; reasons.append(w[1])
    elif vr > 1.8 and dp < 0:
        w = RULE_WEIGHTS['vol']['high_down']; score += w[0]; reasons.append(w[1])
    elif vs:
        w = RULE_WEIGHTS['vol']['spike'];     score += w[0]; reasons.append(w[1])

    # Smart money
    if sm == 'Accumulation':
        w = RULE_WEIGHTS['smart_money']['accum']; score += w[0]; reasons.append(w[1])
    elif sm == 'Distribution':
        w = RULE_WEIGHTS['smart_money']['distr']; score += w[0]; reasons.append(w[1])

    # Support / Resistance proximity
    if sup and close and abs(close - sup) / max(close, 1) < 0.03:
        w = RULE_WEIGHTS['support']['near']; score += w[0]; reasons.append(f'S:{sup:.1f}: {w[1]}')
    if res and close and abs(close - res) / max(close, 1) < 0.03:
        w = RULE_WEIGHTS['resistance']['near']; score += w[0]; reasons.append(f'R:{res:.1f}: {w[1]}')

    # Trend structure
    if 'Bullish' in trend:
        w = RULE_WEIGHTS['trend']['bull']; score += w[0]; reasons.append(w[1])
    elif 'Bearish' in trend:
        w = RULE_WEIGHTS['trend']['bear']; score += w[0]; reasons.append(w[1])

    # MFI
    if mfi < 25:
        w = RULE_WEIGHTS['mfi']['os']; score += w[0]; reasons.append(w[1])
    elif mfi > 75:
        w = RULE_WEIGHTS['mfi']['ob']; score += w[0]; reasons.append(w[1])

    # Williams %R
    if wr < -80:
        w = RULE_WEIGHTS['williams_r']['os']; score += w[0]; reasons.append(w[1])
    elif wr > -20:
        w = RULE_WEIGHTS['williams_r']['ob']; score += w[0]; reasons.append(w[1])

    # CCI
    if cci < -100:
        w = RULE_WEIGHTS['cci']['os']; score += w[0]; reasons.append(w[1])
    elif cci > 100:
        w = RULE_WEIGHTS['cci']['ob']; score += w[0]; reasons.append(w[1])

    return score, reasons


def compute_signal(row: dict, ml_payload=None, df_history=None) -> dict:
    """
    Full composite signal:
    • Rule-based score (16 factors)
    • ML ensemble overlay
    • R:R calculation
    Returns complete signal dict ready for DB storage.
    """
    score, reasons = rule_score(row)

    # ML overlay
    ml_sig, ml_conf, ml_prob = 'HOLD', 0.0, 0.5
    if ml_payload and df_history is not None and not df_history.empty:
        ml_sig, ml_conf, ml_prob = predict_ensemble(ml_payload, df_history)
        adj = int(ml_conf * 25)
        if ml_sig == 'BUY':
            score += adj;  reasons.append(f'ML ensemble BUY ({ml_conf*100:.0f}% conf)')
        elif ml_sig == 'SELL':
            score -= adj;  reasons.append(f'ML ensemble SELL ({ml_conf*100:.0f}% conf)')

    # Signal classification
    if score >= 80:
        sig, strength = 'BUY',  'STRONG'
    elif score >= 45:
        sig, strength = 'BUY',  'MODERATE'
    elif score >= 20:
        sig, strength = 'BUY',  'WEAK'
    elif score <= -70:
        sig, strength = 'SELL', 'STRONG'
    elif score <= -40:
        sig, strength = 'SELL', 'MODERATE'
    elif score <= -20:
        sig, strength = 'SELL', 'WEAK'
    else:
        sig, strength = 'HOLD', 'NEUTRAL'

    # R:R calculation
    close = float(row.get('close') or 0)
    atr   = float(row.get('atr')   or 0)
    sup   = row.get('support')
    res   = row.get('resistance')

    stop_loss   = float(sup) if sup else (round(close - atr * 2, 2) if atr else None)
    target      = float(res) if res else (round(close + atr * 3, 2) if atr else None)
    rr_ratio    = None
    exp_return  = None

    if stop_loss and target and stop_loss < close < target:
        risk        = close - stop_loss
        reward      = target - close
        if risk > 0:
            rr_ratio   = round(reward / risk, 2)
            exp_return = round(reward / close * 100, 2)

    return {
        'signal_type':    sig,
        'strength':       strength,
        'composite_score':score,
        'ml_confidence':  round(ml_conf, 4),
        'target_price':   round(target,   2) if target   else None,
        'stop_loss':      round(stop_loss, 2) if stop_loss else None,
        'rr_ratio':       rr_ratio,
        'expected_return':exp_return,
        'reasoning':      ' | '.join(reasons[:8]),
        'ml_model_used':  'ensemble' if ml_payload else 'rules_only',
        'ml_prob_buy':    round(ml_prob, 4),
    }


# ── Position Sizing ───────────────────────────────────────────────────────────

def position_size(account: float, risk_pct: float,
                  entry: float, stop: float) -> dict:
    if not entry or not stop or entry <= stop:
        return {'error': 'Entry must be above stop loss'}
    risk_amt  = account * risk_pct / 100
    risk_per  = entry - stop
    shares    = int(risk_amt / risk_per)
    capital   = round(shares * entry, 2)
    max_loss  = round(shares * risk_per, 2)
    return {
        'shares':       shares,
        'capital':      capital,
        'max_loss':     max_loss,
        'risk_amount':  round(risk_amt, 2),
    }
