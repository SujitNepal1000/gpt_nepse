"""
NEPSE Analytics Pro v3 — Sector & Market Intelligence
Sector momentum ranking, capital rotation detection, market sentiment analysis.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger('nepse.sector')


def compute_sector_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive sector-level metrics:
    - Average return, volatility
    - Total volume, turnover
    - Momentum score
    - Capital flow direction
    """
    if df.empty or 'sector' not in df.columns:
        return pd.DataFrame()

    latest = df.sort_values('date').groupby('symbol').tail(1)

    sector_stats = []
    for sector, group in latest.groupby('sector'):
        n_stocks = len(group)
        avg_ret_1d = group['ret_1d'].mean() if 'ret_1d' in group.columns else 0
        avg_ret_5d = group['ret_5d'].mean() if 'ret_5d' in group.columns else 0
        avg_ret_20d = group['ret_20d'].mean() if 'ret_20d' in group.columns else 0
        avg_vol = group['volatility_20d'].mean() if 'volatility_20d' in group.columns else 0
        total_volume = group['volume'].sum() if 'volume' in group.columns else 0
        total_turnover = group['turnover'].sum() if 'turnover' in group.columns else 0
        avg_rsi = group['rsi'].mean() if 'rsi' in group.columns else 50

        # Momentum score: weighted returns
        momentum = (avg_ret_1d * 0.1 + avg_ret_5d * 0.3 + avg_ret_20d * 0.6)

        # Gainers vs losers
        if 'diff_pct' in group.columns:
            gainers = len(group[group['diff_pct'] > 0])
            losers = len(group[group['diff_pct'] < 0])
        else:
            gainers = losers = 0
        breadth = gainers / max(gainers + losers, 1) * 100

        # Smart money signal
        if 'smart_money' in group.columns:
            accum_pct = len(group[group['smart_money'] == 'Accumulation']) / max(n_stocks, 1) * 100
        else:
            accum_pct = 0

        # Trend direction
        if avg_ret_20d > 2 and avg_ret_5d > 0:
            trend = 'Strong Uptrend'
        elif avg_ret_20d > 0:
            trend = 'Uptrend'
        elif avg_ret_20d < -2 and avg_ret_5d < 0:
            trend = 'Strong Downtrend'
        elif avg_ret_20d < 0:
            trend = 'Downtrend'
        else:
            trend = 'Sideways'

        sector_stats.append({
            'sector': sector,
            'stocks': n_stocks,
            'avg_ret_1d': round(avg_ret_1d, 4),
            'avg_ret_5d': round(avg_ret_5d, 4),
            'avg_ret_20d': round(avg_ret_20d, 4),
            'momentum_score': round(momentum, 4),
            'avg_volatility': round(avg_vol, 4),
            'total_volume': int(total_volume),
            'total_turnover': round(total_turnover, 2),
            'avg_rsi': round(avg_rsi, 2),
            'breadth_pct': round(breadth, 2),
            'accumulation_pct': round(accum_pct, 2),
            'trend': trend,
            'gainers': gainers,
            'losers': losers,
        })

    result = pd.DataFrame(sector_stats)
    if not result.empty:
        result = result.sort_values('momentum_score', ascending=False)
        result['rank'] = range(1, len(result) + 1)
    return result


def detect_capital_rotation(df: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """
    Detect sector capital rotation by comparing recent vs historical volume flows.
    Rising volume + rising price = capital inflow.
    Rising volume + falling price = capital outflow.
    """
    if df.empty or 'sector' not in df.columns or 'date' not in df.columns:
        return []

    dates = sorted(df['date'].unique())
    if len(dates) < lookback:
        lookback = max(len(dates) // 2, 2)

    recent_dates = dates[-lookback:]
    prior_dates = dates[-lookback*2:-lookback] if len(dates) >= lookback * 2 else dates[:lookback]

    recent = df[df['date'].isin(recent_dates)]
    prior = df[df['date'].isin(prior_dates)]

    rotations = []
    for sector in df['sector'].unique():
        r = recent[recent['sector'] == sector]
        p = prior[prior['sector'] == sector]

        if r.empty or p.empty:
            continue

        r_vol = r['volume'].mean()
        p_vol = p['volume'].mean()
        r_ret = r['ret_1d'].mean() if 'ret_1d' in r.columns else 0
        p_ret = p['ret_1d'].mean() if 'ret_1d' in p.columns else 0

        vol_change = ((r_vol - p_vol) / max(p_vol, 1) * 100)

        if vol_change > 20 and r_ret > 0:
            flow = 'INFLOW'
        elif vol_change > 20 and r_ret < 0:
            flow = 'OUTFLOW'
        elif vol_change < -20:
            flow = 'DECLINING'
        else:
            flow = 'STABLE'

        rotations.append({
            'sector': sector,
            'volume_change_pct': round(vol_change, 2),
            'recent_return': round(r_ret, 4),
            'prior_return': round(p_ret, 4),
            'capital_flow': flow,
        })

    return sorted(rotations, key=lambda x: x['volume_change_pct'], reverse=True)


def get_market_sentiment(df: pd.DataFrame) -> dict:
    """
    Overall market sentiment analysis.
    Returns: Bullish / Bearish / Sideways + supporting metrics.
    """
    if df.empty:
        return {'sentiment': 'Unknown', 'score': 0}

    latest = df.sort_values('date').groupby('symbol').tail(1)

    # Basic metrics
    total = len(latest)
    if 'diff_pct' in latest.columns:
        gainers = len(latest[latest['diff_pct'] > 0])
        losers = len(latest[latest['diff_pct'] < 0])
    else:
        gainers = losers = 0
    avg_ret = latest['ret_1d'].mean() if 'ret_1d' in latest.columns else 0
    avg_rsi = latest['rsi'].mean() if 'rsi' in latest.columns else 50

    # Buy vs sell signals
    if 'signal' in latest.columns:
        buys = len(latest[latest['signal'] == 'BUY'])
        sells = len(latest[latest['signal'] == 'SELL'])
    else:
        buys = sells = 0


    # Sentiment score (-100 to +100)
    score = 0
    reasons = []

    # Breadth
    breadth = gainers / max(gainers + losers, 1)
    if breadth > 0.65:
        score += 25; reasons.append(f'Broad advance ({breadth*100:.0f}% stocks up)')
    elif breadth < 0.35:
        score -= 25; reasons.append(f'Broad decline ({(1-breadth)*100:.0f}% stocks down)')

    # Average RSI
    if avg_rsi > 60:
        score += 15; reasons.append(f'Avg RSI bullish ({avg_rsi:.0f})')
    elif avg_rsi < 40:
        score -= 15; reasons.append(f'Avg RSI bearish ({avg_rsi:.0f})')

    # ML signal balance
    if buys > sells * 2:
        score += 20; reasons.append(f'ML: {buys} buys vs {sells} sells')
    elif sells > buys * 2:
        score -= 20; reasons.append(f'ML: {sells} sells vs {buys} buys')

    # Average return
    if avg_ret > 1:
        score += 20; reasons.append(f'Strong avg return ({avg_ret:.2f}%)')
    elif avg_ret < -1:
        score -= 20; reasons.append(f'Negative avg return ({avg_ret:.2f}%)')

    # Classify
    if score >= 30:
        sentiment = 'Bullish'
    elif score >= 10:
        sentiment = 'Mildly Bullish'
    elif score <= -30:
        sentiment = 'Bearish'
    elif score <= -10:
        sentiment = 'Mildly Bearish'
    else:
        sentiment = 'Sideways'

    return {
        'sentiment': sentiment,
        'score': score,
        'reasons': reasons,
        'gainers': gainers,
        'losers': losers,
        'total': total,
        'avg_return': round(avg_ret, 4),
        'avg_rsi': round(avg_rsi, 2),
        'buy_signals': buys,
        'sell_signals': sells,
        'breadth_pct': round(breadth * 100, 2),
    }
