"""
NEPSE Analytics Pro v3 — Portfolio Optimizer
Sharpe ratio optimization, sector diversification, correlation filtering.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger('nepse.portfolio')


def compute_portfolio_metrics(df: pd.DataFrame) -> dict:
    """
    Compute portfolio-level risk metrics from multi-symbol time-series data.
    Expects df with columns: symbol, date, close, ret_1d.
    """
    if df.empty or 'symbol' not in df.columns:
        return {}

    # Pivot to get returns matrix
    pivot = df.pivot_table(index='date', columns='symbol', values='ret_1d')
    pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.5))  # Drop symbols with >50% NaN

    if pivot.empty:
        return {}

    # Correlation matrix
    corr_matrix = pivot.corr()

    # Mean returns and volatility
    mean_ret = pivot.mean()
    vol = pivot.std()

    # Sharpe Ratio (assuming risk-free = 5% annual / ~0.02% daily)
    rf_daily = 0.02
    sharpe = ((mean_ret - rf_daily) / vol.replace(0, np.nan)).round(4)

    # Sortino (downside deviation)
    downside = pivot.clip(upper=0).std()
    sortino = ((mean_ret - rf_daily) / downside.replace(0, np.nan)).round(4)

    return {
        'returns': mean_ret.to_dict(),
        'volatility': vol.to_dict(),
        'sharpe': sharpe.to_dict(),
        'sortino': sortino.to_dict(),
        'correlation': corr_matrix.to_dict(),
        'n_symbols': len(pivot.columns),
        'n_days': len(pivot),
    }


def optimize_portfolio(df: pd.DataFrame, recs: pd.DataFrame,
                       budget: float = 1_000_000,
                       max_per_stock: float = 0.15,
                       max_per_sector: float = 0.30) -> pd.DataFrame:
    """
    Build optimal portfolio:
    - Filter by BUY/STRONG_BUY signals
    - Rank by Sharpe * Confidence
    - Diversify across sectors
    - Cap per-stock and per-sector exposure
    """
    if recs.empty or df.empty:
        return pd.DataFrame()

    buys = recs[recs['signal'].isin(['STRONG_BUY', 'BUY'])].copy()
    if buys.empty:
        return pd.DataFrame()

    # Compute per-symbol metrics
    metrics = []
    for _, rec in buys.iterrows():
        sym = rec['symbol']
        sym_data = df[df['symbol'] == sym]
        if sym_data.empty or len(sym_data) < 5:
            continue

        returns = sym_data['close'].pct_change().dropna()
        if returns.empty:
            continue

        mean_ret = returns.mean() * 100
        vol = returns.std() * 100
        sharpe = (mean_ret - 0.02) / vol if vol > 0 else 0

        sector = sym_data.iloc[-1].get('sector', 'Unknown')
        metrics.append({
            'symbol': sym,
            'sector': sector,
            'signal': rec['signal'],
            'confidence': rec['confidence'],
            'entry': rec['entry_price'],
            'target': rec['target_price'],
            'stop_loss': rec['stop_loss'],
            'rr_ratio': rec['rr_ratio'],
            'sharpe': round(sharpe, 4),
            'mean_daily_ret': round(mean_ret, 4),
            'volatility': round(vol, 4),
            'risk_level': rec.get('risk_level', 'MEDIUM'),
            # Composite score: Sharpe * Confidence * R:R
            'composite': round(sharpe * rec['confidence'] * max(rec['rr_ratio'], 0.5), 4),
        })

    if not metrics:
        return pd.DataFrame()

    portfolio = pd.DataFrame(metrics)
    portfolio = portfolio.sort_values('composite', ascending=False)

    # ── Diversification constraints
    allocated = {}  # sector -> total allocated
    selected = []
    total_allocated = 0

    for _, stock in portfolio.iterrows():
        sector = stock['sector']
        entry = stock['entry']
        if entry <= 0:
            continue

        # Max per stock
        max_stock_alloc = budget * max_per_stock
        # Max per sector
        sector_used = allocated.get(sector, 0)
        max_sector_alloc = budget * max_per_sector - sector_used

        alloc = min(max_stock_alloc, max_sector_alloc, budget - total_allocated)
        if alloc < entry:  # Can't afford even 1 share
            continue

        shares = int(alloc / entry)
        capital = round(shares * entry, 2)

        stock_dict = stock.to_dict()
        stock_dict['shares'] = shares
        stock_dict['capital'] = capital
        stock_dict['weight_pct'] = round(capital / budget * 100, 2)
        selected.append(stock_dict)

        allocated[sector] = allocated.get(sector, 0) + capital
        total_allocated += capital

        if total_allocated >= budget * 0.95:
            break

    result = pd.DataFrame(selected)
    if not result.empty:
        result['total_budget'] = budget
        result['cash_remaining'] = round(budget - total_allocated, 2)
    return result


def get_correlation_pairs(df: pd.DataFrame, threshold: float = 0.7) -> list[dict]:
    """Find highly correlated stock pairs (to avoid overexposure)."""
    if df.empty or 'symbol' not in df.columns:
        return []

    pivot = df.pivot_table(index='date', columns='symbol', values='ret_1d')
    pivot = pivot.dropna(axis=1, thresh=int(len(pivot) * 0.3))

    if pivot.shape[1] < 2:
        return []

    corr = pivot.corr()
    pairs = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({
                    'stock_a': corr.columns[i],
                    'stock_b': corr.columns[j],
                    'correlation': round(val, 4),
                    'type': 'positive' if val > 0 else 'negative'
                })

    return sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)


def get_risk_dashboard(df: pd.DataFrame) -> dict:
    """Compute portfolio risk metrics for the risk dashboard."""
    if df.empty:
        return {}

    latest = df.sort_values('date').groupby('symbol').tail(1)

    # Most volatile
    if 'volatility_20d' in latest.columns:
        vol_cols = [c for c in ['symbol', 'close', 'volatility_20d', 'drawdown_pct', 'atr'] if c in latest.columns]
        most_volatile = latest.nlargest(10, 'volatility_20d')[vol_cols].to_dict(orient='records')
    else:
        most_volatile = []

    # Biggest drawdowns
    if 'drawdown_pct' in latest.columns:
        dd_cols = [c for c in ['symbol', 'close', 'drawdown_pct', 'max_drawdown_20d'] if c in latest.columns]
        biggest_dd = latest.nsmallest(10, 'drawdown_pct')[dd_cols].to_dict(orient='records')
    else:
        biggest_dd = []

    # Avg market metrics
    avg_vol = latest['volatility_20d'].mean() if 'volatility_20d' in latest.columns else 0
    avg_dd = latest['drawdown_pct'].mean() if 'drawdown_pct' in latest.columns else 0

    return {
        'most_volatile': most_volatile,
        'biggest_drawdowns': biggest_dd,
        'avg_market_volatility': round(avg_vol, 4) if avg_vol else 0,
        'avg_market_drawdown': round(avg_dd, 4) if avg_dd else 0,
        'total_stocks': len(latest),
    }
