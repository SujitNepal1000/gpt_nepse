import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Mock data
data = []
for i in range(30):
    dt = datetime.now().date() - timedelta(days=30-i)
    for sym, sec, base in [('NABIL','Banking',500), ('NTC','Others',800)]:
        p = base + i * 2
        data.append({
            'symbol': sym, 'sector': sec,
            'close': p, 'open': p-3, 'high': p+6, 'low': p-6,
            'volume': 10000, 'date': dt, 'ltp': p,
            'turnover': p * 1000, 'diff_pct': 1.5
        })
df = pd.DataFrame(data)
print(f"Mock: {len(df)} rows")

from analytics_engine import clean_dataframe, compute_derived_features, compute_cross_sectional
df = clean_dataframe(df)
print(f"Clean: {len(df)} rows")

parts = []
for sym in df['symbol'].unique():
    sdf = compute_derived_features(df[df['symbol'] == sym].copy())
    parts.append(sdf)
    print(f"  {sym}: {sdf.shape[1]} features")

all_data = pd.concat(parts, ignore_index=True)
all_data = compute_cross_sectional(all_data)
print(f"Cross-sectional: {all_data.shape[1]} features")

from strategy import classify_signal, generate_recommendations, get_top_opportunities
latest = all_data.sort_values('date').groupby('symbol').tail(1)
recs = generate_recommendations(latest)
print(f"Recs: {len(recs)}, Signals: {recs['signal'].value_counts().to_dict()}")

from portfolio import get_risk_dashboard
risk = get_risk_dashboard(all_data)
print(f"Risk: {risk.get('total_stocks',0)} stocks")

from sector_analysis import compute_sector_metrics, get_market_sentiment
sectors = compute_sector_metrics(all_data)
sentiment = get_market_sentiment(all_data)
print(f"Sectors: {len(sectors)}, Sentiment: {sentiment['sentiment']}")

print("ALL TESTS PASSED")
