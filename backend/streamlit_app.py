"""
NEPSE Analytics Pro v3 — Streamlit Dashboard
Hedge-Fund Grade Analytics & Trading Intelligence
Run: streamlit run streamlit_app.py
"""
import logging
import os
import sys
import warnings
from datetime import date, timedelta, datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='NEPSE Analytics Pro v3',
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#080b12}
[data-testid="stSidebar"]{background:#0f1320}
[data-testid="stHeader"]{background:transparent}
.main .block-container{padding-top:.5rem;max-width:100%}
div[data-testid="metric-container"]{
    background:#111827;border:1px solid rgba(99,120,200,.18);
    border-radius:8px;padding:12px 14px}
div[data-testid="metric-container"] label{
    color:#6b7db3!important;font-size:11px;text-transform:uppercase;letter-spacing:.6px}
.stTabs [data-baseweb="tab-list"]{
    background:#0f1320;border-bottom:1px solid rgba(99,120,200,.2)}
.stTabs [data-baseweb="tab"]{color:#6b7db3}
.stTabs [aria-selected="true"]{color:#4f8ef7;border-bottom-color:#4f8ef7}
.stButton>button{background:#1a2540;border:1px solid rgba(79,142,247,.4);
    color:#4f8ef7;border-radius:6px;font-weight:500}
.stButton>button:hover{background:rgba(79,142,247,.15);border-color:#4f8ef7}
h1,h2,h3{color:#e0e8ff!important}
</style>""", unsafe_allow_html=True)

BG   = '#080b12'
BG2  = '#111827'
GRID = 'rgba(99,120,200,0.06)'
TEXT = '#a0aec0'
PBASE = dict(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(color=TEXT, size=11),
    margin=dict(l=40, r=20, t=35, b=30),
    xaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False, color=TEXT),
    yaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False, color=TEXT),
)


# ── Init ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_app():
    from pipeline import start_scheduler
    sched = start_scheduler()
    return sched

try:
    _sched = init_app()
except Exception as e:
    st.sidebar.warning(f'Scheduler: {e}')


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=90)
def load_snapshot():
    from db import get_latest_snapshot
    return get_latest_snapshot()

@st.cache_data(ttl=300)
def load_symbols():
    from db import get_all_symbols
    return get_all_symbols()

@st.cache_data(ttl=90)
def load_market_hist(days=60):
    from db import get_market_summary_history
    return get_market_summary_history(days)

@st.cache_data(ttl=90)
def load_sym_hist(sym, limit=300):
    from db import get_symbol_history
    return get_symbol_history(sym, limit)

@st.cache_data(ttl=60)
def load_signals(sig_type=None, limit=50):
    from db import get_latest_signals
    return get_latest_signals(sig_type, limit)

@st.cache_data(ttl=30)
def load_alerts(days=7, unread=False):
    from db import get_recent_alerts
    return get_recent_alerts(days, unread)

@st.cache_data(ttl=60)
def load_watchlist():
    from db import get_watchlist
    return get_watchlist()

@st.cache_data(ttl=60)
def load_latest_date():
    from db import get_latest_date
    return get_latest_date()

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_num(val):
    if val is None or np.isnan(val): return '—'
    if val >= 1_000_000: return f'{val/1_000_000:.1f}M'
    if val >= 1_000: return f'{val/1_000:.1f}K'
    return f'{val:,.1f}'

def sig_color(sig):
    if sig in ('BUY', 'STRONG_BUY'): return '#059669'
    if sig == 'SELL': return '#dc2626'
    return '#6b7db3'

@st.cache_data(ttl=900)
def get_sectors():
    from config import SECTORS
    return SECTORS

# ── Charts ────────────────────────────────────────────────────────────────────
def plot_main_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('', 'Volume'),
                        row_heights=[0.7, 0.3])

    df = df.copy()
    df['open'] = pd.to_numeric(df['open'], errors='coerce').fillna(df['close'])
    df['high'] = pd.to_numeric(df['high'], errors='coerce').fillna(df['close'])
    df['low'] = pd.to_numeric(df['low'], errors='coerce').fillna(df['close'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce').fillna(0)

    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='OHLC', increasing_line_color='#22c55e', decreasing_line_color='#ef4444'
    ), row=1, col=1)

    if 'ema20' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ema20'], name='EMA 20', line=dict(color='#3b82f6', width=1)), row=1, col=1)
    if 'ema50' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ema50'], name='EMA 50', line=dict(color='#8b5cf6', width=1)), row=1, col=1)

    colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Vol', marker_color=colors), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, **PBASE)
    fig.update_layout(height=600, showlegend=False)
    return fig

def plot_rsi_macd(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.5])

    fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], name='RSI', line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", row=1, col=1)

    fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], name='MACD', line=dict(color='#3b82f6')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal', line=dict(color='#f59e0b')), row=2, col=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['macd_hist'], name='Hist', marker_color='#6b7db3'), row=2, col=1)

    fig.update_layout(height=400, showlegend=False, **PBASE)
    return fig

def plot_market_breadth(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['gainers'], name='Gainers', fill='tozeroy', line=dict(color='#10b981')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['losers'], name='Losers', fill='tozeroy', line=dict(color='#ef4444')))
    fig.update_layout(title='Market Breadth (Gainers vs Losers)', height=300, **PBASE)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title('NEPSE Pro v3')
    st.subheader('Navigation')
    page = st.radio('', [
        'Overview', 'Technicals', 'Screener', 'ML Signals',
        '📊 Recommendations', '🏆 Top Opportunities',
        '🛡️ Risk Dashboard', '📈 Sector Analysis',
        '🌐 Market Summary', '💡 Advanced Insights',
        'Risk Manager', 'Alerts Central', 'Market History',
        'Live Trading', 'User Guide'
    ], label_visibility='collapsed')

    st.divider()
    latest_dt = load_latest_date()
    st.info(f'📅 Data as of: **{latest_dt or "N/A"}**')

    with st.expander('⚙️ System Controls'):
        if st.button('🔄 Fetch Today', use_container_width=True):
            from pipeline import run_today
            with st.spinner('Scraping...'):
                res = run_today()
                st.toast(str(res))
                st.cache_data.clear()

        if st.button('🧠 Retrain Model', use_container_width=True):
            from pipeline import retrain_model
            with st.spinner('Training ensemble...'):
                res = retrain_model()
                st.toast(f"Trained: {res.get('status')}")

        st.divider()
        st.subheader('📤 Bulk Upload')
        uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv', 'xls'])
        if uploaded_file is not None:
            if st.button('🔬 Analyze & Save', use_container_width=True):
                with st.spinner('Running hedge-fund analysis...'):
                    try:
                        from analytics_engine import load_multi_sheet_excel, run_full_analysis
                        from db import engine

                        if uploaded_file.name.endswith('.csv'):
                            raw_df = pd.read_csv(uploaded_file)
                        else:
                            raw_df = pd.read_excel(uploaded_file, sheet_name=0)

                        analyzed = run_full_analysis(raw_df)
                        if not analyzed.empty:
                            # Save to DB
                            from analytics_engine import get_latest_analysis
                            from sqlalchemy import inspect
                            latest = get_latest_analysis(analyzed)

                            # Filter columns to only those that exist in the DB schema
                            inspector = inspect(engine)
                            db_columns = [c['name'] for c in inspector.get_columns('stock_data')]
                            latest_for_db = latest[[c for c in latest.columns if c in db_columns]]

                            latest_for_db.to_sql("stock_data", engine, if_exists="append", index=False)

                            # Store analysis in session
                            st.session_state['analyzed_data'] = analyzed
                            st.session_state['latest_analysis'] = latest

                            st.success(f"✅ Analyzed {len(latest)} stocks with {analyzed.shape[1]} features!")
                            st.cache_data.clear()
                        else:
                            st.error("No data processed.")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

            if st.button('📊 Analyze Only (No Save)', use_container_width=True):
                with st.spinner('Running analysis...'):
                    try:
                        from analytics_engine import run_full_analysis, get_latest_analysis

                        if uploaded_file.name.endswith('.csv'):
                            raw_df = pd.read_csv(uploaded_file)
                        else:
                            raw_df = pd.read_excel(uploaded_file, sheet_name=0)

                        analyzed = run_full_analysis(raw_df)
                        if not analyzed.empty:
                            latest = get_latest_analysis(analyzed)
                            st.session_state['analyzed_data'] = analyzed
                            st.session_state['latest_analysis'] = latest
                            st.success(f"✅ Analyzed {len(latest)} stocks — view in dashboard pages!")
                        else:
                            st.error("No data processed.")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Get analyzed data (from session or DB)
# ═══════════════════════════════════════════════════════════════════════════════
def get_analysis_data():
    """Get analyzed data from session state or fall back to DB snapshot."""
    if 'analyzed_data' in st.session_state:
        return st.session_state['analyzed_data']
    snap = load_snapshot()
    if not snap.empty:
        try:
            from analytics_engine import compute_derived_features, compute_cross_sectional
            processed = []
            for sym in snap['symbol'].unique():
                sym_df = snap[snap['symbol'] == sym].copy()
                sym_df = compute_derived_features(sym_df)
                processed.append(sym_df)
            if processed:
                result = pd.concat(processed, ignore_index=True)
                return compute_cross_sectional(result)
        except Exception:
            pass
    return snap

def get_latest_data():
    if 'latest_analysis' in st.session_state:
        return st.session_state['latest_analysis']
    return load_snapshot()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

if page == 'Overview':
    st.title('📈 Market Overview')

    snap = load_snapshot()
    hist = load_market_hist()

    if not snap.empty:
        col1, col2, col3, col4 = st.columns(4)
        up = len(snap[snap['diff_pct'] > 0])
        dn = len(snap[snap['diff_pct'] < 0])
        col1.metric('Advancers', up, f'{up/(up+dn)*100:.1f}%' if up+dn > 0 else None)
        col2.metric('Decliners', dn, f'-{dn/(up+dn)*100:.1f}%' if up+dn > 0 else None, delta_color='inverse')
        col3.metric('Total Volume', fmt_num(snap['volume'].sum()))
        top_row = snap.sort_values('diff_pct').iloc[-1]
        top_val = top_row['diff_pct']
        top_str = f"+{top_val:.2f}%" if top_val is not None and not np.isnan(top_val) else "—"
        col4.metric('Top Gainer', top_row['symbol'], top_str)

    c1, c2 = st.columns([2, 1])
    with c1:
        if not hist.empty:
            st.plotly_chart(plot_market_breadth(hist), use_container_width=True)
    with c2:
        st.subheader('⭐ Watchlist')
        wl = load_watchlist()
        if not wl.empty:
            st.dataframe(wl[['symbol', 'target_buy', 'note']], use_container_width=True, hide_index=True)
        else:
            st.caption('Your watchlist is empty')

elif page == 'Technicals':
    st.title('📊 Technical Analysis')
    syms = load_symbols()
    sel_sym = st.selectbox('Select Symbol', syms)

    if sel_sym:
        df = load_sym_hist(sel_sym)
        if not df.empty:
            tab1, tab2 = st.tabs(['Price & Volume', 'Indicators'])
            with tab1:
                st.plotly_chart(plot_main_chart(df), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_rsi_macd(df), use_container_width=True)

            with st.sidebar:
                st.divider()
                st.subheader(f'Add to Watchlist: {sel_sym}')
                note = st.text_input('Note')
                target = st.number_input('Target Buy', value=float(df.iloc[-1]['close']))
                if st.button('Add ⭐'):
                    from db import add_watchlist
                    add_watchlist(sel_sym, note, target)
                    st.toast('Added to watchlist')

elif page == 'Screener':
    st.title('🔍 Multi-Factor Screener')
    snap = load_snapshot()
    if not snap.empty:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader('Filters')
            with st.form("screener_form"):
                min_vol = st.number_input('Min Volume', 0, 10_000_000, 1000)
                available_sectors = sorted(snap['sector'].dropna().unique().tolist()) if 'sector' in snap.columns else []
                sec = st.multiselect('Sectors', available_sectors)
                sig_f = st.multiselect('Signals', ['STRONG_BUY', 'BUY', 'SELL', 'HOLD', 'NONE'] if 'signal' in snap.columns else [])
                submitted = st.form_submit_button("Apply Filters", use_container_width=True)

        with col2:
            if submitted or 'screener_filtered' not in st.session_state:
                filtered = snap[snap['volume'] >= min_vol]
                if sec:
                    filtered = filtered[filtered['sector'].isin(sec)]
                if sig_f and 'signal' in filtered.columns:
                    filtered = filtered[filtered['signal'].isin(sig_f)]
                st.session_state['screener_filtered'] = filtered
            else:
                filtered = st.session_state.get('screener_filtered', snap)

            display_cols = [c for c in ['symbol', 'sector', 'ltp', 'diff_pct', 'volume', 'rsi', 'signal'] if c in filtered.columns]
            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

elif page == 'ML Signals':
    st.title('🤖 AI Trading Signals')
    sigs = load_signals(limit=100)
    if not sigs.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader('Recent BUY Signals')
            buys = sigs[sigs['signal'] == 'BUY']
            st.dataframe(buys[['symbol', 'ltp', 'diff_pct', 'rsi', 'date']], use_container_width=True, hide_index=True)
        with c2:
            st.subheader('Recent SELL Signals')
            sells = sigs[sigs['signal'] == 'SELL']
            st.dataframe(sells[['symbol', 'ltp', 'diff_pct', 'rsi', 'date']], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# A. STOCK RECOMMENDATIONS TABLE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📊 Recommendations':
    st.title('📊 Stock Recommendations')
    st.caption('Hybrid rule-based + ML signal engine')

    data = get_latest_data()
    if not data.empty:
        from strategy import generate_recommendations
        recs = generate_recommendations(data)

        if not recs.empty:
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Strong Buys', len(recs[recs['signal'] == 'STRONG_BUY']))
            c2.metric('Buys', len(recs[recs['signal'] == 'BUY']))
            c3.metric('Holds', len(recs[recs['signal'] == 'HOLD']))
            c4.metric('Sells', len(recs[recs['signal'] == 'SELL']))

            # Filter controls
            sig_filter = st.multiselect('Filter by Signal', ['STRONG_BUY', 'BUY', 'HOLD', 'SELL'], default=['STRONG_BUY', 'BUY'])
            risk_filter = st.multiselect('Filter by Risk', ['LOW', 'MEDIUM', 'HIGH'], default=['LOW', 'MEDIUM'])

            filtered = recs
            if sig_filter:
                filtered = filtered[filtered['signal'].isin(sig_filter)]
            if risk_filter:
                filtered = filtered[filtered['risk_level'].isin(risk_filter)]

            display_cols = ['symbol', 'signal', 'entry_price', 'target_price', 'stop_loss',
                            'rr_ratio', 'confidence', 'risk_level', 'reasoning']
            available = [c for c in display_cols if c in filtered.columns]

            st.dataframe(filtered[available], use_container_width=True, hide_index=True,
                         column_config={
                             'confidence': st.column_config.ProgressColumn('Confidence %', min_value=0, max_value=100),
                             'rr_ratio': st.column_config.NumberColumn('R:R Ratio', format='%.2f'),
                         })
        else:
            st.info('Upload data to generate recommendations')
    else:
        st.info('📤 Upload an Excel/CSV file to get started')


# ═══════════════════════════════════════════════════════════════════════════════
# B. TOP OPPORTUNITIES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🏆 Top Opportunities':
    st.title('🏆 Top 10 High-Conviction Stocks')

    data = get_latest_data()
    if not data.empty:
        from strategy import generate_recommendations, get_top_opportunities

        recs = generate_recommendations(data)
        top = get_top_opportunities(recs, 10)

        if not top.empty:
            for i, (_, stock) in enumerate(top.iterrows()):
                with st.container():
                    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
                    c1.markdown(f"### #{i+1} {stock['symbol']}")
                    c2.markdown(f"**Signal:** :{'green' if 'BUY' in stock['signal'] else 'red'}[{stock['signal']}]  \n"
                                f"**Confidence:** {stock['confidence']}%")
                    c3.metric('Entry', f"₹{stock['entry_price']}")
                    c4.metric('Target', f"₹{stock['target_price']}", f"R:R {stock['rr_ratio']}")
                    st.caption(f"🧠 {stock['reasoning']}")
                    st.divider()
        else:
            st.info('No high-conviction opportunities found')
    else:
        st.info('📤 Upload data to discover opportunities')


# ═══════════════════════════════════════════════════════════════════════════════
# C. RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🛡️ Risk Dashboard':
    st.title('🛡️ Portfolio Risk Dashboard')

    data = get_analysis_data()
    if not data.empty:
        from portfolio import get_risk_dashboard

        risk = get_risk_dashboard(data)

        c1, c2 = st.columns(2)
        c1.metric('Avg Market Volatility', f"{risk.get('avg_market_volatility', 0):.2f}%")
        c2.metric('Avg Market Drawdown', f"{risk.get('avg_market_drawdown', 0):.2f}%")

        tab1, tab2 = st.tabs(['Most Volatile', 'Biggest Drawdowns'])
        with tab1:
            vol_data = risk.get('most_volatile', [])
            if vol_data:
                st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)
            else:
                st.caption('Insufficient data for volatility analysis')

        with tab2:
            dd_data = risk.get('biggest_drawdowns', [])
            if dd_data:
                st.dataframe(pd.DataFrame(dd_data), use_container_width=True, hide_index=True)
            else:
                st.caption('Insufficient data for drawdown analysis')
    else:
        st.info('📤 Upload data to view risk metrics')


# ═══════════════════════════════════════════════════════════════════════════════
# D. SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '📈 Sector Analysis':
    st.title('📈 Sector & Market Intelligence')

    data = get_analysis_data()
    if not data.empty:
        from sector_analysis import compute_sector_metrics, detect_capital_rotation

        sectors = compute_sector_metrics(data)

        if not sectors.empty:
            st.subheader('Sector Ranking')
            display_cols = ['rank', 'sector', 'stocks', 'momentum_score', 'avg_ret_1d',
                            'avg_ret_20d', 'avg_rsi', 'breadth_pct', 'trend']
            available = [c for c in display_cols if c in sectors.columns]
            st.dataframe(sectors[available], use_container_width=True, hide_index=True)

            # Sector momentum chart
            fig = px.bar(sectors, x='sector', y='momentum_score',
                         color='momentum_score',
                         color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                         title='Sector Momentum Scores')
            fig.update_layout(**PBASE, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Capital rotation
            st.subheader('💰 Capital Rotation')
            rotations = detect_capital_rotation(data)
            if rotations:
                rot_df = pd.DataFrame(rotations)
                st.dataframe(rot_df, use_container_width=True, hide_index=True)
            else:
                st.caption('Insufficient historical data for rotation analysis')
        else:
            st.info('No sector data available')
    else:
        st.info('📤 Upload data for sector analysis')


# ═══════════════════════════════════════════════════════════════════════════════
# E. MARKET SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '🌐 Market Summary':
    st.title('🌐 Market Summary & Sentiment')

    data = get_analysis_data()
    if not data.empty:
        from sector_analysis import get_market_sentiment

        sentiment = get_market_sentiment(data)

        # Sentiment display
        sent = sentiment.get('sentiment', 'Unknown')
        score = sentiment.get('score', 0)
        sent_color = '#22c55e' if 'Bullish' in sent else '#ef4444' if 'Bearish' in sent else '#f59e0b'
        st.markdown(f"<h2 style='color:{sent_color};text-align:center;'>{sent} (Score: {score})</h2>",
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Gainers', sentiment.get('gainers', 0))
        c2.metric('Losers', sentiment.get('losers', 0))
        c3.metric('Breadth', f"{sentiment.get('breadth_pct', 0):.1f}%")
        c4.metric('Avg Return', f"{sentiment.get('avg_return', 0):.2f}%")

        c5, c6, c7 = st.columns(3)
        c5.metric('Buy Signals', sentiment.get('buy_signals', 0))
        c6.metric('Sell Signals', sentiment.get('sell_signals', 0))
        c7.metric('Avg RSI', f"{sentiment.get('avg_rsi', 50):.1f}")

        # Reasoning
        reasons = sentiment.get('reasons', [])
        if reasons:
            st.subheader('📋 Analysis')
            for r in reasons:
                st.markdown(f"• {r}")

        # Strategy recommendation
        st.subheader('🎯 Strategy Recommendation')
        if 'Bullish' in sent:
            st.success('📈 **Favor long positions.** Look for pullbacks to support, breakouts with volume. Increase equity exposure.')
        elif 'Bearish' in sent:
            st.error('📉 **Defensive mode.** Reduce exposure, tighten stop-losses, consider hedging. Focus on quality stocks only.')
        else:
            st.warning('↔️ **Range-bound strategy.** Trade within support/resistance. Avoid large positions. Wait for directional clarity.')
    else:
        st.info('📤 Upload data for market analysis')


# ═══════════════════════════════════════════════════════════════════════════════
# F. ADVANCED INSIGHTS (HEDGE FUND LEVEL)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == '💡 Advanced Insights':
    st.title('💡 Advanced Hedge Fund Insights')

    data = get_analysis_data()
    if not data.empty:
        latest = data.sort_values('date').groupby('symbol').tail(1) if 'symbol' in data.columns else data

        tab1, tab2, tab3, tab4 = st.tabs([
            'Smart Money', 'Breakout Scanner', 'Valuation', 'Trade Ideas'
        ])

        with tab1:
            st.subheader('🏦 Smart Money Detection')
            st.caption('Volume + price divergence analysis reveals institutional activity')
            if 'smart_money' in latest.columns:
                accum = latest[latest['smart_money'] == 'Accumulation']
                distr = latest[latest['smart_money'] == 'Distribution']

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('#### 🟢 Accumulation (Institutions Buying)')
                    if not accum.empty:
                        cols = ['symbol', 'close', 'volume_ratio', 'rsi']
                        avail = [c for c in cols if c in accum.columns]
                        st.dataframe(accum[avail], use_container_width=True, hide_index=True)
                    else:
                        st.caption('No accumulation detected')
                with c2:
                    st.markdown('#### 🔴 Distribution (Institutions Selling)')
                    if not distr.empty:
                        cols = ['symbol', 'close', 'volume_ratio', 'rsi']
                        avail = [c for c in cols if c in distr.columns]
                        st.dataframe(distr[avail], use_container_width=True, hide_index=True)
                    else:
                        st.caption('No distribution detected')

        with tab2:
            st.subheader('🚀 Breakout Scanner')
            if 'breakout_up' in latest.columns:
                breakouts = latest[latest['breakout_up'] == True]
                if not breakouts.empty:
                    cols = ['symbol', 'close', 'volume_ratio', 'rsi', 'ret_1d']
                    avail = [c for c in cols if c in breakouts.columns]
                    st.dataframe(breakouts[avail], use_container_width=True, hide_index=True)
                else:
                    st.caption('No breakouts detected today')

            if 'gap_up' in latest.columns:
                gaps = latest[latest['gap_up'] == True]
                if not gaps.empty:
                    st.subheader('📊 Gap-Up Stocks')
                    cols = ['symbol', 'close', 'gap_pct', 'volume']
                    avail = [c for c in cols if c in gaps.columns]
                    st.dataframe(gaps[avail], use_container_width=True, hide_index=True)

        with tab3:
            st.subheader('📐 Overbought vs Oversold')
            if 'rsi' in latest.columns:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('#### Oversold (Potential Value)')
                    oversold = latest[latest['rsi'] < 30].sort_values('rsi')
                    if not oversold.empty:
                        cols = ['symbol', 'close', 'rsi', 'ret_20d']
                        avail = [c for c in cols if c in oversold.columns]
                        st.dataframe(oversold[avail].head(10), use_container_width=True, hide_index=True)
                    else:
                        st.caption('No stocks in oversold territory')
                with c2:
                    st.markdown('#### Overbought (Caution)')
                    overbought = latest[latest['rsi'] > 70].sort_values('rsi', ascending=False)
                    if not overbought.empty:
                        cols = ['symbol', 'close', 'rsi', 'ret_20d']
                        avail = [c for c in cols if c in overbought.columns]
                        st.dataframe(overbought[avail].head(10), use_container_width=True, hide_index=True)
                    else:
                        st.caption('No stocks in overbought territory')

        with tab4:
            st.subheader('🎯 Probabilistic Trade Ideas')
            st.caption('Data-driven, high-probability setups with reasoning')

            from strategy import generate_recommendations, get_top_opportunities
            recs = generate_recommendations(latest)
            top = get_top_opportunities(recs, 5)

            if not top.empty:
                for _, idea in top.iterrows():
                    with st.expander(f"{'🟢' if 'BUY' in idea['signal'] else '🔴'} {idea['symbol']} — {idea['signal']} ({idea['confidence']}% conf)"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric('Entry', f"₹{idea['entry_price']}")
                        c2.metric('Target', f"₹{idea['target_price']}")
                        c3.metric('Stop Loss', f"₹{idea['stop_loss']}")
                        st.markdown(f"**R:R Ratio:** {idea['rr_ratio']}")
                        st.markdown(f"**Risk Level:** {idea['risk_level']}")
                        st.markdown(f"**Reasoning:** {idea['reasoning']}")
            else:
                st.caption('No high-probability setups found')


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING PAGES (Risk Manager, Alerts, etc.)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == 'Risk Manager':
    st.title('🛡️ Portfolio Risk Manager')
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader('Position Sizing Calculator')
        acc = st.number_input('Account Balance (NPR)', 1000, 10_000_000, 100_000)
        risk_p = st.slider('Risk per Trade (%)', 0.5, 5.0, 2.0)
        entry = st.number_input('Entry Price', 1.0, 10000.0, 500.0)
        stop = st.number_input('Stop Loss', 1.0, 10000.0, 480.0)

        if st.button('Calculate Size'):
            from ml_engine import position_size
            res = position_size(acc, risk_p, entry, stop)
            if 'error' in res: st.error(res['error'])
            else:
                st.success(f"Suggested Shares: {res['shares']}")
                st.info(f"Capital Required: NPR {res['capital']:,.2f}")
                st.warning(f"Max Loss: NPR {res['max_loss']:,.2f}")

elif page == 'Alerts Central':
    st.title('🔔 System Alerts')
    if st.button('Mark all as Read'):
        from db import mark_alerts_read
        mark_alerts_read()
        st.cache_data.clear()
        st.rerun()

    alerts = load_alerts()
    if not alerts.empty:
        st.dataframe(alerts[['date', 'symbol', 'alert_type', 'message', 'severity']],
                     use_container_width=True, hide_index=True)
    else:
        st.success('No active alerts')

elif page == 'Market History':
    st.title('📅 Market History')
    hist = load_market_hist(days=180)
    if not hist.empty:
        st.line_chart(hist.set_index('date')[['total_vol']], use_container_width=True)
        st.dataframe(hist, use_container_width=True, hide_index=True)

elif page == 'Live Trading':
    st.title('⚡ Live Market Feed')
    from scraper import fetch_live
    live = fetch_live()
    if live:
        st.dataframe(pd.DataFrame(live), use_container_width=True, hide_index=True)
    else:
        st.info('Market is closed or no live data available.')

elif page == 'User Guide':
    st.title('📖 NEPSE Pro User Guide')
    st.markdown("""
    ### Quick Start
    1. **Upload Excel/CSV** via System Controls → Bulk Upload
    2. **Click "Analyze & Save"** to run the full analytics pipeline
    3. **Explore pages**: Recommendations, Top Opportunities, Sector Analysis

    ### Signal Definitions
    - **STRONG_BUY**: High ML confidence + RSI oversold + breakout (Score ≥ 70)
    - **BUY**: Positive confluence of indicators (Score ≥ 15)
    - **HOLD**: No clear directional bias
    - **SELL**: Bearish indicators dominate (Score ≤ -10)

    ### Pages
    | Page | Description |
    |------|-------------|
    | 📊 Recommendations | Full stock signal table with entry/target/stop |
    | 🏆 Top Opportunities | Top 10 highest conviction trades |
    | 🛡️ Risk Dashboard | Volatility, drawdown analysis |
    | 📈 Sector Analysis | Sector ranking, capital rotation |
    | 🌐 Market Summary | Overall sentiment & strategy |
    | 💡 Advanced Insights | Smart money, breakouts, trade ideas |

    ### Methodology
    - **36+ technical indicators** computed per stock
    - **Hybrid scoring**: Rule-based (16 factors) + ML ensemble (XGBoost/LightGBM)
    - **Portfolio optimization**: Sharpe ratio, sector diversification
    - **Risk management**: Position sizing, R:R filtering, trailing stops
    """)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass
