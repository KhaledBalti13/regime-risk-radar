import streamlit as st
import requests
import pandas as pd
import numpy as np
import pywt
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Regime & Risk Radar (Crypto)", layout="wide")

@st.cache_data(ttl=3600)
def fetch_coingecko_market_chart(coin_id: str, days: int = 365, vs: str = "usd") -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    prices = pd.DataFrame(data["prices"], columns=["ts_ms", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["ts_ms", "volume"])

    df = prices.merge(volumes, on="ts_ms", how="left")
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
    df = df.drop(columns=["ts_ms"]).groupby("date", as_index=False).agg({"price": "last", "volume": "last"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def list_top_coins(vs="usd", per_page=50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs, "order": "market_cap_desc", "per_page": per_page, "page": 1, "sparkline": "false"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    # return mapping: label -> id
    mapping = {f"{c['name']} ({c['symbol'].upper()})": c["id"] for c in rows}
    return mapping

def wavelet_energy_ratio(returns: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    x = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) < 64:
        return np.nan
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # coeffs[0] = approximation (low freq), coeffs[1:] = details (higher freq)
    approx = coeffs[0]
    details = coeffs[1:]
    e_low = np.sum(np.square(approx))
    e_high = np.sum([np.sum(np.square(d)) for d in details])
    if (e_low + e_high) == 0:
        return np.nan
    return float(e_high / (e_low + e_high))

def compute_features(df: pd.DataFrame, vol_window=30, mom_window=14) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["price"].pct_change()
    out["logret"] = np.log(out["price"]).diff()
    out["vol"] = out["ret"].rolling(vol_window).std() * np.sqrt(365)
    out["mom"] = out["price"].pct_change(mom_window)
    out["roll_max"] = out["price"].cummax()
    out["drawdown"] = out["price"] / out["roll_max"] - 1.0

    # Rolling wavelet ratio on a trailing window
    w = 128
    ratios = [np.nan]*len(out)
    r = out["logret"].values
    for i in range(w, len(out)):
        ratios[i] = wavelet_energy_ratio(r[i-w:i], level=4)
    out["w_energy_ratio"] = ratios

    # Simple risk score (0-100): higher = riskier
    # Normalize using rolling quantiles to stay robust
    out["vol_q"] = out["vol"].rolling(180, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    out["dd_q"] = out["drawdown"].rolling(180, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    out["risk_score"] = (50*out["vol_q"] + 50*out["dd_q"]).clip(0, 100)

    return out

def decide_regime_and_signal(row) -> tuple[str, str, list]:
    reasons = []
    vol = row.get("vol", np.nan)
    mom = row.get("mom", np.nan)
    dd = row.get("drawdown", np.nan)
    wr = row.get("w_energy_ratio", np.nan)

    # Regime heuristic
    if pd.notna(vol) and pd.notna(dd) and (vol > 0.8 or dd < -0.25):
        regime = "RISK-OFF"
        reasons.append(f"High risk: vol={vol:.2f} or drawdown={dd:.1%}")
    elif pd.notna(mom) and mom > 0.05 and pd.notna(wr) and wr < 0.60:
        regime = "RISK-ON"
        reasons.append(f"Positive momentum: {mom:.1%}")
        reasons.append(f"Trend-dominant wavelet ratio: {wr:.2f} (lower = smoother trend)")
    else:
        regime = "NEUTRAL"
        if pd.notna(vol): reasons.append(f"Vol={vol:.2f}")
        if pd.notna(mom): reasons.append(f"Mom={mom:.1%}")
        if pd.notna(dd): reasons.append(f"DD={dd:.1%}")
        if pd.notna(wr): reasons.append(f"Wavelet ratio={wr:.2f}")

    # Signal heuristic (paper)
    if regime == "RISK-ON" and pd.notna(vol) and vol < 1.2:
        signal = "BUY"
        reasons.append("Rule: risk-on + vol not extreme")
    elif regime == "RISK-OFF" or (pd.notna(mom) and mom < -0.05):
        signal = "SELL"
        reasons.append("Rule: risk-off or negative momentum")
    else:
        signal = "HOLD"
        reasons.append("Rule: otherwise hold")

    return regime, signal, reasons

def backtest_signals(df: pd.DataFrame) -> pd.DataFrame:
    bt = df.dropna(subset=["price", "ret"]).copy()
    # Generate signal based on today's features; execute next day (shift)
    regimes, signals = [], []
    for _, row in bt.iterrows():
        reg, sig, _ = decide_regime_and_signal(row)
        regimes.append(reg)
        signals.append(sig)
    bt["regime"] = regimes
    bt["signal"] = signals

    # position: BUY=1, HOLD=0, SELL=0 (long-only paper)
    bt["pos_raw"] = (bt["signal"] == "BUY").astype(int)
    bt["pos"] = bt["pos_raw"].shift(1).fillna(0)

    bt["strategy_ret"] = bt["pos"] * bt["ret"]
    bt["bh_ret"] = bt["ret"]

    bt["strategy_equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["bh_equity"] = (1 + bt["bh_ret"]).cumprod()
    return bt

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe_like(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 30 or r.std() == 0:
        return np.nan
    return float((r.mean() / r.std()) * np.sqrt(365))

st.title("Regime & Risk Radar (Crypto) — Wavelets + Risk Regimes + Paper Signals")

with st.sidebar:
    st.header("Settings")
    coins = list_top_coins(per_page=50)
    coin_label = st.selectbox("Asset", list(coins.keys()), index=0)
    coin_id = coins[coin_label]
    days = st.selectbox("History (days)", [180, 365, 730], index=1)
    vol_window = st.slider("Vol window (days)", 10, 60, 30)
    mom_window = st.slider("Momentum window (days)", 7, 60, 14)

df = fetch_coingecko_market_chart(coin_id, days=days)
feat = compute_features(df, vol_window=vol_window, mom_window=mom_window)
latest = feat.iloc[-1]
regime, signal, reasons = decide_regime_and_signal(latest)

col1, col2, col3 = st.columns(3)
col1.metric("Current Regime", regime)
col2.metric("Paper Signal", signal)
col3.metric("Risk Score (0–100)", f"{latest.get('risk_score', np.nan):.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Signals & Why", "Backtest", "Methodology"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feat["date"], y=feat["price"], name="Price"))
    fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=feat["date"], y=feat["vol"], name="Ann. Vol"))
    fig2.update_layout(height=280, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Decision explanation")
    st.write("These are transparent heuristic rules (paper signals) to avoid overfitting.")
    for r in reasons:
        st.write(f"- {r}")

    st.subheader("Latest features")
    show_cols = ["date","price","ret","vol","mom","drawdown","w_energy_ratio","risk_score"]
    st.dataframe(feat[show_cols].tail(10), use_container_width=True)

with tab3:
    bt = backtest_signals(feat)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_equity"], name="Strategy"))
    fig3.add_trace(go.Scatter(x=bt["date"], y=bt["bh_equity"], name="Buy & Hold"))
    fig3.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    strat_mdd = max_drawdown(bt["strategy_equity"])
    bh_mdd = max_drawdown(bt["bh_equity"])
    strat_sh = sharpe_like(bt["strategy_ret"])
    bh_sh = sharpe_like(bt["bh_ret"])
    hit_rate = (bt["strategy_ret"] > 0).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy Max DD", f"{strat_mdd:.1%}")
    c2.metric("Buy&Hold Max DD", f"{bh_mdd:.1%}")
    c3.metric("Strategy Sharpe-like", f"{strat_sh:.2f}")
    c4.metric("Hit rate (daily)", f"{hit_rate:.1%}")

with tab4:
    st.markdown("""
### What this is
A **crypto regime + risk dashboard** using **public CoinGecko data** and **wavelet-based multi-scale features**.
It produces **paper signals (BUY/HOLD/SELL)** with transparent heuristic rules.

### Leakage prevention
- Signals are computed from day *t* features and executed on day *t+1* (position is shifted).

### Limitations
- Heuristic signals are not a trading recommendation.
- Daily data only (no order-book microstructure).
- API rate limits may apply.

### Next improvements (after MVP)
- Walk-forward model (logistic / gradient boosting) with strict time splits
- More assets (ETH, SOL, BTC, etc.) and correlation/regime map
- Alerts & monitoring page
""")
