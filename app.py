import streamlit as st
import requests
import pandas as pd
import numpy as np
import pywt
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="Regime & Risk Radar (Crypto)", layout="wide")


# -----------------------------
# Data
# -----------------------------
@st.cache_data(ttl=3600)
def list_top_coins(vs="usd", per_page=100):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "false",
    }
    headers = {
        "accept": "application/json",
        "User-Agent": "regime-risk-radar/1.0 (Streamlit; contact: KhaledBalti95@gmail.com)",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    rows = r.json()
    return {f"{c['name']} ({c['symbol'].upper()})": c["id"] for c in rows}


@st.cache_data(ttl=3600)
def fetch_coingecko_market_chart(coin_id: str, days: int = 365, vs: str = "usd") -> tuple[pd.DataFrame, str]:
    """
    Returns (df, fetched_at_utc_str)
    df columns: date, price, volume
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days, "interval": "daily"}
    headers = {
        "accept": "application/json",
        "User-Agent": "regime-risk-radar/1.0 (Streamlit; contact: KhaledBalti95@gmail.com)",
    }

    last_status = None
    last_text = None

    # small retry for rate limits / transient errors
    for _ in range(3):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        last_status = r.status_code
        last_text = r.text[:250]

        if r.status_code == 200:
            data = r.json()
            prices = pd.DataFrame(data.get("prices", []), columns=["ts_ms", "price"])
            volumes = pd.DataFrame(data.get("total_volumes", []), columns=["ts_ms", "volume"])

            if prices.empty:
                break

            df = prices.merge(volumes, on="ts_ms", how="left")
            df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
            df = (
                df.drop(columns=["ts_ms"])
                .groupby("date", as_index=False)
                .agg({"price": "last", "volume": "last"})
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            return df, fetched_at

        if r.status_code in (429, 500, 502, 503, 504):
            continue
        break

    st.error(
        f"CoinGecko request failed (status {last_status}). "
        f"Try 365 days or wait. Snippet: {last_text}"
    )
    return pd.DataFrame(columns=["date", "price", "volume"]), datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# -----------------------------
# Features / Signals
# -----------------------------
def wavelet_energy_ratio(log_returns: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    x = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    if len(x) < 64:
        return np.nan
    coeffs = pywt.wavedec(x, wavelet, level=level)
    approx = coeffs[0]          # low freq
    details = coeffs[1:]        # high freq
    e_low = float(np.sum(np.square(approx)))
    e_high = float(np.sum([np.sum(np.square(d)) for d in details]))
    denom = e_low + e_high
    if denom == 0:
        return np.nan
    return e_high / denom


def compute_features(df: pd.DataFrame, vol_window=30, mom_window=14, wavelet_window=128) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["price"].pct_change()
    out["logret"] = np.log(out["price"]).diff()

    # Annualized vol (daily)
    out["vol"] = out["ret"].rolling(vol_window).std() * np.sqrt(365)

    # Momentum
    out["mom"] = out["price"].pct_change(mom_window)

    # Drawdown
    out["roll_max"] = out["price"].cummax()
    out["drawdown"] = out["price"] / out["roll_max"] - 1.0

    # Wavelet ratio (rolling)
    ratios = [np.nan] * len(out)
    lr = out["logret"].values
    w = int(wavelet_window)
    for i in range(w, len(out)):
        ratios[i] = wavelet_energy_ratio(lr[i - w:i], level=4)
    out["w_energy_ratio"] = ratios

    # More dynamic risk score: weighted z-scores (bounded 0..100)
    vol_mu = out["vol"].rolling(180, min_periods=60).mean()
    vol_sd = out["vol"].rolling(180, min_periods=60).std()
    dd_mu = (-out["drawdown"]).rolling(180, min_periods=60).mean()
    dd_sd = (-out["drawdown"]).rolling(180, min_periods=60).std()

    vol_z = (out["vol"] - vol_mu) / vol_sd
    dd_z = (-out["drawdown"] - dd_mu) / dd_sd

    risk_raw = 0.6 * vol_z + 0.4 * dd_z
    out["risk_score"] = (50 + 15 * risk_raw).clip(0, 100)

    return out


def decide_regime(row) -> tuple[str, float]:
    vol = row.get("vol", np.nan)
    mom = row.get("mom", np.nan)
    dd = row.get("drawdown", np.nan)
    wr = row.get("w_energy_ratio", np.nan)

    conf = 50.0

    # Less "always risk-off"
    if pd.notna(vol) and pd.notna(dd) and pd.notna(mom) and ((vol > 1.0) or (dd < -0.30 and mom < 0)):
        regime = "RISK-OFF"
        conf = 70.0 + (10.0 if vol > 1.2 else 0.0)
    elif pd.notna(mom) and pd.notna(wr) and pd.notna(dd) and (mom > 0.06 and wr < 0.62 and dd > -0.25):
        regime = "RISK-ON"
        conf = 70.0 + (10.0 if mom > 0.10 else 0.0)
    else:
        regime = "NEUTRAL"
        conf = 55.0

    return regime, float(np.clip(conf, 0, 100))


def decide_signal(row) -> tuple[str, list[str]]:
    reasons = []
    regime, conf = decide_regime(row)

    vol = row.get("vol", np.nan)
    mom = row.get("mom", np.nan)
    dd = row.get("drawdown", np.nan)
    wr = row.get("w_energy_ratio", np.nan)
    risk = row.get("risk_score", np.nan)

    reasons.append(f"Regime={regime} (confidence {conf:.0f}/100)")
    if pd.notna(risk): reasons.append(f"Risk score={risk:.0f}/100")
    if pd.notna(vol):  reasons.append(f"Vol={vol:.2f}")
    if pd.notna(mom):  reasons.append(f"Momentum={mom:.1%}")
    if pd.notna(dd):   reasons.append(f"Drawdown={dd:.1%}")
    if pd.notna(wr):   reasons.append(f"Wavelet energy ratio={wr:.2f} (higher=choppy, lower=smoother)")

    # tuned to actually trade sometimes
    if regime == "RISK-ON" and pd.notna(risk) and risk < 70:
        signal = "BUY"
        reasons.append("Rule: risk-on + risk not extreme → BUY")
    elif regime == "RISK-OFF" or (pd.notna(risk) and risk > 80) or (pd.notna(mom) and mom < -0.06):
        signal = "SELL"
        reasons.append("Rule: risk-off OR risk very high OR momentum strongly negative → SELL")
    else:
        signal = "HOLD"
        reasons.append("Rule: otherwise → HOLD (keep prior position)")

    return signal, reasons


def build_regime_series(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    regimes, confs, signals = [], [], []
    for _, row in out.iterrows():
        reg, conf = decide_regime(row)
        sig, _ = decide_signal(row)
        regimes.append(reg)
        confs.append(conf)
        signals.append(sig)
    out["regime"] = regimes
    out["regime_conf"] = confs
    out["signal"] = signals
    return out


def backtest_signals(df_reg: pd.DataFrame) -> pd.DataFrame:
    bt = df_reg.dropna(subset=["price", "ret"]).copy()
    if bt.empty:
        return bt

    bt["buy_flag"] = (bt["signal"] == "BUY").astype(int)
    bt["sell_flag"] = (bt["signal"] == "SELL").astype(int)

    # HOLD keeps previous position
    pos = []
    current = 0
    for b, s in zip(bt["buy_flag"].values, bt["sell_flag"].values):
        if s == 1:
            current = 0
        elif b == 1:
            current = 1
        pos.append(current)

    bt["pos_raw"] = pos
    bt["pos"] = bt["pos_raw"].shift(1).fillna(0)  # execute next day

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
    if len(r) < 30:
        return np.nan
    sd = r.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((r.mean() / sd) * np.sqrt(365))


def add_regime_shading(fig: go.Figure, df_reg: pd.DataFrame):
    colors = {
        "RISK-ON": "rgba(0, 200, 0, 0.08)",
        "NEUTRAL": "rgba(120, 120, 120, 0.06)",
        "RISK-OFF": "rgba(200, 0, 0, 0.08)",
    }

    d = df_reg.dropna(subset=["regime"]).copy()
    if d.empty:
        return fig

    d = d[["date", "regime"]].reset_index(drop=True)
    seg_start = d.loc[0, "date"]
    seg_regime = d.loc[0, "regime"]

    for i in range(1, len(d)):
        if d.loc[i, "regime"] != seg_regime:
            seg_end = d.loc[i - 1, "date"]
            fig.add_vrect(
                x0=seg_start,
                x1=seg_end,
                fillcolor=colors.get(seg_regime, "rgba(0,0,0,0.03)"),
                opacity=1,
                line_width=0,
                layer="below",
            )
            seg_start = d.loc[i, "date"]
            seg_regime = d.loc[i, "regime"]

    fig.add_vrect(
        x0=seg_start,
        x1=d.loc[len(d) - 1, "date"],
        fillcolor=colors.get(seg_regime, "rgba(0,0,0,0.03)"),
        opacity=1,
        line_width=0,
        layer="below",
    )
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("Regime & Risk Radar (Crypto) — Wavelets + Risk Regimes + Paper Signals")

with st.sidebar:
    st.header("Settings")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    coins = list_top_coins(per_page=100)
    coin_keys = list(coins.keys())

    coin_label = st.selectbox("Asset", coin_keys, index=0)
    coin_id = coins[coin_label]

    compare_on = st.checkbox("Compare with a 2nd asset", value=True)

    default_compare_label = next((k for k in coin_keys if "(ETH)" in k), coin_keys[1] if len(coin_keys) > 1 else coin_keys[0])
    if compare_on:
        compare_label = st.selectbox(
            "2nd Asset",
            coin_keys,
            index=coin_keys.index(default_compare_label) if default_compare_label in coin_keys else 1,
        )
        compare_id = coins[compare_label]
    else:
        compare_label, compare_id = None, None

    days = st.selectbox("History (days)", [180, 365, 730], index=1)
    vol_window = st.slider("Vol window (days)", 10, 60, 30)
    mom_window = st.slider("Momentum window (days)", 7, 60, 14)

# ---- Fetch primary with 730 fallback ----
with st.spinner("Fetching data..."):
    df1, fetched1 = fetch_coingecko_market_chart(coin_id, days=days)
    if df1.empty and days == 730:
        st.warning("730-day fetch failed (likely rate limit). Falling back to 365 days.")
        df1, fetched1 = fetch_coingecko_market_chart(coin_id, days=365)

if df1.empty:
    st.stop()

feat1 = compute_features(df1, vol_window=vol_window, mom_window=mom_window)
reg1 = build_regime_series(feat1)

latest1 = reg1.iloc[-1] if len(reg1) else pd.Series(dtype=float)
signal1, reasons1 = decide_signal(latest1)
regime1, conf1 = decide_regime(latest1)
risk1 = latest1.get("risk_score", np.nan)

# ---- Fetch compare with 730 fallback ----
if compare_on and compare_id:
    with st.spinner("Fetching comparison asset..."):
        df2, fetched2 = fetch_coingecko_market_chart(compare_id, days=days)
        if df2.empty and days == 730:
            st.warning("730-day fetch failed for 2nd asset. Falling back to 365 days.")
            df2, fetched2 = fetch_coingecko_market_chart(compare_id, days=365)

    if not df2.empty:
        feat2 = compute_features(df2, vol_window=vol_window, mom_window=mom_window)
        reg2 = build_regime_series(feat2)
    else:
        feat2, reg2 = None, None
else:
    df2, fetched2, feat2, reg2 = None, None, None, None

# Freshness banner
st.caption(
    f"Data source: CoinGecko (public API). "
    f"Fetched {coin_label} at {fetched1}."
    + (f" Fetched {compare_label} at {fetched2}." if compare_on and fetched2 else "")
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Regime", regime1)
col2.metric("Regime Confidence", f"{conf1:.0f}/100")
col3.metric("Paper Signal", signal1)
col4.metric("Risk Score (0–100)", f"{risk1:.0f}" if pd.notna(risk1) else "—")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Signals & Why", "Backtest", "Methodology"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reg1["date"], y=reg1["price"], name=f"{coin_label} Price"))
    fig = add_regime_shading(fig, reg1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if compare_on and reg2 is not None and not reg2.empty and not reg1.empty:
        a = reg1[["date", "price"]].rename(columns={"price": "p1"})
        b = reg2[["date", "price"]].rename(columns={"price": "p2"})
        m = a.merge(b, on="date", how="inner")
        if not m.empty:
            m["p1_norm"] = m["p1"] / m["p1"].iloc[0]
            m["p2_norm"] = m["p2"] / m["p2"].iloc[0]

            figc = go.Figure()
            figc.add_trace(go.Scatter(x=m["date"], y=m["p1_norm"], name=f"{coin_label} (norm)"))
            figc.add_trace(go.Scatter(x=m["date"], y=m["p2_norm"], name=f"{compare_label} (norm)"))
            figc.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(figc, use_container_width=True)

with tab2:
    st.subheader("Decision explanation (paper, transparent rules)")
    for r in reasons1:
        st.write(f"- {r}")

    st.subheader("Latest rows")
    show_cols = ["date", "price", "ret", "vol", "mom", "drawdown", "w_energy_ratio", "risk_score", "regime", "signal"]
    st.dataframe(reg1[show_cols].tail(12), use_container_width=True)

with tab3:
    bt = backtest_signals(reg1)
    if bt.empty:
        st.warning("Not enough data to backtest yet.")
    else:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=bt["date"], y=bt["strategy_equity"], name="Strategy"))
        fig3.add_trace(go.Scatter(x=bt["date"], y=bt["bh_equity"], name="Buy & Hold"))
        fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig3, use_container_width=True)

        strat_mdd = max_drawdown(bt["strategy_equity"])
        bh_mdd = max_drawdown(bt["bh_equity"])
        strat_sh = sharpe_like(bt["strategy_ret"])
        bh_sh = sharpe_like(bt["bh_ret"])
        hit_rate = float((bt["strategy_ret"] > 0).mean())
        exposure = float(bt["pos"].mean())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy Max DD", f"{strat_mdd:.1%}" if pd.notna(strat_mdd) else "—")
        c2.metric("Buy&Hold Max DD", f"{bh_mdd:.1%}" if pd.notna(bh_mdd) else "—")
        c3.metric("Strategy Sharpe-like", f"{strat_sh:.2f}" if pd.notna(strat_sh) else "—")
        c4.metric("Hit rate (daily)", f"{hit_rate:.1%}" if pd.notna(hit_rate) else "—")
        c5.metric("Exposure (time in market)", f"{exposure:.1%}" if pd.notna(exposure) else "—")

with tab4:
    st.markdown(
        """
### What this is
A **public, deployable data product** that turns crypto market data into:
- regime classification (Risk-On / Risk-Off / Neutral)
- risk scoring (0–100)
- paper BUY/HOLD/SELL decisions with explanations
- backtest vs buy & hold

### Why wavelets
Wavelet energy features help separate **trend-dominant** periods from **choppy/noise-dominant** periods across multiple time scales.

### Leakage prevention (basic)
Signals are computed on day *t* and executed on day *t+1* by shifting the position.

### Limitations
- Daily data only (no intraday microstructure)
- Rule-based regime + signals (transparent by design)
- This is educational and not financial advice

### Next steps
- Probabilistic regime switching (HMM / change-point)
- Multi-asset regime maps (correlation clusters)
- Monitoring: signal decay, drift, regime stability
"""
    )
