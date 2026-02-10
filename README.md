# Regime & Risk Radar (Crypto)

This project is a small but complete data product that explores how market
risk regimes can be detected using public crypto data and multi-scale features.

The goal is not to build a “trading bot”, but to show how raw data can be turned
into signals, explanations, and decisions inside a real web application.

## What the app shows
- Live crypto market data fetched from public APIs
- Volatility, momentum, drawdown, and wavelet-based features
- A simple but transparent market regime classification
  (Risk-On / Risk-Off / Neutral)
- Paper BUY / HOLD / SELL signals with clear reasoning
- A basic walk-forward backtest compared to Buy & Hold

## Why this exists
I built this to demonstrate end-to-end data product skills:
data ingestion, feature engineering, time-series analysis,
decision logic, backtesting, and deployment — all in one place.

## Tech stack
- Python
- Streamlit
- CoinGecko public API
- PyWavelets
- Pandas, NumPy, Plotly

## Disclaimer
This is an educational research project.
All signals are paper-only and not financial advice.
