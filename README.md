# Quant Backtester (Binance USDT-M Perps)

A fast, single-position backtester for Binance USDT-M futures strategies. 
It downloads 1–2 years of OHLCV via ccxt (mainnet market data), runs a vectorized strategy (MA cross + slope by default), 
simulates SL/TP exits with conservative intrabar rules, applies fees + slippage + leverage/margin caps, and outputs trades, metrics, and an equity curve plot.

## Project layout
- `data_fetch.py` — backfills OHLCV from Binance (mainnet) using ccxt and caches to CSV.
- `strategy.py` — defines indicators and generates entry signals (default MA cross + slope).
- `backtester.py` — vectorized simulator with risk-based sizing, leverage caps, fees, slippage.
- `run_backtest.py` — CLI to fetch data (or use cache), run the backtest, and save outputs.
- `config.py` — global defaults you can tweak.
- `requirements.txt` — dependencies.
- `outputs/` — equity curve and CSVs will be saved here.
- `data/` — cached data CSVs are saved here (per symbol+timeframe).

## Quickstart
```bash
python -m venv .venv
.venv\Scripts\activate   # (Windows)  |  source .venv/bin/activate  (macOS/Linux)
pip install -r requirements.txt

# 1-year backtest on BTC/USDT 5m with 14k equity
python run_backtest.py --symbol "BTC/USDT" --timeframe 5m --start 2024-08-22 --end 2025-08-22 --equity 14000 --risk-pct 0.003 --sl-pct 0.006 --tp-pct 0.010 --leverage 5 --slippage-bps 1 --taker-fee 0.0005
```

Notes: Uses mainnet market data via ccxt; no keys needed. Entry at next candle’s open, exits intrabar; conservative (SL wins if SL & TP hit in same bar).
