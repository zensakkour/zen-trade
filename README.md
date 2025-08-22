# Quant Backtester

This is a simple backtesting framework for quantitative trading strategies on cryptocurrency markets.

## Project Structure

The project is organized as follows:

-   `run_backtest.py`: The main script to run the backtest.
-   `.env`: The configuration file for the backtest.
-   `src/`: The main source code directory.
    -   `backtester.py`: The core backtesting logic.
    -   `data_fetch.py`: The data fetching logic.
    -   `config.py`: The default configuration values.
    -   `strategies/`: The directory for trading strategies.
        -   `base.py`: The base class for all strategies.
        -   `simple_ma.py`: An example strategy based on moving averages.
-   `data/`: The directory where cached OHLCV data is stored.
-   `outputs/`: The directory where backtest results (trades, equity curve, etc.) are saved.

## How to Run

The backtest is configured entirely through the `.env` file. You can copy the `.env.example` file to `.env` and modify the parameters as needed.

To run the backtest, simply execute the following command:

```bash
python run_backtest.py
```

The script will automatically load the configuration from the `.env` file and run the backtest.

### Configuration

The `.env` file contains all the parameters for the backtest, including:

-   `EXCHANGE`: The name of the exchange to use (e.g., `bybit`).
-   `STRATEGY`: The name of the strategy to run (e.g., `simple_ma`).
-   `SYMBOL`: The symbol to backtest (e.g., `BTC/USDT`).
-   `TIMEFRAME`: The timeframe to use (e.g., `5m`).
-   `START`: The start date of the backtest (e.g., `2024-08-22`).
-   `END`: The end date of the backtest (e.g., `2025-08-22`).
-   ...and other backtesting parameters.

## How to Add a New Strategy

To add a new strategy, you need to:

1.  Create a new Python file in the `src/strategies/` directory (e.g., `my_strategy.py`).
2.  In the new file, create a new class that inherits from `BaseStrategy` (from `src.strategies.base`).
3.  Implement the `generate_signals` method in your new class. This method should take a pandas DataFrame with OHLCV data and return a DataFrame with `entry_long` and `entry_short` boolean columns.
4.  Set the `STRATEGY` variable in your `.env` file to the name of your new strategy file (without the `.py` extension), e.g., `STRATEGY=my_strategy`.
5.  Add any strategy-specific parameters to the `.env` file and access them in your strategy class through the `self.params` dictionary.
