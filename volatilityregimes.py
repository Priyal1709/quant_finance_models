"""
Volatility Regimes and Regime Changes
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_2025_daily_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start="2025-01-01",
        end="2026-01-01",   
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.xs(df.columns.get_level_values(-1)[0], axis=1, level=-1)

    out = df[["Open", "Close"]].dropna().copy()
    out.index = pd.to_datetime(out.index)
    return out


def realized_volatility(close, window: int = 20, annualization: int = 252) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    close = pd.Series(close, name="Close")
    log_ret = np.log(close).diff()
    rv = log_ret.rolling(window).std() * np.sqrt(annualization)
    rv.name = f"RV_{window}d"
    return rv


def assign_vol_regime(rv: pd.Series, q_low: float = 0.33, q_high: float = 0.66) -> pd.Series:
    rv_clean = rv.dropna()
    regime = pd.Series(index=rv.index, dtype="float64", name="vol_regime")

    if rv_clean.empty:
        return regime

    low_thr = rv_clean.quantile(q_low)
    high_thr = rv_clean.quantile(q_high)

    regime[rv < low_thr] = 0
    regime[(rv >= low_thr) & (rv <= high_thr)] = 1
    regime[rv > high_thr] = 2
    return regime


def regime_switches(regime: pd.Series) -> pd.Series:
    sw = (regime != regime.shift(1)) & regime.notna() & regime.shift(1).notna()
    sw.name = "regime_switch"
    return sw


def main(ticker: str = "META", window: int = 20):
    prices = fetch_2025_daily_ohlc(ticker)

    prices["RV"] = realized_volatility(prices["Close"], window=window)
    prices["regime"] = assign_vol_regime(prices["RV"])
    prices["switch"] = regime_switches(prices["regime"])

    print(f"\nTicker: {ticker}")
    print(f"Rows (2025 trading days): {len(prices)}")
    print(f"Realized vol window: {window} days")
    print(f"Regime switches detected: {int(prices['switch'].sum())}\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(prices.index, prices["Close"], label="Close")
    switch_dates = prices.index[prices["switch"] == True]
    axes[0].scatter(switch_dates, prices.loc[switch_dates, "Close"], marker="*", label="Regime switch", color="black")
    axes[0].set_title(f"{ticker} Close Price (2025) with Regime Switch Markers")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].plot(prices.index, prices["RV"], label=f"Realized Vol ({window}d, annualized)")
    for reg, name in [(0, "Low"), (1, "Mid"), (2, "High")]:
        mask = prices["regime"] == reg
        axes[1].scatter(prices.index[mask], prices.loc[mask, "RV"], s=10, label=f"{name} vol regime")

    axes[1].set_title("Realized Volatility + Simple Regime Labels (Quantile-based)")
    axes[1].set_ylabel("Annualized Vol")
    axes[1].legend(ncol=2)

    plt.tight_layout()
    plt.show()

    return prices


if __name__ == "__main__":
    df_out = main(ticker="META", window=20)

