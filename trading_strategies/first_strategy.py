#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------- Utilities ---------
def rsi(series: pd.Series, window: int = 14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd

def annualize_return(daily_ret: pd.Series, periods_per_year=252):
    growth = (1 + daily_ret).prod()
    n_periods = daily_ret.shape[0]
    if n_periods == 0:
        return np.nan
    return growth ** (periods_per_year / n_periods) - 1

def annualize_vol(daily_ret: pd.Series, periods_per_year=252):
    return daily_ret.std() * np.sqrt(periods_per_year)

def sharpe(daily_ret: pd.Series, rf=0.0, periods_per_year=252):
    ex = daily_ret - (rf / periods_per_year)
    vol = annualize_vol(ex, periods_per_year)
    return np.nan if vol == 0 or np.isnan(vol) else annualize_return(ex, periods_per_year) / vol

def month_ends(idx: pd.DatetimeIndex):
    return idx.to_series().groupby([idx.year, idx.month]).tail(1).index

# --------- Strategy core ---------
@dataclass
class Params:
    sma_window: int = 200
    rsi_window: int = 2
    rsi_threshold: float = 5.0
    tactical_boost: float = 0.5      # +50% weight for 5 days
    tactical_days: int = 5
    fee_bps: float = 5.0              # one-way fee in basis points
    periods_per_year: int = 252

def download_prices(tickers, start, end=None):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how='all')

def build_signals(prices: pd.DataFrame, p: Params):
    sma = prices.rolling(p.sma_window).mean()
    uptrend = prices > sma

    rsi2 = prices.apply(lambda s: rsi(s, p.rsi_window))
    washed_out = (rsi2 < p.rsi_threshold)

    # Core target weights: equal-weight among uptrend members, 0 otherwise
    core_active = uptrend.astype(int)
    core_counts = core_active.replace(0, np.nan).sum(axis=1)
    core_w = core_active.div(core_counts, axis=0).fillna(0)

    # Tactical overlay: when washed out *and* uptrend, add boost for p.tactical_days
    tact_mask = (washed_out & uptrend)
    tact_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        starts = tact_mask.index[tact_mask[col]]
        # mark boost for next p.tactical_days
        for ts in starts:
            idx_loc = tact_mask.index.get_loc(ts)
            end_loc = min(idx_loc + p.tactical_days, len(tact_mask.index))
            tact_w.iloc[idx_loc:end_loc, tact_w.columns.get_loc(col)] += p.tactical_boost

    # Combine and cap (never more than 2x core weight to avoid runaway concentration)
    target_w = core_w * (1 + tact_w)
    max_cap = (core_w * 2.0).where(core_w > 0, 0.0)
    target_w = target_w.clip(lower=0.0).where(core_w > 0, 0.0)  # zero where core is zero
    target_w = pd.DataFrame(np.minimum(target_w.values, max_cap.values),
                            index=target_w.index, columns=target_w.columns)

    # Normalize weights daily (if any float-up happened)
    w_sum = target_w.sum(axis=1).replace(0, np.nan)
    target_w = target_w.div(w_sum, axis=0).fillna(0.0)
    return target_w, core_w, uptrend, rsi2

def simulate(prices: pd.DataFrame, target_w: pd.DataFrame, p: Params):
    rebal_dates = month_ends(prices.index)

    # Only trade (move to target) on rebal dates; otherwise hold previous weights
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_w = pd.Series(0.0, index=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)

    for dt in prices.index:
        if dt in rebal_dates:
            tgt = target_w.loc[dt]
            trn = (tgt - last_w).abs().sum()
            turnover.loc[dt] = trn
            last_w = tgt
        w.loc[dt] = last_w

    # Compute PnL
    rets = prices.pct_change().fillna(0.0)
    gross_port_ret = (w.shift(1) * rets).sum(axis=1).fillna(0.0)

    # Fees: apply on rebalance days only, proportional to turnover
    fee_rate = p.fee_bps / 10000.0
    fee_series = turnover * fee_rate
    net_port_ret = gross_port_ret - fee_series.fillna(0.0)

    equity = (1 + net_port_ret).cumprod()
    return net_port_ret, equity, turnover

def summarize(net_ret: pd.Series, equity: pd.Series, turnover: pd.Series, p: Params):
    cagr = annualize_return(net_ret, p.periods_per_year)
    vol = annualize_vol(net_ret, p.periods_per_year)
    shp = sharpe(net_ret, 0.0, p.periods_per_year)
    mdd, dd_series = max_drawdown(equity)
    avg_turnover = turnover[turnover.index.isin(month_ends(turnover.index))].mean()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": shp,
        "Max Drawdown": mdd,
        "Avg Monthly Turnover": avg_turnover
    }, dd_series

# --------- CLI ---------
def main():
    parser = argparse.ArgumentParser(description="Long-term trend + short-term mean-reversion backtest")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers (stocks/ETFs)")
    parser.add_argument("--start", type=str, default="2005-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--fee_bps", type=float, default=5.0, help="One-way fee in basis points")
    parser.add_argument("--sma", type=int, default=200, help="SMA window for trend filter")
    parser.add_argument("--rsiw", type=int, default=2, help="RSI window")
    parser.add_argument("--rsith", type=float, default=5.0, help="RSI oversold threshold")
    parser.add_argument("--boost", type=float, default=0.5, help="Tactical boost (+50% = 0.5)")
    parser.add_argument("--boost_days", type=int, default=5, help="Days to keep tactical boost")
    args = parser.parse_args()

    p = Params(
        sma_window=args.sma,
        rsi_window=args.rsiw,
        rsi_threshold=args.rsith,
        tactical_boost=args.boost,
        tactical_days=args.boost_days,
        fee_bps=args.fee_bps
    )

    tickers = args.tickers
    prices = download_prices(tickers, args.start, args.end).dropna(how="all")
    prices = prices.dropna()  # ensure full alignment

    target_w, core_w, uptrend, rsi2 = build_signals(prices, p)
    net_ret, equity, turnover = simulate(prices, target_w, p)
    stats, dd_series = summarize(net_ret, equity, turnover, p)

    # Print stats
    print("\n=== Strategy Results ===")
    for k, v in stats.items():
        if k in ["CAGR", "Volatility", "Avg Monthly Turnover"]:
            print(f"{k:>20}: {v: .2%}")
        elif k == "Max Drawdown":
            print(f"{k:>20}: {v: .2%}")
        else:
            print(f"{k:>20}: {v: .2f}")

    # Plot equity curve
    plt.figure(figsize=(10,5))
    plt.plot(equity.index, equity.values, label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("equity_curve.png", dpi=150)
    print("Saved equity curve as equity_curve.png")

if __name__ == "__main__":
    main()
