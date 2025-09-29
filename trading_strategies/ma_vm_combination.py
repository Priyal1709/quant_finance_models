import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------- Params ----------------
TICKERS = ["SPY", "QQQ", "TLT", "GLD"]   # change as you wish
START   = "2010-01-01"
SMA_WIN = 200
LOOKBACK = 252           # trading days for mean/var estimates (~1y)
FREQ = "M"               # we’ll compute true business month-ends from data anyway
RIDGE = 1e-6             # stabilizer for covariance
ALLOW_SHORTS = False     # set True if you want long/short

# ------------- Helpers (robust) -------------
def business_month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available trading day for each (year, month) in idx."""
    s = pd.Series(index=idx, data=True)
    return s.groupby([idx.year, idx.month]).tail(1).index

def mean_var_weights(returns: pd.DataFrame,
                     ridge: float = 1e-6,
                     allow_shorts: bool = False) -> pd.Series:
    """Max-Sharpe style vector ~ inv(cov) * mu, with guards."""
    if returns.shape[0] < 5 or returns.shape[1] == 0:
        return pd.Series(dtype=float)

    mu = returns.mean().values  # daily mean; scaling constant cancels after normalization
    cov = returns.cov().values
    n = cov.shape[0]
    cov = cov + ridge * np.eye(n)

    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        return pd.Series(dtype=float)

    w = inv @ mu

    if not allow_shorts:
        w = np.clip(w, 0, None)

    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        # fallback: equal-weight
        w = np.ones(n) / n
    else:
        w = w / s

    return pd.Series(w, index=returns.columns)

# ------------- Data -------------
raw = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)
if "Close" in raw:
    prices = raw["Close"].copy()
else:
    # yf can return a Series when single ticker is used
    prices = raw.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(TICKERS[0])

prices = prices.dropna(how="all")
prices = prices.loc[:, prices.columns.notna()]   # guard weird NaN col names

# Align columns to declared tickers (drop silently-missing tickers from the list)
available = [t for t in TICKERS if t in prices.columns]
if len(available) == 0:
    raise ValueError("None of the requested tickers have price data.")
TICKERS = available
prices = prices[TICKERS]

rets = prices.pct_change().dropna(how="all")
# ------------- Signals (shifted to avoid look-ahead) -------------
sma = prices.rolling(SMA_WIN, min_periods=1).mean()
trend_raw = prices > sma
trend = trend_raw.shift(1).fillna(False)  # use yesterday’s signal to trade today

# ------------- Rebalance dates (true BME from your data) -------------
rebalance_dates = business_month_ends(prices.index)

# ------------- Portfolio simulation -------------
weights = pd.DataFrame(index=rets.index, columns=TICKERS, data=0.0)

for date in rebalance_dates:
    if date not in rets.index:
        # if BME lands on a day with no returns row (rare), skip
        continue

    # robust lookback window ending at 'date'
    window = rets.loc[:date].tail(LOOKBACK)
    if window.shape[0] < int(0.8 * LOOKBACK):  # need enough history
        continue

    # survivors by trend (guard missing)
    if date not in trend.index:
        continue
    tr_row = trend.loc[date].reindex(TICKERS).fillna(False)
    survivors = [t for t, ok in tr_row.items() if bool(ok)]

    if len(survivors) == 0:
        # no positions this month
        continue

    # run optimizer on survivors only
    w_series = mean_var_weights(window[survivors], ridge=RIDGE, allow_shorts=ALLOW_SHORTS)

    if w_series.empty:
        # fallback equal-weight among survivors
        w_series = pd.Series(1.0 / len(survivors), index=survivors)

    weights.loc[date, survivors] = w_series.values

# Forward-fill weights to daily frequency, reindex to returns index
weights = weights.reindex(rets.index).ffill().fillna(0.0)

# Use yesterday’s weights for today’s return to avoid look-ahead
port_rets = (weights.shift(1).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
equity = (1 + port_rets).cumprod()

# ------------- Performance -------------
n = port_rets.shape[0]
if n == 0:
    raise RuntimeError("No returns computed — check your tickers and date range.")

cagr = equity.iloc[-1] ** (252 / n) - 1
vol = port_rets.std() * np.sqrt(252)
sharpe = (cagr / vol) if vol > 0 else np.nan
max_dd = (equity / equity.cummax() - 1).min()

print("=== MA + Mean-Variance Strategy ===")
print(f"Tickers: {TICKERS}")
print(f"Start:   {equity.index.min().date()}  End: {equity.index.max().date()}")
print(f"CAGR:    {cagr:.2%}")
print(f"Vol:     {vol:.2%}")
print(f"Sharpe:  {sharpe:.2f}")
print(f"Max DD:  {max_dd:.2%}")

plt.figure(figsize=(10,5))
plt.plot(equity.index, equity.values, label="Equity")
plt.title("MA + Mean-Variance Strategy")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
try:
    plt.show()
except Exception:
    plt.savefig("equity_curve.png", dpi=150)
    print("Saved plot to equity_curve.png")
