import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- Parameters ---
tickers = ["SPY", "QQQ", "TLT", "GLD"]
start_date = "2010-01-01"
sma_window = 200
rebalance_freq = "M"   # Monthly
lookback = 252         # 1-year window for mean/variance estimates

# --- Download data ---
prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"]
rets = prices.pct_change().dropna()

# --- Moving average filter ---
sma = prices.rolling(sma_window).mean()
trend = prices > sma   # True if above SMA

# --- Function: mean-variance optimizer (max Sharpe with no shorting) ---
def mean_var_weights(returns):
    mu = returns.mean().values
    cov = returns.cov().values
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mu
    w = np.maximum(w, 0)     # no shorting
    w /= w.sum()
    return pd.Series(w, index=returns.columns)

# --- Portfolio simulation ---
weights = pd.DataFrame(index=rets.index, columns=tickers, data=0.0)

for date in rets.resample(rebalance_freq).last().index:  # rebal dates
    end = date
    start = date - pd.tseries.offsets.BDay(lookback)
    window = rets.loc[start:end]

    if len(window) < lookback * 0.8:  # skip if not enough history
        continue

    # filter assets by SMA rule
    survivors = [t for t in tickers if trend.loc[date, t]]
    if len(survivors) == 0:
        continue

    # run mean-variance optimizer on survivors
    w = mean_var_weights(window[survivors])
    weights.loc[date, survivors] = w

# forward-fill weights daily
weights = weights.resample("D").ffill().reindex(rets.index).fillna(0)

# portfolio returns
port_rets = (weights.shift(1) * rets).sum(axis=1)
equity = (1 + port_rets).cumprod()

# --- Performance summary ---
cagr = (equity.iloc[-1] ** (252 / len(port_rets))) - 1
vol = port_rets.std() * np.sqrt(252)
sharpe = cagr / vol
max_dd = ((equity / equity.cummax()) - 1).min()

print(f"CAGR: {cagr:.2%}")
print(f"Volatility: {vol:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

# --- Plot ---
equity.plot(title="MA + Mean-Variance Strategy", figsize=(10,5))
plt.ylabel("Growth of $1")
plt.show()
