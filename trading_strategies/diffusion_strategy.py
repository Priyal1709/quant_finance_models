import numpy as np, pandas as pd, yfinance as yf
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

tickers = ["SPY","QQQ","IWM","EFA","EEM","TLT","GLD","HYG"]
prices = yf.download(tickers, start="2010-01-01", auto_adjust=True, progress=False)["Close"]
rets = prices.pct_change().dropna()
logp = np.log(prices).dropna()






def causal_gaussian(series, sigma, cutoff=5):
    import numpy as np, pandas as pd
    n = int(cutoff * sigma)
    t = np.arange(0, n+1)                # 0..past
    k = np.exp(-0.5 * (t / sigma)**2)
    k /= k.sum()
    x = series.values
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i - n)
        kc = k[:i - j0 + 1]              # trim kernel to available past
        yc = x[j0:i+1][::-1]             # align past to kernel
        y[i] = (kc * yc).sum()
    return pd.Series(y, index=series.index)

# Diffusion time scales (in trading days). Larger = smoother.
taus = [5, 21, 63]     # ~1w, 1m, 1q
D = 1.0
def smooth(series, tau):
    sigma = np.sqrt(2*D*tau)  # heat-kernel std
    return pd.Series(gaussian_filter1d(series.values, sigma=sigma, mode="nearest"),
                     index=series.index)

signals = pd.DataFrame(index=rets.index, columns=tickers)

for t in tickers:
    lp = logp[t]
    sigma = np.sqrt(2*1.0*21)
    base = causal_gaussian(lp, sigma=sigma)

    resid = lp - base
    resid_vol = resid.rolling(63).std().replace(0, np.nan)
    z = (resid / resid_vol).clip(-5, 5)

    trend_slope = base.diff(5)
    trend_sign = np.sign(trend_slope).fillna(0)

    raw_sig = (-z) + 0.3 * trend_sign
    raw_sig = raw_sig.clip(-2, 2)

    signals[t] = raw_sig
for col in logp:
    lp = logp[col]
    sigma = np.sqrt(2*D*21)
    base = causal_gaussian(lp, sigma=sigma)         # choose a principal scale
    dev = lp - base
    vol = lp.diff().rolling(21).std().replace(0,np.nan)
    z = (dev / vol).clip(-5,5)           # deviation in local-vol units
    slope = base.diff()                  # slow trend
    sig = (-z).clip(-1,1) + 0.5*np.sign(slope).fillna(0)  # mean-revert + gentle trend
    signals[col] = sig

# --- CAUSAL GAUSSIAN BASELINE (you already have causal_gaussian) ---
sigma = np.sqrt(2*1.0*21)  # tau=21 trading days
base = causal_gaussian(lp, sigma=sigma)

# --- Residual + z-score using residual volatility ---
resid = lp - base
resid_vol = resid.rolling(63).std().replace(0, np.nan)  # 3m residual vol
z = (resid / resid_vol).clip(-5, 5)

# --- Trend term from causal baseline slope ---
trend_slope = base.diff(5)   # 1-week slope of the slow baseline
trend_sign = np.sign(trend_slope).fillna(0)

# --- Raw signal: mean-revert deviations, plus gentle trend bias ---
raw_sig = (-z) + 0.3 * trend_sign
raw_sig = raw_sig.clip(-2, 2)         # cap per-asset signal

# --- Turn signal into position, shift for execution (no look-ahead) ---
pos = raw_sig.shift(1).fillna(0)

# Shift for execution (avoid lookahead)
positions = signals.shift(1).fillna(0)

# Normalize daily so weights sum to 1 (or stay within [-1,1])
positions = positions.div(positions.abs().sum(axis=1), axis=0).fillna(0)

# --- Volatility targeting to 12% annualized ---
# target_vol = 0.12
# Use *historical realized vol of the strategy* to scale exposure
raw_ret = (positions * rets).sum(axis=1)  # for single asset
target_vol = 0.12
roll_vol = raw_ret.rolling(63).std() * np.sqrt(252)
scale = (target_vol / (roll_vol + 1e-8)).clip(0, 3)

strategy_ret = raw_ret * scale.shift(1).fillna(1.0)
equity = (1 + strategy_ret).cumprod()

# --- Simple transaction cost (optional), e.g., 2 bps per unit turnover ---
turnover = pos.diff().abs().fillna(0)
strategy_ret = strategy_ret - 0.0002 * turnover

# --- Equity curve ---
equity = (1 + strategy_ret).cumprod()
print(f"CAGR: {equity.iloc[-1] ** (252/len(strategy_ret)) - 1: .2%}")

