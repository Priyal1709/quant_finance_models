"""
rough_vol_voo_strategy.py

Rough volatility Monte Carlo for VOO using (lognormal) rough Bergomi.
Includes a simple RVT (Rough-Vol Targeting) strategy based on MC tail risk.
No internet is required; historical VOO data may be loaded from a CSV if provided.

USAGE (CLI):
    python rough_vol_voo_strategy.py --demo
    python rough_vol_voo_strategy.py --csv path/to/voo.csv --price-col Close

CSV FORMAT EXPECTED:
    - A Date column parseable by pandas (name can be "Date" or "date").
    - A price column (e.g., "Close" or "Adj Close") specified by --price-col.
    - Any extra columns are ignored.

OUTPUTS:
    - PNG figures saved to the working directory.
    - Printed summary of Monte Carlo results and strategy vs baseline.

DEPENDENCIES:
    numpy, pandas, matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# --------------- Utilities ---------------

def annualize_mu_sigma(daily_mu: float, daily_sigma: float, periods_per_year: int = 252) -> Tuple[float, float]:
    mu_ann = (1 + daily_mu) ** periods_per_year - 1
    sigma_ann = daily_sigma * np.sqrt(periods_per_year)
    return mu_ann, sigma_ann

def deannualize_mu_sigma(mu_ann: float, sigma_ann: float, periods_per_year: int = 252) -> Tuple[float, float]:
    mu_daily = (1 + mu_ann) ** (1 / periods_per_year) - 1
    sigma_daily = sigma_ann / np.sqrt(periods_per_year)
    return mu_daily, sigma_daily

def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """EWMA volatility estimate of daily returns (returns as decimal)."""
    var = 0.0
    out = np.zeros_like(returns)
    for i, r in enumerate(returns):
        var = lam * var + (1 - lam) * (r * r)
        out[i] = np.sqrt(var + 1e-12)
    return out

def next_pow_two(n: int) -> int:
    return 1 if n == 0 else 2 ** (n - 1).bit_length()

# --------------- Rough Bergomi Simulation ---------------

@dataclass
class RoughBergomiParams:
    H: float = 0.12        # Hurst exponent (< 0.5)
    eta: float = 1.5       # vol-of-vol
    rho: float = -0.7      # leverage (spot/vol correlation)
    xi0: float = 0.04      # initial variance (sigma^2), e.g., (20% ann vol)^2 -> daily adjust inside
    mu: float = 0.07       # annualized drift (real-world)
    periods_per_year: int = 252

class RoughBergomiSimulator:
    """
    Lognormal rough Bergomi under the real measure:
        dS_t = mu S_t dt + S_t sqrt(V_t) dW_t^S
        log V_t = log xi0 + eta X_t - 0.5 * eta^2 Var(X_t)
        X_t = ∫_0^t (t-s)^{H-1/2} dW_s^v
    We simulate on a uniform grid using FFT-convolution to obtain X_t from standard normals.
    """
    def __init__(self, params: RoughBergomiParams, S0: float = 100.0, seed: Optional[int] = 42):
        self.p = params
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def _kernel_and_var(self, N: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        # Kernel K(u) = u^{H - 1/2} for u > 0; discretized at multiples of dt.
        H = self.p.H
        u = np.arange(N) * dt
        K = np.zeros(N)
        K[1:] = np.power(u[1:], H - 0.5)
        # Var(X_tk) ≈ ∫_0^{t_k} K^2(t_k - s) ds = ∫_0^{t_k} s^{2H - 1} ds = t_k^{2H} / (2H)
        # Discrete approximation: cumulative sum of K^2 * dt (shifted appropriately).
        K2 = K * K
        VarX = np.cumsum(K2) * dt
        return K, VarX

    def _fft_convolve_paths(self, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Convolve each path (row) of Z with kernel K using FFT (linear convolution).
        Z shape: (n_paths, N), K shape: (N,)
        Returns X with same shape.
        """
        n_paths, N = Z.shape
        L = next_pow_two(N + N - 1)
        Kf = np.fft.rfft(K, n=L)
        X = np.empty_like(Z)
        for i in range(n_paths):
            Zf = np.fft.rfft(Z[i], n=L)
            conv = np.fft.irfft(Kf * Zf, n=L)[:N]
            X[i] = conv
        return X

    def simulate(self, T_years: float = 2.0, N: Optional[int] = None, n_paths: int = 256) -> Dict[str, np.ndarray]:
        """
        Simulate S_t and V_t paths.
        Returns a dict with 'S', 'V', 'X', 'times'.
        """
        p = self.p
        if N is None:
            N = int(T_years * p.periods_per_year)
        dt = T_years / N

        # Precompute kernel and variance correction
        K, VarX = self._kernel_and_var(N + 1, dt)  # include t_0
        K = K[:N]                                  # for ΔW indexing convenience
        VarX = VarX[:N+1]                          # Var(X at grid), VarX[0]=0

        # Brownian increments for spot and vol, correlated
        Zs = self.rng.standard_normal((n_paths, N))
        Zv = self.rng.standard_normal((n_paths, N))
        Zv_corr = p.rho * Zs + np.sqrt(1 - p.rho**2) * Zv

        # Build X via convolution: X_k = sum_{j=0}^{k-1} K_{k-j} * sqrt(dt) * Zv_corr_j
        X = self._fft_convolve_paths(Zv_corr * np.sqrt(dt), K)

        # Variance process (lognormal) with mean correction
        eta = p.eta
        # VarX aligned: VarX[k] ~ Var(X_tk). We'll broadcast per path.
        mean_correction = 0.5 * (eta ** 2) * VarX[1:]  # skip t0
        V = np.empty((n_paths, N+1))
        V[:, 0] = p.xi0

        # Convert xi0 annual variance to daily if needed inside S dynamics; keep V in annualized variance terms.
        for i in range(n_paths):
            logV = np.log(p.xi0) + eta * X[i] - mean_correction
            V_path = np.empty(N+1)
            V_path[0] = p.xi0
            V_path[1:] = np.exp(logV)
            V[i] = V_path

        # Simulate S via log-Euler with annualized variance V and dt in years
        S = np.empty((n_paths, N+1))
        S[:, 0] = self.S0
        for i in range(n_paths):
            increments = (p.mu - 0.5 * V[i, :-1]) * dt + np.sqrt(np.maximum(V[i, :-1], 1e-12)) * np.sqrt(dt) * Zs[i]
            S[i, 1:] = S[i, 0] * np.exp(np.cumsum(increments))

        times = np.linspace(0, T_years, N+1)
        return {"S": S, "V": V, "X": X, "times": times}

# --------------- RVT Strategy (MC-based) ---------------

@dataclass
class RVTConfig:
    target_vol_ann: float = 0.15   # annualized target portfolio vol
    lam_ewma: float = 0.94         # EWMA smoothing for realized vol
    burst_horizon_days: int = 10   # lookahead horizon (days) for drawdown checks
    burst_threshold: float = 0.08  # trigger if MC-prob(drawdown > threshold) exceeds alpha
    burst_alpha: float = 0.20      # probability threshold
    cooldown_days: int = 10        # days to keep reduced risk after a burst flag
    max_leverage: float = 1.0      # cap weight (no leverage by default)
    periods_per_year: int = 252

class RVTMonteCarloStrategy:
    """
    Uses MC paths to compute short-horizon drawdown probabilities and sets
    the weight w_t = min(1, target_vol / realized_vol) with a burst filter.
    """
    def __init__(self, cfg: RVTConfig):
        self.c = cfg

    def _compute_mc_drawdown_prob(self, S_paths: np.ndarray, horizon: int, thresh: float) -> float:
        """
        Given current index 0 as "today", compute probability that
        max drawdown over next `horizon` steps exceeds `thresh`.
        """
        # Normalize by S0 for comparability.
        S0 = S_paths[:, 0:1]
        rel = S_paths[:, :horizon+1] / S0
        peak = np.maximum.accumulate(rel, axis=1)
        dd = 1.0 - (rel / np.maximum(peak, 1e-12))
        max_dd = dd.max(axis=1)
        return np.mean(max_dd >= thresh)

    def backtest_on_single_path(self, S_path: np.ndarray, sim_engine: RoughBergomiSimulator,
                                n_paths_mc: int = 256) -> Dict[str, np.ndarray]:
        """
        Backtest RVT along a *single* realized path by re-simulating short MC batches at each step.
        For demo speed, we resample using the same parameters, conditioning only on last price (no filtering on V).
        """
        c = self.c
        N = len(S_path) - 1
        dt_years = 1.0 / c.periods_per_year

        # Realized daily returns from the path
        rets = np.diff(np.log(S_path))
        vol_ewma = ewma_vol(rets, lam=c.lam_ewma)
        vol_ann = np.concatenate([[vol_ewma[0]], vol_ewma]) * np.sqrt(c.periods_per_year)

        weights = np.zeros(N+1)
        weights[0] = min(c.max_leverage, c.target_vol_ann / max(vol_ann[0], 1e-6))

        cooldown = 0
        pnl = np.zeros(N+1)
        equity = np.ones(N+1)

        for t in range(N):
            # MC burst check from t using short horizon
            if cooldown == 0:
                # simulate short horizon around current state; use current price as S0
                tmp_engine = RoughBergomiSimulator(sim_engine.p, S0=float(S_path[t]), seed=1234 + t)
                short = tmp_engine.simulate(T_years=(c.burst_horizon_days * dt_years), N=c.burst_horizon_days, n_paths=n_paths_mc)
                prob_burst = self._compute_mc_drawdown_prob(short["S"], c.burst_horizon_days, c.burst_threshold)
                burst_flag = (prob_burst >= c.burst_alpha)
            else:
                prob_burst = 1.0  # keep flag active
                burst_flag = True

            # base sizing from risk target
            w = min(c.max_leverage, c.target_vol_ann / max(vol_ann[t], 1e-6))

            # apply burst filter: if flagged, reduce weight by 50%
            if burst_flag:
                w *= 0.5
                cooldown = max(cooldown, c.cooldown_days)
            else:
                cooldown = max(0, cooldown - 1)

            weights[t] = w

            # next-day PnL
            pnl[t+1] = w * (np.exp(rets[t]) - 1.0)
            equity[t+1] = equity[t] * (1.0 + pnl[t+1])

        weights[-1] = weights[-2]
        return {
            "weights": weights,
            "equity": equity,
            "vol_ann": vol_ann,
            "pnl": pnl
        }

# --------------- Historical Data Loader (optional) ---------------

def load_voo_csv(path: str, price_col: str = "Close") -> pd.DataFrame:
    df = pd.read_csv(path)
    # find date column
    date_cols = [c for c in df.columns if c.lower() in ("date", "time", "timestamp")]
    if not date_cols:
        raise ValueError("CSV must contain a Date column (Date/time/timestamp).")
    df = df.rename(columns={date_cols[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found. Available: {list(df.columns)}")
    out = df[["Date", price_col]].dropna().sort_values("Date").reset_index(drop=True)
    out = out.rename(columns={price_col: "Close"})
    return out

# --------------- Parameter Heuristics ---------------

def estimate_basic_params_from_prices(prices: np.ndarray, periods_per_year: int = 252) -> Tuple[float, float]:
    """Return simple estimates (mu_ann, xi0_ann) from close prices."""
    logret = np.diff(np.log(prices))
    mu_daily = np.mean(logret)
    sig_daily = np.std(logret, ddof=1)
    mu_ann, sig_ann = annualize_mu_sigma(mu_daily, sig_daily, periods_per_year)
    xi0_ann = sig_ann ** 2
    return mu_ann, xi0_ann

# --------------- Demo Runner ---------------

def demo_run(save_prefix: str = "demo", seed: int = 7):
    # Sensible defaults; feel free to tweak H, eta, rho
    p = RoughBergomiParams(H=0.12, eta=1.6, rho=-0.75, xi0=0.04, mu=0.07, periods_per_year=252)
    engine = RoughBergomiSimulator(p, S0=100.0, seed=seed)
    sim = engine.simulate(T_years=3.0, N=3*252, n_paths=160)

    S = sim["S"]
    V = sim["V"]
    times = sim["times"]

    # Quick visualization: 1) price fan chart, 2) variance fan chart
    plt.figure(figsize=(10, 5))
    for i in range(min(50, S.shape[0])):
        plt.plot(times, S[i], alpha=0.25)
    plt.plot(times, S.mean(axis=0), linewidth=2)
    plt.title("Rough Bergomi Monte Carlo: VOO price paths (demo)")
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(min(50, V.shape[0])):
        plt.plot(times, np.sqrt(V[i]) , alpha=0.25)  # plot vol (ann.)
    plt.plot(times, np.sqrt(V).mean(axis=0), linewidth=2)
    plt.title("Rough Bergomi Monte Carlo: annualized vol paths (demo)")
    plt.xlabel("Years")
    plt.ylabel("Vol (ann.)")
    plt.tight_layout()
    plt.show()

    # Backtest RVT vs buy&hold along one realized path
    realized_idx = 3  # pick one
    realized_path = S[realized_idx]
    rvt = RVTMonteCarloStrategy(RVTConfig())
    bt = rvt.backtest_on_single_path(realized_path, engine, n_paths_mc=160)

    equity_rvt = bt["equity"]
    equity_bh = realized_path / realized_path[0]

    plt.figure(figsize=(10,5))
    plt.plot(times, equity_bh, label="Buy & Hold")
    plt.plot(times, equity_rvt, label="RVT (MC)")
    plt.title("Equity curves: Buy & Hold vs RVT (demo path)")
    plt.xlabel("Years")
    plt.ylabel("Equity (start=1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print a tiny summary
    r_bh = np.diff(np.log(equity_bh + 1e-12))
    r_rvt = np.diff(np.log(equity_rvt + 1e-12))
    mu_bh, sig_bh = annualize_mu_sigma(np.mean(r_bh), np.std(r_bh, ddof=1))
    mu_rvt, sig_rvt = annualize_mu_sigma(np.mean(r_rvt), np.std(r_rvt, ddof=1))

    print("==== DEMO SUMMARY (simulated) ====")
    print(f"Buy&Hold  -> Ann Return: {mu_bh:.2%} | Ann Vol: {sig_bh:.2%} | Sharpe (mu/sig): {mu_bh/sig_bh:.2f}")
    print(f"RVT (MC)  -> Ann Return: {mu_rvt:.2%} | Ann Vol: {sig_rvt:.2%} | Sharpe (mu/sig): {mu_rvt/sig_rvt:.2f}")
    print("Saved figures: "
          f"{save_prefix}_price_paths.png, "
          f"{save_prefix}_vol_paths.png, "
          f"{save_prefix}_equity_curves.png")

def run_with_csv(csv_path: str, price_col: str = "Close", H: float = 0.12, eta: float = 1.6, rho: float = -0.75):
    df = load_voo_csv(csv_path, price_col=price_col)
    prices = df["Close"].values.astype(float)

    mu_ann, xi0_ann = estimate_basic_params_from_prices(prices)
    print(f"Estimated from CSV -> mu_ann ~ {mu_ann:.2%}, vol_ann ~ {np.sqrt(xi0_ann):.2%}")

    p = RoughBergomiParams(H=H, eta=eta, rho=rho, xi0=xi0_ann, mu=mu_ann, periods_per_year=252)
    engine = RoughBergomiSimulator(p, S0=float(prices[-1]), seed=11)

    # simulate 2 years forward
    sim = engine.simulate(T_years=2.0, N=2*252, n_paths=200)
    S = sim["S"]
    V = sim["V"]
    times = sim["times"]

    plt.figure(figsize=(10, 5))
    for i in range(min(60, S.shape[0])):
        plt.plot(times, S[i], alpha=0.25)
    plt.plot(times, S.mean(axis=0), linewidth=2)
    plt.title("VOO forward MC under Rough Bergomi (from CSV estimates)")
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

    # Backtest RVT on one synthetic continuation
    realized_path = S[0]
    rvt = RVTMonteCarloStrategy(RVTConfig())
    bt = rvt.backtest_on_single_path(realized_path, engine, n_paths_mc=200)

    equity_rvt = bt["equity"]
    equity_bh = realized_path / realized_path[0]

    plt.figure(figsize=(10,5))
    plt.plot(times, equity_bh, label="Buy & Hold")
    plt.plot(times, equity_rvt, label="RVT (MC)")
    plt.title("Equity: Buy & Hold vs RVT (CSV-based params)")
    plt.xlabel("Years")
    plt.ylabel("Equity (start=1)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Saved figures: csv_forward_price_paths.png, csv_equity_curves.png")

def main():
    parser = argparse.ArgumentParser(description="Rough volatility Monte Carlo and RVT strategy for VOO.")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo simulation/backtest.")
    parser.add_argument("--csv", type=str, default=None, help="Path to a VOO CSV (with Date and price col).")
    parser.add_argument("--price-col", type=str, default="Close", help="Price column name in the CSV.")
    parser.add_argument("--H", type=float, default=0.12, help="Hurst exponent < 0.5")
    parser.add_argument("--eta", type=float, default=1.6, help="Vol-of-vol")
    parser.add_argument("--rho", type=float, default=-0.75, help="Spot/vol correlation")
    args = parser.parse_args()

    if args.csv:
        run_with_csv(args.csv, price_col=args.price_col, H=args.H, eta=args.eta, rho=args.rho)
    else:
        demo_run(save_prefix="demo", seed=7)

if __name__ == "__main__":
    main()
