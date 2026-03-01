import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pykalman import KalmanFilter
from datetime import datetime, timedelta, timezone

# --- 1. CONFIGURATION ---
API_KEY = "PK5ZCYTPJZHSMWB2V4HFWGW6GC"
SECRET_KEY = "5iDUwQPm2fxg5i834mzQNigaGhpLHfvhi35AwoDqN7R1"
TICKER_A = "META"
TICKER_B = "QQQ"

# --- 2. DATA FETCHING ---
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Fetch 30 days of 15-minute data
start_date = datetime.now(timezone.utc) - timedelta(days=180)

request_params = StockBarsRequest(
    symbol_or_symbols=[TICKER_A, TICKER_B],
    timeframe=TimeFrame(15, TimeFrameUnit.Minute),
    start=start_date
)

print(f"Fetching data for {TICKER_A} and {TICKER_B}...")
bars = client.get_stock_bars(request_params).df

# Pivot data: Rows = Time, Columns = Tickers
df = bars.pivot_table(index='timestamp', columns='symbol', values='close').dropna()

# --- 3. KALMAN FILTER LOGIC ---
# We want to find: Price_B = beta * Price_A + alpha
obs_mat = np.vstack([df[TICKER_A], np.ones(len(df))]).T[:, np.newaxis]

kf = KalmanFilter(
    n_dim_obs=1, 
    n_dim_state=2, # We are tracking [beta, alpha]
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.ones((2, 2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=1.0,     # R: Measurement noise
    transition_covariance=1e-4 * np.eye(2) # Q: Process noise (how fast beta can change)
)

state_means, _ = kf.filter(df[TICKER_B].values)

df['beta'] = state_means[:, 0]
df['alpha'] = state_means[:, 1]

# --- 4. SIGNAL GENERATION ---
# Spread = Actual_B - (beta * Actual_A + alpha)
df['spread'] = df[TICKER_B] - (df['beta'] * df[TICKER_A] + df['alpha'])

# Standardize the spread (Z-Score) using a rolling window
window = 20
df['z_score'] = (df['spread'] - df['spread'].rolling(window).mean()) / df['spread'].rolling(window).std()

# --- 5. VISUALIZATION ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df['beta'], label='Dynamic Hedge Ratio (Beta)', color='orange')
plt.title(f'Kalman Filter: Dynamic Hedge Ratio ({TICKER_A} vs {TICKER_B})')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['z_score'], label='Z-Score', color='blue')
plt.axhline(2, color='red', linestyle='--')
plt.axhline(-2, color='green', linestyle='--')
plt.title('Trading Signal (Z-Score of Spread)')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('KalmanTradingPairs.png')

print("Processing complete. Check the plot for trading signals.")

# --- 6. BACKTESTING LOOP ---
# Shift signals by 1 to enter on the NEXT bar (prevent look-ahead bias)
df['position'] = 0
# Long the spread: Buy B, Sell Beta * A
df.loc[df['z_score'] < -2, 'position'] = 1
# Short the spread: Sell B, Buy Beta * A
df.loc[df['z_score'] > 2, 'position'] = -1

# Exit Signal: Close position when Z-score returns to near 0
# We use a simple 'ffill' to keep the position open until we cross 0
df['position'] = df['position'].replace(0, np.nan).ffill().fillna(0)
# Simple exit logic: if z-score crosses 0, set position to 0
df.loc[df['z_score'].abs() < 0.5, 'position'] = 0

# Calculate Returns
# Percentage change for both assets
df['ret_a'] = df[TICKER_A].pct_change()
df['ret_b'] = df[TICKER_B].pct_change()

# Strategy Return = Position * (Return_B - Beta * Return_A)
# We use .shift(1) on position because we trade based on the PREVIOUS bar's signal
df['strategy_ret'] = df['position'].shift(1) * (df['ret_b'] - df['beta'].shift(1) * df['ret_a'])

# Cumulative Returns
df['cum_ret'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

# --- 7. PERFORMANCE METRICS ---
total_return = (df['cum_ret'].iloc[-1] - 1) * 100
sharpe_ratio = (df['strategy_ret'].mean() / df['strategy_ret'].std()) * np.sqrt(252 * 6.5 * 4) # Annualized for 15-min bars

print(f"--- Backtest Results ---")
print(f"Total Return: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# --- 8. PLOT EQUITY CURVE ---
plt.figure(figsize=(12, 6))
plt.plot(df['cum_ret'], label='Strategy Equity Curve', color='green')
plt.title(f'Kalman Pairs Trading: {TICKER_A} vs {TICKER_B} Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('backtest_plot.png')