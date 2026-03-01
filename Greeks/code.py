import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

class GreeksEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma
        
    def calculate_d1(self, S, T):
        return (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))

    def gamma_surface(self, S_range, T_range):
        S, T = np.meshgrid(S_range, T_range)
        d1 = self.calculate_d1(S, T)
        # Gamma Formula: N'(d1) / (S * sigma * sqrt(T))
        gamma = norm.pdf(d1) / (S * self.sigma * np.sqrt(T))
        return S, T, gamma

    def vanna_surface(self, S_range, T_range):
        S, T = np.meshgrid(S_range, T_range)
        d1 = self.calculate_d1(S, T)
        d2 = d1 - self.sigma * np.sqrt(T)
        # Vanna Formula: -N'(d1) * (d2 / sigma)
        vanna = -norm.pdf(d1) * (d2 / self.sigma)
        return S, T, vanna

# --- Parameters ---
S_space = np.linspace(70, 130, 50)  # Spot Price range
T_space = np.linspace(0.01, 1.0, 50) # Time to Expiry range (1 week to 1 year)
engine = GreeksEngine(S=100, K=100, T=1, r=0.05, sigma=0.2)

# Generate Gamma Surface
S, T, Z = engine.gamma_surface(S_space, T_space)

# --- Plotting ---
fig = go.Figure(data=[go.Surface(z=Z, x=S, y=T, colorscale='Viridis')])
fig.update_layout(
    title='Gamma Sensitivity Surface (Spot vs Time)',
    scene=dict(
        xaxis_title='Spot Price',
        yaxis_title='Time to Expiry',
        zaxis_title='Gamma'
    ),
    width=800, height=800
)
fig.show()