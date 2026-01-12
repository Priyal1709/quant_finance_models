

import numpy as np
import matplotlib.pyplot as plt

def simulate_true_process(T=600, x0=np.log(0.02), Q0=1e-4, gamma=0.05, U=None, seed=0):
    rng = np.random.default_rng(seed)
    if U is None:
        U = np.zeros(T)

    x = np.zeros(T)           # latent log(kappa)
    kappa = np.zeros(T)       # latent kappa = exp(x)
    r = np.zeros(T)           # observed returns

    x[0] = x0
    kappa[0] = np.exp(x[0])

    for t in range(1, T):
        Qt = Q0 * (1.0 + gamma * U[t])          # reflexive process noise
        x[t] = x[t-1] + rng.normal(0.0, np.sqrt(Qt))
        kappa[t] = np.exp(x[t])

    # returns with variance kappa_t (simple)
    r = rng.normal(0.0, np.sqrt(kappa))
    return x, kappa, r


def reflexive_kalman_filter(y, U, x0, P0, Q0, gamma, R):
    """
    1D Kalman Filter for:
      x_t = x_{t-1} + w_t, w_t ~ N(0, Q_t)
      y_t = x_t + v_t,     v_t ~ N(0, R)
    with Q_t = Q0*(1 + gamma*U_t)
    """
    T = len(y)
    x_filt = np.zeros(T)
    P_filt = np.zeros(T)
    K_hist = np.zeros(T)

    x_prev, P_prev = x0, P0

    for t in range(T):
        # Predict
        Qt = Q0 * (1.0 + gamma * U[t])
        x_pred = x_prev
        P_pred = P_prev + Qt

        # Update
        S = P_pred + R               # innovation variance
        K = P_pred / S               # Kalman gain (scalar)
        x_new = x_pred + K * (y[t] - x_pred)
        P_new = (1.0 - K) * P_pred

        x_filt[t] = x_new
        P_filt[t] = P_new
        K_hist[t] = K

        x_prev, P_prev = x_new, P_new

    return x_filt, P_filt, K_hist


def main():
    # ---- knobs to tweak ----
    T = 700

    # usage schedule (reflexivity input)
    U = np.zeros(T)
    U[200:320] = 20.0       # a "crowding/usage" spike
    U[450:520] = 10.0

    # true process settings
    x0_true = np.log(0.02)  # baseline kappa ~ 2% (think daily variance-ish)
    Q0_true = 2e-4
    gamma_true = 0.06

    # filter settings (can be different from true; you calibrate these later)
    Q0 = 2e-4
    gamma = 0.06
    R = 0.50                # measurement noise on log(r^2); larger = trust measurements less
    P0 = 1.0
    eps = 1e-12             # prevents log(0)

    # ---- simulate "market" ----
    x_true, kappa_true, r = simulate_true_process(
        T=T, x0=x0_true, Q0=Q0_true, gamma=gamma_true, U=U, seed=42
    )

    # observation from returns: y = log(r^2 + eps)
    y = np.log(r**2 + eps)

    # ---- run Reflexive KF ----
    x_filt, P_filt, K = reflexive_kalman_filter(
        y=y, U=U, x0=x0_true, P0=P0, Q0=Q0, gamma=gamma, R=R
    )

    kappa_filt = np.exp(x_filt)

    # ---- plots ----
    # Plot A: usage schedule
    plt.figure()
    plt.plot(U)
    plt.title("Usage intensity U_t (reflexivity driver)")
    plt.xlabel("t")
    plt.ylabel("U_t")
    plt.tight_layout()
    plt.show()

    # Plot B: true vs filtered kappa
    plt.figure()
    plt.plot(kappa_true, label="True kappa_t")
    plt.plot(kappa_filt, label="KF estimate exp(x̂_t)")
    plt.title("Latent diffusivity/variance: true vs Reflexive Kalman estimate")
    plt.xlabel("t")
    plt.ylabel("kappa")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot C: log-space view with uncertainty band (±2 std)
    std = np.sqrt(P_filt)
    plt.figure()
    plt.plot(x_true, label="True x_t = log(kappa)")
    plt.plot(x_filt, label="Filtered x̂_t")
    plt.fill_between(np.arange(T), x_filt - 2*std, x_filt + 2*std, alpha=0.2, label="±2σ band")
    plt.title("Log diffusivity x_t with Kalman uncertainty band")
    plt.xlabel("t")
    plt.ylabel("x = log(kappa)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot D: how reflexivity changes process noise Q_t and Kalman gain
    Qt = Q0 * (1.0 + gamma * U)
    plt.figure()
    plt.plot(Qt, label="Q_t (process noise)")
    plt.plot(K, label="Kalman gain K_t")
    plt.title("Reflexivity effect: process noise & Kalman gain")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
