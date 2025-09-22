% Black-Scholes Call Option Pricing and Plot
clc;
clear;

% === INPUT PARAMETERS ===
S0 = 100;        % Initial stock price
K = 100;         % Strike price
r = 0.08;        % Risk-free interest rate (annual)
T = 1;           % Time to maturity in years
sigma = 0.2;     % Volatility (20%)

% === OPTION PRICE FUNCTION ===
BlackScholesCall = @(S, K, r, T, sigma) ...
    S .* normcdf((log(S ./ K) + (r + 0.5 * sigma.^2) .* T) ./ (sigma .* sqrt(T))) ...
    - K * exp(-r * T) .* normcdf((log(S ./ K) + (r - 0.5 * sigma.^2) .* T) ./ (sigma .* sqrt(T)));

% === RANGE OF STOCK PRICES FOR PLOTTING ===
S = linspace(50, 150, 100);  % Range of stock prices

% === CALCULATE OPTION PRICES ===
CallPrices = BlackScholesCall(S, K, r, T, sigma);

% === PLOT ===
figure;
plot(S, CallPrices, 'b', 'LineWidth', 2);
xlabel('Stock Price (S)');
ylabel('Call Option Price (C)');
title('Black-Scholes European Call Option');
grid on;

% === DISPLAY SINGLE VALUE ===
C = BlackScholesCall(S0, K, r, T, sigma);
fprintf('Call Option Price at S = %.2f: %.4f\n', S0, C);
