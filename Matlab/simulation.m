clear 
clc


%importing data from MSFT csv
data = readtable('MSFT.csv');
Date = data.Date;
ClosePrices = data.Close;

%Parameters
S0 = ClosePrices(1);
logreturns = diff(log(ClosePrices));

mu = mean(logreturns) * 252;
sigma = std(logreturns) * sqrt(252);
T = 1;
dt = 1/252;
N = T/dt;
M = 10000;

% Simulate stock price paths using Geometric Brownian Motion
paths = zeros(N+1, M);
paths(1, :) = S0;

for i = 1:M
    for j = 2:N+1
        Dw = sqrt(dt) * randn;
        paths(j, i) = paths(j-1, i) * exp((mu - 0.5 * sigma^2) * dt + sigma * Dw);
    end
end

figure(1)
plot(0:N, paths(:,1:200)); % plot 20 paths
xlabel('Days'); ylabel('Stock Price');
title('Monte Carlo Simulation of MSFT Stock Price (GBM)');
grid on;

final_prices = paths(end,:);

figure(2)
histogram(final_prices, 50);
xlabel('Price at T=1 year'); ylabel('Frequency');
title('Distribution of Simulated MSFT Prices after 1 Year');

% Expected value and 95% CI
mean_price = mean(final_prices);
CI = prctile(final_prices, [2.5 97.5]);

fprintf('Expected price after 1 year: %.2f\n', mean_price);
fprintf('95%% Confidence Interval: [%.2f , %.2f]\n', CI(1), CI(2));