% Monte Carlo Pricing of an Arithmetic Asian Call Option

clear
clc

rng(42); % reproducibility

% Parameters
S0 = 150;       % initial stock price
K = 100;        % strike price
r = 0.03;       % risk-free rate
q = 0.0;        % dividend yield
sigma = 0.50;   % volatility
T = 4.0;        % maturity (years)
nSteps = 50;    % number of monitoring points
nSims = 100000; % number of Monte Carlo simulations

dt = T / nSteps;
discount = exp(-r * T);

% Simulate paths
S_paths = zeros(nSims, nSteps+1);
S_paths(:,1) = S0;

for t = 2:nSteps+1
    Z = randn(nSims,1);
    S_paths(:,t) = S_paths(:,t-1) .* exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt).*Z);
end

% Compute arithmetic averages (exclude S0 at t=0)
arith_mean = mean(S_paths(:,2:end), 2);

% Payoff: Arithmetic Asian Call
payoffs = max(arith_mean - K, 0);

% Monte Carlo price
priceMC = discount * mean(payoffs);
stderrMC = discount * std(payoffs)/sqrt(nSims);

% Display result
fprintf('Arithmetic Asian Call Option Price (MC): %.4f\n', priceMC);
fprintf('Standard Error: %.6f\n', stderrMC);



S0_values = 50:5:180;      % range of initial stock prices
prices = zeros(size(S0_values));

for i = 1:length(S0_values)
    S0_test = S0_values(i);
    
    % Re-simulate paths for each initial S0
    S_paths = zeros(nSims, nSteps+1);
    S_paths(:,1) = S0_test;
    
    for t = 2:nSteps+1
        Z = randn(nSims,1);
        S_paths(:,t) = S_paths(:,t-1) .* exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt).*Z);
    end
    
    % Arithmetic average
    arith_mean = mean(S_paths(:,2:end), 2);
    
    % Payoff
    payoffs = max(arith_mean - K, 0);
    
    % Discounted price
    prices(i) = discount * mean(payoffs);
end

% Plot
figure(1)
plot(S0_values, prices, 'b-o','LineWidth',1.5,'MarkerSize',6);
xlabel('Initial Stock Price (S_0)');
ylabel('Asian Call Option Price');
title('Asian Call Option Price vs Initial Stock Price');
grid on;