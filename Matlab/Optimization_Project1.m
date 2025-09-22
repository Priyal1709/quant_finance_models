clc
clear
AAPL = readtable('AAPL.csv');
MSFT = readtable('MSFT.csv');
TSLA = readtable('TSLA.csv');

AAPL.Date = datetime(AAPL.Date);
MSFT.Date = datetime(MSFT.Date);
TSLA.Date = datetime(TSLA.Date);

% Extract relevant columns
AAPL = AAPL(:, {'Date', 'Close'});
MSFT = MSFT(:, {'Date', 'Close'});
TSLA = TSLA(:, {'Date', 'Close'});

% Rename for clarity
AAPL.Properties.VariableNames{'Close'} = 'AAPL';
MSFT.Properties.VariableNames{'Close'} = 'MSFT';
TSLA.Properties.VariableNames{'Close'} = 'TSLA';

% Join tables on the common Date
temp = outerjoin(AAPL, MSFT, 'Keys', 'Date', ...
    'MergeKeys', true, 'Type', 'inner');
merged = outerjoin(temp, TSLA, 'Keys', 'Date',...
    'MergeKeys', true, 'Type', 'inner');

% Sort by date
merged = sortrows(merged, 'Date');

% Extract prices
dates = merged.Date;
prices = [merged.AAPL, merged.MSFT, merged.TSLA];

% Compute log returns
returns = diff(log(prices));
dates_r = dates(2:end);  % adjust dates for returns

figure (1);
plot(dates, prices);
legend({'AAPL', 'MSFT', 'TSLA'});
title('Adjusted Close Prices');
xlabel('Date'); ylabel('Price'); grid on;

figure (2);
plot(dates_r, returns);
legend({'AAPL', 'MSFT', 'TSLA'});
title('Log Returns');
xlabel('Date'); ylabel('Return'); grid on;

% === Markowitz Mean-Variance Optimization ===

% Step 1: Estimate expected returns and covariance
mu = mean(returns)';         % 2x1 vector (mean return of each asset)
Sigma = cov(returns);        % 2x2 covariance matrix

% Step 2: Set up portfolio constraints
n = length(mu);              % Number of assets
R_target = linspace(min(mu)+0.001, max(mu)-0.001, 50);  % Target returns

% Initialize arrays
risk = zeros(size(R_target));
weights = zeros(n, length(R_target));

% Step 3: Optimization loop using quadprog
Aeq = [mu'; ones(1,n)];  % constraints: target return and full investment
lb = [0.1; 0.1; 0.2];         % No short-selling
ub = [inf; inf; inf];
options = optimoptions('quadprog','Display','off');

% A = eye(n);
% b = 0.5 * ones(n,1);

for i = 1:length(R_target)
    beq = [R_target(i); 1];   % current target return
    w = quadprog(2*Sigma, [], [], [], Aeq, beq, lb, ub, [], options);
    weights(:, i) = w;
    risk(i) = sqrt(w' * Sigma * w);
end

% Step 4: Plot Efficient Frontier
figure;
plot(risk, R_target, 'b-', 'LineWidth', 2);
xlabel('Portfolio Risk (Std Dev)');
ylabel('Expected Return');
title('Efficient Frontier (3-Asset)');
grid on;

Rf = 0.05;
sharpeRatios = (R_target - Rf) ./ risk;

[~, idx_maxSharpe] = max(sharpeRatios);
w_tan = weights(:, idx_maxSharpe);        % Weights of tangency portfolio
R_tan = R_target(idx_maxSharpe);          % Return
risk_tan = risk(idx_maxSharpe);           % Risk (volatility)

% Extend the line from Rf through the tangency point
x_cml = [0, 1.5 * risk_tan];   % Extend beyond the tangency point
y_cml = Rf + ((R_tan - Rf)/risk_tan) * x_cml;

figure(5)
plot(risk, R_target, 'b-', 'LineWidth', 2);
hold on
plot(x_cml, y_cml, 'r--', 'LineWidth', 2, 'DisplayName', 'Capital Market Line');
plot(risk_tan, R_tan, 'ro', 'MarkerSize', 8, 'DisplayName', 'Tangency Portfolio');
legend show;
ylim([-0.01, 0.01])
xlabel('Portfolio Risk (Std Dev)');
ylabel('Expected Return');
title('Efficient Frontier + Capital Market Line');
legend('Efficient Frontier', 'Capital Market Line', 'Tangency Portfolio');
grid on;

tickers = {'AAPL', 'MSFT', 'TSLA'};
fprintf('\nTangency Portfolio (Max Sharpe):\n');
for i = 1:length(tickers)
    fprintf('%s: %.2f%%\n', tickers{i}, 100 * w_tan(i));
end

sharpeMax = sharpeRatios(idx_maxSharpe);
fprintf('Expected Return: %.2f%%\n', 100 * R_tan);
fprintf('Risk (Std Dev): %.2f%%\n', 100 * risk_tan);
fprintf('Sharpe Ratio: %.4f\n', sharpeMax);

targetReturn = 0.08;  % e.g., 10% expected return
[~, idx] = min(abs(R_target - targetReturn));
optimalWeights = weights(:, idx);

tickers = {'AAPL', 'MSFT', 'TSLA'};
fprintf('Portfolio for ~%.2f%% return:\n', targetReturn*100);
for i = 1:length(tickers)
    fprintf('%s: %.2f%%\n', tickers{i}, 100 * optimalWeights(i));
end

figure;
area(R_target, weights'*100);  % transpose weights for plotting
xlabel('Expected Return (%)');
ylabel('Weight (%)');
title('Asset Allocation Across Efficient Frontier');
legend({'AAPL','MSFT','TSLA'}, 'Location', 'best');
grid on;






