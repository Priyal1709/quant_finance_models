%% Load data
T = readtable('MSFT.csv');   
times = datetime(T.Date);    % dates
prices = T.Close;            % closing prices

logp = log(prices);
n = length(logp);

%% Variance vs lag
taus = (1:25);              % up to 50-day lag
varsTau = zeros(size(taus));

for k = 1:length(taus)
    tau = taus(k);
    diffs = logp(1+tau:end) - logp(1:end-tau);
    varsTau(k) = var(diffs, 1);   % population variance
end

%% Linear fit
Lfit = 25;                               % fit first 50 lags
coeffs = polyfit(taus(1:Lfit), varsTau(1:Lfit), 1);
slope = coeffs(1); intercept = coeffs(2);

% Diffusion coefficient estimate (Var ≈ 2Dτ)
D_est = slope / 2;

%% Plot variance vs lag
figure(1);
plot(taus, varsTau, 'go-','DisplayName','Empirical variance'); 
hold on;
plot(taus(1:Lfit), polyval(coeffs, taus(1:Lfit)), 'r-','LineWidth',1.5,'DisplayName','Quadratic fit');
xlabel('Lag τ (days)');
ylabel('Var[log-price increment]');
title(sprintf('Variance scaling for MSFT — Estimated D ≈ %.4g', D_est));
legend('Location','best');
grid on;

%% --------------- Diagnostics -----------------

% Residual diagnostics (using τ = 1 day returns)
rets = diff(logp);                        % 1-day log returns
stdModel = sqrt(slope * 1);               % model predicted std at lag=1
z = rets / stdModel;                      % standardized residuals

fprintf('\n--- Residual Diagnostics ---\n');
fprintf('Mean of z: %.3f (should be ~0)\n', mean(z));
fprintf('Std of z:  %.3f (should be ~1)\n', std(z));
fprintf('Skewness:  %.3f\n', skewness(z));
fprintf('Kurtosis:  %.3f (Gaussian=3)\n', kurtosis(z));

maxLag = 40;
[acf,lags] = xcorr(z-mean(z), maxLag, 'coeff');

figure(2);
subplot(2,1,1);
histogram(z, 40);
xlabel('Standardized residuals z'); ylabel('Frequency');
title('Histogram of standardized residuals');

subplot(2,1,2);
stem(lags, acf, 'filled');
xlabel('Lag');
ylabel('Autocorrelation');
title('ACF of standardized residuals (computed via xcorr)');
grid on;
title('ACF of standardized residuals');

%% Bootstrap test: shuffle log-prices
nboot = 500;
slopesNull = zeros(nboot,1);

for b = 1:nboot
    rp = logp(randperm(n));   % permute log-price order
    varsB = zeros(length(taus),1);
    for k = 1:length(taus)
        tau = taus(k);
        diffs = rp(1+tau:end) - rp(1:end-tau);
        varsB(k) = var(diffs,1);
    end
    co = polyfit(taus(1:Lfit), varsB(1:Lfit), 1);
    slopesNull(b) = co(1);
end

pval = mean(slopesNull >= slope);

fprintf('\n--- Bootstrap Test ---\n');
fprintf('Observed slope: %.4g\n', slope);
fprintf('Mean null slope: %.4g\n', mean(slopesNull));
fprintf('Bootstrap p-value (slope > null): %.3f\n', pval);

figure(3);
histogram(slopesNull,30,'FaceColor',[0.6 0.6 0.9]);
hold on;
xline(slope,'r','LineWidth',2);
xlabel('Slope under null (shuffled logp)');
ylabel('Count');
title(sprintf('Bootstrap null vs observed slope (p=%.3f)', pval));

logp = log(prices);      % your log-prices
n = length(logp);
W = 25;                  % rolling window size in days
Drolling = nan(n-W,1);   % store D estimates
timeRolling = times(W+1:end);

for t = W+1:n
    windowLogp = logp(t-W:t-1);          % last W days
    rets = diff(windowLogp);             % 1-day returns in window
    var1 = var(rets,1);                  % population variance
    Drolling(t-W) = var1 / 2;            % diffusion coefficient estimate
end

%% Plot rolling D
figure(4);
plot(timeRolling, Drolling, 'b','LineWidth',1.5);
xlabel('Date'); ylabel('Estimated D (diffusion coefficient)');
title(sprintf('Rolling diffusion coefficient D over %d-day window', W));
grid on;
hold on
yyaxis left
plot(times, prices,'r'); ylabel('Price (USD)');
yyaxis right
plot(timeRolling, Drolling,'b','LineWidth',1.5); ylabel('D');
