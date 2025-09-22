clear
clc


%Data Extraction
data = readtable("MSFT.csv");
Date = data.Date;
Prices = data.Close;

%Daily returns (Log)
log_Returns = diff(log(Prices));
Mu = mean(log_Returns);
Sigma = std(log_Returns);

%GBM Set-Up
S_0 = Prices(1);
N = length(log_Returns);
dt = 1;
n_Paths = 50;

%GBM 
GBM = zeros(N+1, n_Paths);
GBM(1,:) = S_0;

for i = 1:n_Paths
   Z = randn(N,1);
   GBM(2:end, i) = S_0 * exp(cumsum((Mu - 0.5 * Sigma^2) * dt + Sigma * sqrt(dt) * Z));
end

figure(1)
plot(1:N+1, GBM, 'Color', [0.7 0.7 0.7])
hold on
plot(1:N+1, Prices(1:N+1), 'g','LineWidth', 2)
xlabel("Time [Days]")
ylabel("Price [$]")
title("MSFT: Real v/s Simulated Price")
legend("GBM Simulation","Actual Price","Location","best")
grid on

% GBM_mean = mean(GBM, 2);  % mean across all simulated paths
% realPrice = Prices(1:length(GBM_mean));
% 
% % Mean Absolute Error (MAE)
% MAE = mean(abs(GBM_mean - realPrice));
% 
% % Root Mean Square Error (RMSE)
% RMSE = sqrt(mean((GBM_mean - realPrice).^2));
% 
% % Normalized RMSE (optional, scales to price level)
% NRMSE = RMSE / mean(realPrice);
% 
% % Mean Bias (signed error)
% bias = mean(GBM_mean - realPrice);
% 
% GBM_std = std(GBM, 0, 2);  % standard deviation across simulations
% 
% upper = GBM_mean + GBM_std;
% lower = GBM_mean - GBM_std;
% 
% figure;
% plot(realPrice, 'g', 'LineWidth', 2); hold on;
% plot(GBM_mean, 'w', 'LineWidth', 1.5);
% fill([1:N+1, fliplr(1:N+1)], [upper', fliplr(lower')], ...
%      [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
% legend('Actual Price', 'Mean GBM', '±1 Std Dev');
% title('Real Price vs. Simulated GBM (Mean ± Std Dev)');
% xlabel('Days'); ylabel('Price [$]');
% grid on;




