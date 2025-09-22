% Monte Carlo Simulation v/s Black-Scholes

clear; clc; rng(1);
%importing data from MSFT csv
data = readtable('MSFT.csv');
Date = data.Date;
ClosePrices = data.Close;

%Parameters
S0 = ClosePrices(1);   
K  = 340;             
r  = 0.05;             
sigma = 0.25;          
T  = 1;               
M  = 1e5;               

% Step 1: Simulate terminal stock prices under risk-neutral measure
Z = randn(M,1);  % standard normal random numbers
ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T).*Z);

% Step 2: Compute payoffs
call_payoff = max(ST - K, 0);
put_payoff  = max(K - ST, 0);

% Step 3: Discount back to present
call_price = exp(-r*T) * mean(call_payoff);
put_price  = exp(-r*T) * mean(put_payoff);

fprintf('European Call Price (MC): %.4f\n', call_price);
fprintf('European Put Price  (MC): %.4f\n', put_price);

% Compare with analytical Black-Scholes formula
d1 = (log(S0/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);

call_BS = S0*normcdf(d1) - K*exp(-r*T)*normcdf(d2);
put_BS  = K*exp(-r*T)*normcdf(-d2) - S0*normcdf(-d1);

fprintf('European Call Price (BS Formula): %.4f\n', call_BS);
fprintf('European Put Price  (BS Formula): %.4f\n', put_BS);
