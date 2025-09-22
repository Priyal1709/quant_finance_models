%Import Data
T = readtable ('MSFT.csv');
dates = datetime(T.Date, 'InputFormat', 'mm/dd/yyyy');
prices = T.Close;

% Plot the closing prices against the dates
figure (1);
plot(dates, prices);
grid on;
xlabel('Date');
ylabel('Closing Price');
title('MSFT Closing Prices Over Time');

%SMA parameters
shortN = 20;
longN = 50;

SMA20 = movmean(prices, shortN, 'omitnan');
SMA50 = movmean(prices, longN,  'omitnan');

%signals
signal = double(SMA20 > SMA50);      % 1 = long, 0 = flat
signal = [NaN; signal(1:end-1)];     % lag by one day (enter next bar)

%returns
ret = [NaN; diff(prices)./prices(1:end-1)];
stratRet = signal .* ret;

%buy and sell points
changes = diff(signal);       % +1 = buy, -1 = sell
buyIdx  = find(changes == 1) + 1;   % where we enter long
sellIdx = find(changes == -1) + 1;  % where we exit long

%curves and plot
bhEquity = cumprod(1 + fillmissing(ret,'constant',0));
stratEquity = cumprod(1 + fillmissing(stratRet,'constant',0));


figure(2);
plot(dates, bhEquity, 'b-', 'LineWidth', 1.5); 
hold on;
plot(dates, stratEquity, 'r-', 'LineWidth', 1.5);
plot(dates(buyIdx), stratEquity(buyIdx), 'g^', 'MarkerSize', 8, 'MarkerFaceColor','g'); % buys
plot(dates(sellIdx), stratEquity(sellIdx), 'rv', 'MarkerSize', 8, 'MarkerFaceColor','r'); % sells
grid on; 
xlabel('Date'); 
ylabel('Equity (Start=1)');
title('MSFT: SMA(20/50) Strategy vs Buy & Hold');
legend('Buy & Hold','SMA Strategy', 'buy signal', 'hold signal','Location','best');




