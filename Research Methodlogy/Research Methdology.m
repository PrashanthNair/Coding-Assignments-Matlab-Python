%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 1: Currency Returns           %
% Author: Prashanth Nair,5669869                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
file_path = 'C:\Users\nairp\OneDrive\Desktop\5669869 DATA FILE RM\RM DATA.xlsx';
sheet_name = 'MAIN';

% Load data from Excel
[data, text_data] = xlsread(file_path, sheet_name);

% Extract date and exchange rate data
raw_dates = text_data(2:end, 1); % Skip header, take date column
exchange_rates = data;           % Numerical exchange rates
headers = text_data(1, 2:end);   % Column headers for currencies

%% Convert Dates to MATLAB Date Format
try
    dates = datetime(raw_dates, 'InputFormat', 'MM/dd/yyyy'); % Adjust format as per input
catch
    error('Error parsing dates. Please check the date format in the Excel sheet.');
end

%% Log Transformation and Handling Missing Data
% Convert exchange rates to log values
log_rates = log(exchange_rates);

% Interpolate missing data linearly
log_rates = fillmissing(log_rates, 'linear');

%% Calculate Logarithmic Changes and Currency Returns
% Compute daily log differences (logarithmic changes)
log_diff = diff(log_rates);

% Compute daily currency returns (negative log differences)
currency_returns = -log_diff;

% Adjust dates to align with changes
return_dates = dates(2:end);

%% Combine Data into a Single Table
% Create a table with raw exchange rates, log values, log differences, and returns
combined_table = array2table([exchange_rates(2:end, :) log_rates(2:end, :) log_diff currency_returns], ...
    'VariableNames', [headers, strcat(headers, '_Log'), strcat(headers, '_LogDiff'), strcat(headers, '_Returns')]);

% Add dates to the table
combined_table = addvars(combined_table, return_dates, 'Before', 1, 'NewVariableNames', 'Date');

%% Write Output to Excel
output_file = 'Currency_Returns_Complete.xlsx';
writetable(combined_table, output_file);

%% Visualization
% Plot daily currency returns
figure;
hold on;
for i = 1:size(currency_returns, 2)
    plot(return_dates, currency_returns(:, i), 'DisplayName', headers{i});
end
hold off;
grid on;
legend('show', 'Location', 'bestoutside');
title('Daily Currency Returns');
xlabel('Date');
ylabel('Currency Returns');

% Adjust x-axis ticks to show more years
xticks(datetime(2000:2:2025, 1, 1)); % Add tick marks every two years
xlim([min(return_dates) max(return_dates)]); % Ensure the full date range is covered

% Save the plot
saveas(gcf, 'Daily_Currency_Returns.png');


%% Completion Message
disp('Currency returns calculation and output completed successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 3: Dollar Portfolio Returns    %                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Currency_Returns_Complete.xlsx';
output_file = 'Dollar_Portfolio_Results.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns for developed market currencies (returns)
currency_columns = contains(input_data.Properties.VariableNames, '_Returns');
developed_currencies = input_data(:, currency_columns);

% Ensure only the 9 developed market currencies are selected
selected_currencies = developed_currencies(:, 1:9); % First 9 returns columns (AUD to SEK)

% Extract dates
dates = input_data.Date;

%% Calculate Dollar Portfolio Returns
% Compute the cross-sectional average for the 9 currencies
dollar_portfolio_returns = mean(table2array(selected_currencies), 2, 'omitnan');

% Add the Dollar Portfolio Returns to the dataset
input_data.Dollar_Portfolio_Returns = dollar_portfolio_returns;

%% Save Updated Data to Excel
% Write the updated dataset to a new Excel file
writetable(input_data, output_file);

%% Visualization: Dollar Portfolio Returns Over Time
figure;
plot(dates, dollar_portfolio_returns, 'b', 'LineWidth', 1.5);
grid on;

% Customize x-axis to show more years
xtickformat('yyyy'); % Format the ticks to show the year
xticks(dates(1):calyears(2):dates(end)); % Create ticks every 2 years

title('Dollar Portfolio Returns Over Time');
xlabel('Date');
ylabel('Returns');
legend('Dollar Portfolio Returns');
saveas(gcf, 'Dollar_Portfolio_Returns_TimeSeries.png');

%% Visualization: Histogram of Dollar Portfolio Returns
figure;
histogram(dollar_portfolio_returns, 50, 'FaceColor', 'blue');
grid on;
title('Histogram of Dollar Portfolio Returns');
xlabel('Daily Returns');
ylabel('Frequency');
saveas(gcf, 'Dollar_Portfolio_Returns_Histogram.png');

%% Completion Message
disp('Dollar Portfolio Returns calculation and visualization completed successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 4: Volatility Forecasts         %
% Revised GARCH Implementation for Emerging Market Volatility%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Dollar_Portfolio_Results.xlsx';
output_file = 'Volatility_Forecasts_Revised.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
dollar_returns = input_data.Dollar_Portfolio_Returns;
emerging_market_returns = input_data.ARSTOUSD_LSEGDS__EXCHANGERATE_Returns;

% Filter data for burn-in period (1999) and calculation period (2000 onwards)
burn_in_index = find(year(dates) == 1999, 1, 'last');
start_index = burn_in_index + 1;
filtered_dates = dates(start_index:end);
dollar_returns_filtered = dollar_returns(start_index:end);
emerging_returns_filtered = emerging_market_returns(start_index:end);

%% Moving Average Volatility (10-week window)
ma_window = 50; % 10 weeks * 5 trading days
dollar_ma_volatility = movstd(dollar_returns_filtered, ma_window, 'omitnan');
emerging_ma_volatility = movstd(emerging_returns_filtered, ma_window, 'omitnan');

%% EWMA Volatility
lambda = 0.94;
dollar_ewma_volatility = zeros(size(dollar_returns_filtered));
emerging_ewma_volatility = zeros(size(emerging_returns_filtered));

% Initialize with variance over burn-in period
dollar_ewma_volatility(1) = std(dollar_returns(1:burn_in_index));
emerging_ewma_volatility(1) = std(emerging_market_returns(1:burn_in_index));

for t = 2:length(dollar_returns_filtered)
    dollar_ewma_volatility(t) = sqrt(lambda * dollar_ewma_volatility(t-1)^2 + (1 - lambda) * dollar_returns_filtered(t-1)^2);
    emerging_ewma_volatility(t) = sqrt(lambda * emerging_ewma_volatility(t-1)^2 + (1 - lambda) * emerging_returns_filtered(t-1)^2);
end

%% Revised GARCH Volatility for Emerging Markets
% Fit GARCH(1,1) model for Dollar Portfolio
mdl_dollar = garch(1, 1);
est_dollar = estimate(mdl_dollar, dollar_returns(1:burn_in_index));
[garch_dollar_volatility, ~] = infer(est_dollar, dollar_returns_filtered);
dollar_garch_volatility = sqrt(garch_dollar_volatility);

% Independent GARCH Implementation for Emerging Market Currency
omega = 0.00001; % Initial value for omega
alpha = 0.1;    % Initial value for alpha
beta = 0.85;    % Initial value for beta

% Initialize GARCH variance with burn-in period variance
emerging_garch_variance = zeros(size(emerging_returns_filtered));
emerging_garch_variance(1) = var(emerging_market_returns(1:burn_in_index));

for t = 2:length(emerging_returns_filtered)
    emerging_garch_variance(t) = omega + alpha * emerging_returns_filtered(t-1)^2 + beta * emerging_garch_variance(t-1);
end

emerging_garch_volatility = sqrt(emerging_garch_variance);

%% Save Results to Excel
results_table = table(filtered_dates, dollar_returns_filtered, emerging_returns_filtered, ...
    dollar_ma_volatility, emerging_ma_volatility, ...
    dollar_ewma_volatility, emerging_ewma_volatility, ...
    dollar_garch_volatility, emerging_garch_volatility, ...
    'VariableNames', {'Date', 'Dollar_Returns', 'Emerging_Returns', ...
    'MA_Volatility_Dollar', 'MA_Volatility_EM', ...
    'EWMA_Volatility_Dollar', 'EWMA_Volatility_EM', ...
    'GARCH_Volatility_Dollar', 'GARCH_Volatility_EM'});

writetable(results_table, output_file);

%% Visualization: Dollar Portfolio Volatility
figure;
plot(filtered_dates, dollar_ma_volatility, 'b', 'LineWidth', 1.2); hold on;
plot(filtered_dates, dollar_ewma_volatility, 'Color', [1, 0.5, 0], 'LineWidth', 1.2); % Corrected for RGB color
plot(filtered_dates, dollar_garch_volatility, 'g', 'LineWidth', 1.2);
grid on;

% Customize x-axis ticks for yearly intervals
xtickformat('yyyy'); % Set tick format to show only the year
xticks(filtered_dates(1):calyears(1):filtered_dates(end)); % Add ticks every year

title('Volatility Forecasts for Dollar Portfolio (2000-2024)');
xlabel('Date');
ylabel('Volatility');
legend('Moving Average Volatility', 'EWMA Volatility', 'GARCH Volatility');
saveas(gcf, 'Dollar_Portfolio_Volatility_Revised_Corrected.png');

%% Visualization: Emerging Market Currency Volatility
figure;
plot(filtered_dates, emerging_ma_volatility, 'b', 'LineWidth', 1.2); hold on;
plot(filtered_dates, emerging_ewma_volatility, 'Color', [1, 0.5, 0], 'LineWidth', 1.2); % Corrected for RGB color
plot(filtered_dates, emerging_garch_volatility, 'g', 'LineWidth', 1.2);
grid on;
title('Volatility Forecasts for Emerging Market Currency (2000-2024)');
xlabel('Date');
ylabel('Volatility');
legend('Moving Average Volatility', 'EWMA Volatility', 'GARCH Volatility');
saveas(gcf, 'Emerging_Market_Volatility_Revised.png');


%% Completion Message
disp('Revised volatility forecasts calculated and visualized successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 5: Value-at-Risk (VaR) Forecasts %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Volatility_Forecasts_Revised.xlsx';
output_file = 'VaR_Forecasts.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
dollar_returns = input_data.Dollar_Returns;
emerging_returns = input_data.Emerging_Returns;
dollar_ma_volatility = input_data.MA_Volatility_Dollar;
emerging_ma_volatility = input_data.MA_Volatility_EM;
dollar_ewma_volatility = input_data.EWMA_Volatility_Dollar;
emerging_ewma_volatility = input_data.EWMA_Volatility_EM;
dollar_garch_volatility = input_data.GARCH_Volatility_Dollar;
emerging_garch_volatility = input_data.GARCH_Volatility_EM;

%% Parameters
confidence_level = 0.01; % 1% VaR
z_score = norminv(confidence_level); % Z-score for parametric VaR

%% Calculate Parametric VaR
% Dollar Portfolio
parametric_var_ma_dollar = -z_score * dollar_ma_volatility;
parametric_var_ewma_dollar = -z_score * dollar_ewma_volatility;
parametric_var_garch_dollar = -z_score * dollar_garch_volatility;

% Emerging Market
parametric_var_ma_em = -z_score * emerging_ma_volatility;
parametric_var_ewma_em = -z_score * emerging_ewma_volatility;
parametric_var_garch_em = -z_score * emerging_garch_volatility;

%% Calculate Historical Simulation VaR
window_size = 250; % 1-year rolling window
historical_var_dollar = NaN(size(dollar_returns));
historical_var_em = NaN(size(emerging_returns));

for t = window_size+1:length(dollar_returns)
    % Dollar Portfolio
    historical_var_dollar(t) = -quantile(dollar_returns(t-window_size:t-1), confidence_level);
    % Emerging Market
    historical_var_em(t) = -quantile(emerging_returns(t-window_size:t-1), confidence_level);
end

%% Save Results to Excel
results_table = table(dates, parametric_var_ma_dollar, parametric_var_ewma_dollar, ...
    parametric_var_garch_dollar, historical_var_dollar, ...
    parametric_var_ma_em, parametric_var_ewma_em, ...
    parametric_var_garch_em, historical_var_em, ...
    'VariableNames', {'Date', 'Parametric_VaR_MA_Dollar', 'Parametric_VaR_EWMA_Dollar', ...
    'Parametric_VaR_GARCH_Dollar', 'Historical_VaR_Dollar', ...
    'Parametric_VaR_MA_EM', 'Parametric_VaR_EWMA_EM', ...
    'Parametric_VaR_GARCH_EM', 'Historical_VaR_EM'});

writetable(results_table, output_file);

%% Visualization: Dollar Portfolio VaR
figure;
plot(dates, abs(parametric_var_ma_dollar), 'b', 'LineWidth', 1.2); hold on;
plot(dates, abs(parametric_var_ewma_dollar), 'Color', [1, 0.5, 0], 'LineWidth', 1.2);
plot(dates, abs(parametric_var_garch_dollar), 'g', 'LineWidth', 1.2);
plot(dates, abs(historical_var_dollar), 'r--', 'LineWidth', 1.2);
grid on;

% Customize x-axis ticks for more dates
xtickformat('yyyy'); % Set tick format to show year only
xticks(filtered_dates(1):calyears(1):filtered_dates(end)); % Add ticks every year

title('VaR (1%) for Dollar Portfolio');
xlabel('Date');
ylabel('Value-at-Risk (1%)');
legend('MA Volatility VaR', 'EWMA Volatility VaR', 'GARCH Volatility VaR', 'Historical Simulation VaR');
saveas(gcf, 'VaR_Dollar_Portfolio_Enhanced.png');


%% Visualization: Emerging Market VaR
figure;
plot(dates, abs(parametric_var_ma_em), 'b', 'LineWidth', 1.2); hold on;
plot(dates, abs(parametric_var_ewma_em), 'Color', [1, 0.5, 0], 'LineWidth', 1.2);
plot(dates, abs(parametric_var_garch_em), 'g', 'LineWidth', 1.2);
plot(dates, abs(historical_var_em), 'r--', 'LineWidth', 1.2);
grid on;
title('VaR (1%) for Emerging Market Currency');
xlabel('Date');
ylabel('Value-at-Risk (1%)');
legend('MA Volatility VaR', 'EWMA Volatility VaR', 'GARCH Volatility VaR', 'Historical Simulation VaR');
saveas(gcf, 'VaR_Emerging_Market.png');

%% Completion Message
disp('VaR forecasts calculated, visualized, and saved successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Combine Volatility and VaR Data into a Single File         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File names
volatility_file = 'Volatility_Forecasts_Revised.xlsx';
var_file = 'VaR_Forecasts.xlsx';
output_file = 'Combined_Volatility_and_VaR.xlsx';

%% Load Data
% Load volatility data
volatility_data = readtable(volatility_file);

% Load VaR data
var_data = readtable(var_file);

%% Ensure Data Alignment
if ~isequal(volatility_data.Date, var_data.Date)
    error('Dates in the two files do not match. Ensure both files have the same date range.');
end

%% Combine Data
% Combine columns from both tables
combined_data = [volatility_data, var_data(:, 2:end)]; % Skip repeating the Date column from var_data

%% Save Combined Data to Excel
writetable(combined_data, output_file);

%% Completion Message
disp('Volatility and VaR data combined successfully! Saved to Combined_Volatility_and_VaR.xlsx.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 6: Backtesting Risk Forecasts              %
% Robust Implementation for Dollar and Emerging Market Portfolios       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Combined_Volatility_and_VaR.xlsx';
output_file = 'Backtesting_Results.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
parametric_var_ma_dollar = input_data.Parametric_VaR_MA_Dollar;
parametric_var_ewma_dollar = input_data.Parametric_VaR_EWMA_Dollar;
parametric_var_garch_dollar = input_data.Parametric_VaR_GARCH_Dollar;
historical_var_dollar = input_data.Historical_VaR_Dollar;
parametric_var_ma_em = input_data.Parametric_VaR_MA_EM;
parametric_var_ewma_em = input_data.Parametric_VaR_EWMA_EM;
parametric_var_garch_em = input_data.Parametric_VaR_GARCH_EM;
historical_var_em = input_data.Historical_VaR_EM;
dollar_returns = input_data.Dollar_Returns;
emerging_returns = input_data.Emerging_Returns;

%% Backtesting: Bernoulli Coverage and Independence Tests
% Initialize results storage
var_models = {'MA', 'EWMA', 'GARCH', 'Historical'};
dollar_results = [];
emerging_results = [];

% Define functions for Bernoulli and independence tests
bernoulli_test = @(breaches, alpha, T) 1 - chi2cdf((sum(breaches) - T*alpha)^2 / (T*alpha*(1-alpha)), 1);
independence_test = @(breaches) sum(diff(find(breaches)) == 1);

% NEW FUNCTION: Independence p-value calculation
independence_p_value = @(stat) 1 - chi2cdf(stat, 1); % Chi-squared test with 1 degree of freedom

% Perform backtesting for each model (Dollar Portfolio)
var_forecasts_dollar = {parametric_var_ma_dollar, parametric_var_ewma_dollar, parametric_var_garch_dollar, historical_var_dollar};
for i = 1:length(var_models)
    breaches = dollar_returns < -abs(var_forecasts_dollar{i});
    p_value_bernoulli = bernoulli_test(breaches, 0.01, length(breaches));
    independence_stat = independence_test(breaches);
    p_value_independence = independence_p_value(independence_stat); % NEW: Calculate p-value for independence test
    dollar_results = [dollar_results; {var_models{i}, 'Dollar Portfolio', sum(breaches), p_value_bernoulli, independence_stat, p_value_independence}];
end

% Perform backtesting for each model (Emerging Market Portfolio)
var_forecasts_em = {parametric_var_ma_em, parametric_var_ewma_em, parametric_var_garch_em, historical_var_em};
for i = 1:length(var_models)
    breaches = emerging_returns < -abs(var_forecasts_em{i});
    p_value_bernoulli = bernoulli_test(breaches, 0.01, length(breaches));
    independence_stat = independence_test(breaches);
    p_value_independence = independence_p_value(independence_stat); % NEW: Calculate p-value for independence test
    emerging_results = [emerging_results; {var_models{i}, 'Emerging Market', sum(breaches), p_value_bernoulli, independence_stat, p_value_independence}];
end

%% Save Results to Excel
results_table = cell2table([dollar_results; emerging_results], ...
    'VariableNames', {'VaR_Model', 'Portfolio', 'Number_of_Breaches', 'P_Value_Bernoulli', 'Independence_Stat', 'P_Value_Independence'});

writetable(results_table, output_file);

%% Print Results to Command Window
fprintf('Backtesting Results:\n');

% Print Dollar Portfolio Results
fprintf('\nDollar Portfolio Results:\n');
fprintf('%-10s %-20s %-20s %-20s %-20s %-20s\n', 'VaR Model', 'Portfolio', 'No. of Breaches', 'P-Value (Bernoulli)', 'Independence Stat', 'P-Value (Independence)');
for i = 1:size(dollar_results, 1)
    fprintf('%-10s %-20s %-20d %-20.10f %-20d %-20.10f\n', dollar_results{i, 1}, dollar_results{i, 2}, dollar_results{i, 3}, dollar_results{i, 4}, dollar_results{i, 5}, dollar_results{i, 6});
end

% Print Emerging Market Results
fprintf('\nEmerging Market Results:\n');
fprintf('%-10s %-20s %-20s %-20s %-20s %-20s\n', 'VaR Model', 'Portfolio', 'No. of Breaches', 'P-Value (Bernoulli)', 'Independence Stat', 'P-Value (Independence)');
for i = 1:size(emerging_results, 1)
    fprintf('%-10s %-20s %-20d %-20.10f %-20d %-20.10f\n', emerging_results{i, 1}, emerging_results{i, 2}, emerging_results{i, 3}, emerging_results{i, 4}, emerging_results{i, 5}, emerging_results{i, 6});
end

%% Completion Message
disp('Combined visualizations for backtesting results have been saved successfully!');



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 7: Kupiec's Proportion of Failures Test     %
% Robust Implementation for Dollar and Emerging Market Portfolios       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Combined_Volatility_and_VaR.xlsx';
output_file = 'Kupiec_POF_Results.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
dollar_returns = input_data.Dollar_Returns;
emerging_returns = input_data.Emerging_Returns;
parametric_var_ma_dollar = input_data.Parametric_VaR_MA_Dollar;
parametric_var_ewma_dollar = input_data.Parametric_VaR_EWMA_Dollar;
parametric_var_garch_dollar = input_data.Parametric_VaR_GARCH_Dollar;
historical_var_dollar = input_data.Historical_VaR_Dollar;
parametric_var_ma_em = input_data.Parametric_VaR_MA_EM;
parametric_var_ewma_em = input_data.Parametric_VaR_EWMA_EM;
parametric_var_garch_em = input_data.Parametric_VaR_GARCH_EM;
historical_var_em = input_data.Historical_VaR_EM;

%% Kupiec's Proportion of Failures Test
% Parameters
confidence_level = 0.01; % 1% VaR
alpha = confidence_level;

% Initialize results storage
var_models = {'MA', 'EWMA', 'GARCH', 'Historical'};
dollar_results = [];
emerging_results = [];

% Define Kupiec's POF test function
kupiec_pof_test = @(n, T, alpha) -2 * log(((1 - alpha)^(T - n)) * (alpha^n)) + 2 * log(((1 - n/T)^(T - n)) * ((n/T)^n));

% Perform Kupiec's POF test for each model (Dollar Portfolio)
var_forecasts_dollar = {parametric_var_ma_dollar, parametric_var_ewma_dollar, parametric_var_garch_dollar, historical_var_dollar};
for i = 1:length(var_models)
    breaches = dollar_returns < -abs(var_forecasts_dollar{i});
    n = sum(breaches); % Number of breaches
    T = length(breaches); % Total observations
    kupiec_stat = kupiec_pof_test(n, T, alpha);
    p_value = 1 - chi2cdf(kupiec_stat, 1); % p-value with 1 degree of freedom
    dollar_results = [dollar_results; {var_models{i}, 'Dollar Portfolio', n, T, kupiec_stat, p_value}];
end

% Perform Kupiec's POF test for each model (Emerging Market Portfolio)
var_forecasts_em = {parametric_var_ma_em, parametric_var_ewma_em, parametric_var_garch_em, historical_var_em};
for i = 1:length(var_models)
    breaches = emerging_returns < -abs(var_forecasts_em{i});
    n = sum(breaches); % Number of breaches
    T = length(breaches); % Total observations
    kupiec_stat = kupiec_pof_test(n, T, alpha);
    p_value = 1 - chi2cdf(kupiec_stat, 1); % p-value with 1 degree of freedom
    emerging_results = [emerging_results; {var_models{i}, 'Emerging Market', n, T, kupiec_stat, p_value}];
end

%% Save Results to Excel
results_table = cell2table([dollar_results; emerging_results], ...
    'VariableNames', {'VaR_Model', 'Portfolio', 'Number_of_Breaches', 'Total_Observations', 'Kupiec_Statistic', 'P_Value'});

writetable(results_table, output_file);

%% Visualization: POF Test Results
% Combine results for breaches
breaches_data = [cell2mat(dollar_results(:, 3)); cell2mat(emerging_results(:, 3))];
expected_breaches = alpha * cell2mat(dollar_results(:, 4)); % Expected breaches (same for all models)

% Generate unique labels for each combination of VaR_Model and Portfolio
labels = [strcat(dollar_results(:, 1), ' (Dollar Portfolio)'); ...
          strcat(emerging_results(:, 1), ' (Emerging Market)')];

% Create bar plot
figure;
bar(categorical(labels), breaches_data, 'FaceColor', 'b');
hold on;
% Add a horizontal line for expected breaches
yline(expected_breaches(1), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Expected Breaches');
hold off;

% Add grid, title, and labels
grid on;
title('Kupiec POF Test: Number of Breaches');
xlabel('VaR Models and Portfolios');
ylabel('Number of Breaches');
legend('Actual Breaches', 'Expected Breaches', 'Location', 'Best');
saveas(gcf, 'Kupiec_POF_Breaches.png');

%% Completion Message
disp('Kupiec POF Test Results:');
disp('-------------------------------------------------------------');
disp('VaR_Model      Portfolio         Breaches    Total Obs    Statistic      P-Value');
disp('-------------------------------------------------------------');
for i = 1:size(dollar_results, 1)
    fprintf('%-12s  %-18s  %-10d  %-10d  %-12.4f  %-12.4e\n', ...
        dollar_results{i, 1}, dollar_results{i, 2}, dollar_results{i, 3}, ...
        dollar_results{i, 4}, dollar_results{i, 5}, dollar_results{i, 6});
end
for i = 1:size(emerging_results, 1)
    fprintf('%-12s  %-18s  %-10d  %-10d  %-12.4f  %-12.4e\n', ...
        emerging_results{i, 1}, emerging_results{i, 2}, emerging_results{i, 3}, ...
        emerging_results{i, 4}, emerging_results{i, 5}, emerging_results{i, 6});
end
disp('-------------------------------------------------------------');
disp('Kupiec POF test completed, results saved to Excel, and visualizations generated successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 8: Christoffersen's Test                    %
% Robust Implementation for Dollar and Emerging Market Portfolios        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File and sheet details
input_file = 'Combined_Volatility_and_VaR.xlsx';
output_file = 'Christoffersen_Test_Results.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
dollar_returns = input_data.Dollar_Returns;
emerging_returns = input_data.Emerging_Returns;
parametric_var_ma_dollar = input_data.Parametric_VaR_MA_Dollar;
parametric_var_ewma_dollar = input_data.Parametric_VaR_EWMA_Dollar;
parametric_var_garch_dollar = input_data.Parametric_VaR_GARCH_Dollar;
historical_var_dollar = input_data.Historical_VaR_Dollar;
parametric_var_ma_em = input_data.Parametric_VaR_MA_EM;
parametric_var_ewma_em = input_data.Parametric_VaR_EWMA_EM;
parametric_var_garch_em = input_data.Parametric_VaR_GARCH_EM;
historical_var_em = input_data.Historical_VaR_EM;

%% Christoffersen's Independence and Conditional Coverage Tests
% Parameters
confidence_level = 0.01; % 1% VaR

% Initialize results storage
var_models = {'MA', 'EWMA', 'GARCH', 'Historical'};
dollar_results = [];
emerging_results = [];

% Define test functions
independence_test = @(transitions, n00, n01, n10, n11) -2 * log(((n00/(n00 + n01))^n00 * (n01/(n00 + n01))^n01 * ...
    (n10/(n10 + n11))^n10 * (n11/(n10 + n11))^n11) / ((transitions(1)/(sum(transitions)))^transitions(1) * ...
    (transitions(2)/(sum(transitions)))^transitions(2)));
conditional_coverage_test = @(lr_ind, lr_uc) lr_uc + lr_ind;

% Perform tests for each model (Dollar Portfolio)
var_forecasts_dollar = {parametric_var_ma_dollar, parametric_var_ewma_dollar, parametric_var_garch_dollar, historical_var_dollar};
for i = 1:length(var_models)
    breaches = dollar_returns < -abs(var_forecasts_dollar{i});
    n00 = 0; n01 = 0; n10 = 0; n11 = 0;
    for t = 2:length(breaches)
        if breaches(t-1) == 0 && breaches(t) == 0
            n00 = n00 + 1;
        elseif breaches(t-1) == 0 && breaches(t) == 1
            n01 = n01 + 1;
        elseif breaches(t-1) == 1 && breaches(t) == 0
            n10 = n10 + 1;
        elseif breaches(t-1) == 1 && breaches(t) == 1
            n11 = n11 + 1;
        end
    end
    lr_ind = independence_test([n00 + n01, n10 + n11], n00, n01, n10, n11);
    lr_uc = -2 * log(((1 - confidence_level)^(length(breaches) - sum(breaches))) * (confidence_level^sum(breaches)) / ...
        (((length(breaches) - sum(breaches))/length(breaches))^(length(breaches) - sum(breaches)) * ...
        (sum(breaches)/length(breaches))^sum(breaches)));
    lr_cc = conditional_coverage_test(lr_ind, lr_uc);
    p_value_ind = 1 - chi2cdf(lr_ind, 1);
    p_value_cc = 1 - chi2cdf(lr_cc, 2);
    dollar_results = [dollar_results; {var_models{i}, 'Dollar Portfolio', lr_ind, p_value_ind, lr_cc, p_value_cc}];
end

% Perform tests for each model (Emerging Market Portfolio)
var_forecasts_em = {parametric_var_ma_em, parametric_var_ewma_em, parametric_var_garch_em, historical_var_em};
for i = 1:length(var_models)
    breaches = emerging_returns < -abs(var_forecasts_em{i});
    n00 = 0; n01 = 0; n10 = 0; n11 = 0;
    for t = 2:length(breaches)
        if breaches(t-1) == 0 && breaches(t) == 0
            n00 = n00 + 1;
        elseif breaches(t-1) == 0 && breaches(t) == 1
            n01 = n01 + 1;
        elseif breaches(t-1) == 1 && breaches(t) == 0
            n10 = n10 + 1;
        elseif breaches(t-1) == 1 && breaches(t) == 1
            n11 = n11 + 1;
        end
    end
    lr_ind = independence_test([n00 + n01, n10 + n11], n00, n01, n10, n11);
    lr_uc = -2 * log(((1 - confidence_level)^(length(breaches) - sum(breaches))) * (confidence_level^sum(breaches)) / ...
        (((length(breaches) - sum(breaches))/length(breaches))^(length(breaches) - sum(breaches)) * ...
        (sum(breaches)/length(breaches))^sum(breaches)));
    lr_cc = conditional_coverage_test(lr_ind, lr_uc);
    p_value_ind = 1 - chi2cdf(lr_ind, 1);
    p_value_cc = 1 - chi2cdf(lr_cc, 2);
    emerging_results = [emerging_results; {var_models{i}, 'Emerging Market', lr_ind, p_value_ind, lr_cc, p_value_cc}];
end

%% Save Results to Excel
results_table = cell2table([dollar_results; emerging_results], ...
    'VariableNames', {'VaR_Model', 'Portfolio', 'LR_Ind', 'P_Value_Independence', 'LR_CC', 'P_Value_Conditional_Coverage'});

writetable(results_table, output_file);

%% Visualization: Independence and Conditional Coverage Tests
% Bar plot for LR_CC comparison
figure;

% Combine Dollar and Emerging results with unique labels
cc_data = [cell2mat(dollar_results(:, 5)); cell2mat(emerging_results(:, 5))];
categories = [strcat(dollar_results(:, 1), ' (Dollar)'); strcat(emerging_results(:, 1), ' (Emerging)')];

% Create the bar chart
bar(categorical(categories), cc_data, 'FaceColor', 'b');
grid on;

% Add title, labels, and save the figure
title('Christoffersen Conditional Coverage Test Statistics');
xlabel('VaR Models and Portfolios');
ylabel('LR_CC');
saveas(gcf, 'Christoffersen_LR_CC.png');
%% Completion Message
disp('Christoffersen tests completed, results saved to Excel, and visualizations generated successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Part 9: Rolling Window Experiments         %
% Comparing 252-Day and 504-Day Rolling Windows for Volatility/VAR %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% File details
input_file = 'Combined_Volatility_and_VaR.xlsx';
output_file = 'Rolling_Window_Experiments.xlsx';

%% Load Data
% Read the data from the Excel file
input_data = readtable(input_file);

% Extract relevant columns
dates = input_data.Date;
dollar_returns = input_data.Dollar_Returns;
emerging_returns = input_data.Emerging_Returns;

%% Parameters
window_1 = 252; % 252-day (1 year)
window_2 = 504; % 504-day (2 years)
confidence_level = 0.01; % 1% VaR
z_score = norminv(confidence_level); % Z-score for parametric VaR

%% Volatility Calculations
% Initialize arrays for 252-day and 504-day rolling volatility
volatility_252_dollar = NaN(size(dollar_returns));
volatility_504_dollar = NaN(size(dollar_returns));
volatility_252_emerging = NaN(size(emerging_returns));
volatility_504_emerging = NaN(size(emerging_returns));

% Calculate rolling volatility
for t = max(window_1, window_2):length(dollar_returns)
    % Dollar Portfolio
    volatility_252_dollar(t) = std(dollar_returns(t-window_1+1:t), 'omitnan');
    volatility_504_dollar(t) = std(dollar_returns(t-window_2+1:t), 'omitnan');
    % Emerging Market
    volatility_252_emerging(t) = std(emerging_returns(t-window_1+1:t), 'omitnan');
    volatility_504_emerging(t) = std(emerging_returns(t-window_2+1:t), 'omitnan');
end

%% VaR Calculations
% Parametric VaR using 252-day and 504-day volatility
parametric_var_252_dollar = -z_score * volatility_252_dollar;
parametric_var_504_dollar = -z_score * volatility_504_dollar;
parametric_var_252_emerging = -z_score * volatility_252_emerging;
parametric_var_504_emerging = -z_score * volatility_504_emerging;

%% Combine Results
results_table = table(dates, dollar_returns, emerging_returns, ...
    volatility_252_dollar, volatility_504_dollar, ...
    volatility_252_emerging, volatility_504_emerging, ...
    parametric_var_252_dollar, parametric_var_504_dollar, ...
    parametric_var_252_emerging, parametric_var_504_emerging, ...
    'VariableNames', {'Date', 'Dollar_Returns', 'Emerging_Returns', ...
    'Volatility_252_Dollar', 'Volatility_504_Dollar', ...
    'Volatility_252_Emerging', 'Volatility_504_Emerging', ...
    'Parametric_VaR_252_Dollar', 'Parametric_VaR_504_Dollar', ...
    'Parametric_VaR_252_Emerging', 'Parametric_VaR_504_Emerging'});

%% Save Results to Excel
writetable(results_table, output_file);

%% Visualization: Volatility Comparison (Dollar Portfolio)
figure;
hold on;
plot(dates, volatility_252_dollar, 'b', 'LineWidth', 1.2, 'DisplayName', '252-Day Volatility');
plot(dates, volatility_504_dollar, 'r', 'LineWidth', 1.2, 'DisplayName', '504-Day Volatility');
hold off;
grid on;
title('Dollar Portfolio Volatility Comparison');
xlabel('Date');
ylabel('Volatility');
legend('show');
saveas(gcf, 'Volatility_Comparison_Dollar.png');

%% Visualization: Volatility Comparison (Emerging Market)
figure;
hold on;
plot(dates, volatility_252_emerging, 'b', 'LineWidth', 1.2, 'DisplayName', '252-Day Volatility');
plot(dates, volatility_504_emerging, 'r', 'LineWidth', 1.2, 'DisplayName', '504-Day Volatility');
hold off;
grid on;
title('Emerging Market Volatility Comparison');
xlabel('Date');
ylabel('Volatility');
legend('show');
saveas(gcf, 'Volatility_Comparison_Emerging.png');

%% Visualization: VaR Comparison (Dollar Portfolio)
figure;
hold on;
plot(dates, abs(parametric_var_252_dollar), 'b', 'LineWidth', 1.2, 'DisplayName', '252-Day VaR');
plot(dates, abs(parametric_var_504_dollar), 'r', 'LineWidth', 1.2, 'DisplayName', '504-Day VaR');
hold off;
grid on;
title('Dollar Portfolio VaR Comparison');
xlabel('Date');
ylabel('Value-at-Risk (1%)');
legend('show');
saveas(gcf, 'VaR_Comparison_Dollar.png');

%% Visualization: VaR Comparison (Emerging Market)
figure;
hold on;
plot(dates, abs(parametric_var_252_emerging), 'b', 'LineWidth', 1.2, 'DisplayName', '252-Day VaR');
plot(dates, abs(parametric_var_504_emerging), 'r', 'LineWidth', 1.2, 'DisplayName', '504-Day VaR');
hold off;
grid on;
title('Emerging Market VaR Comparison');
xlabel('Date');
ylabel('Value-at-Risk (1%)');
legend('show');
saveas(gcf, 'VaR_Comparison_Emerging.png');

%% Completion Message
disp('Rolling window experiments completed successfully!');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Research Methodology - Combine All Excel Files into One Workbook      %
% Each dataset is stored in a separate sheet named after the file      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization

% Define file names (input files)
file_names = { 
    'Currency_Returns_Complete.xlsx',
    'Dollar_Portfolio_Results.xlsx',
    'Volatility_Forecasts_Revised.xlsx',
    'VaR_Forecasts.xlsx',
    'Combined_Volatility_and_VaR.xlsx',
    'Backtesting_Results.xlsx',
    'Kupiec_POF_Results.xlsx',
    'Christoffersen_Test_Results.xlsx',
    'Rolling_Window_Experiments.xlsx'
};

% Define the output file
output_file = 'A_Final_Combined_Results.xlsx';

%% Read Each File and Write to Separate Sheets in Output File

for i = 1:length(file_names)
    % Read the data from the current Excel file
    data = readtable(file_names{i});
    
    % Extract the base name of the file (without extension) to use as sheet name
    [~, sheet_name, ~] = fileparts(file_names{i});
    
    % Write data to a new sheet in the output file
    writetable(data, output_file, 'Sheet', sheet_name);
    
    % Display progress message
    fprintf('File "%s" successfully added as sheet "%s"\n', file_names{i}, sheet_name);
end

%% Completion Message
disp('All files have been successfully combined into a single Excel file with separate sheets!');
