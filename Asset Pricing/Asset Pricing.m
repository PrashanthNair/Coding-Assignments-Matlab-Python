% Comprehensive MATLAB Code for Asset Pricing Analysis (Questions 2.1, 2.2, and 2.3)
% Written to Evaluate Portfolio Alphas, Betas, and Conduct Cross-Sectional Regression
% Steps Include: Time-Series Regression, GLS vs. OLS Comparison, MOM Factor Construction, and Multi-Factor Analysis

%% Section 1: Data Import and Preparation (Question 2.1, 2.2, and 2.3)
% Load necessary Excel data for Portfolio Returns and Corporate Bond Risk Factors
% Prepare data for time-series analysis and cross-sectional analysis

clear all; clc;

% Define file path for Excel data
filePath = 'C:\Users\nairp\OneDrive\Desktop\Asset pricing Group Project\Portfolio Data.xlsx';

% Load portfolio returns and risk factors
portfolio_returns = readtable(filePath, 'Sheet', 'Portfolio Returns');
risk_factors = readtable(filePath, 'Sheet', 'Corporate Bond Risk Factors');

% Extract MKTB and MKTDB from risk_factors
mktb = risk_factors.MKTB;
mktdb = risk_factors.MKTDB;

% Extract portfolio data (excluding Date column)
portfolios = portfolio_returns(:, 2:end);
portfolioNames = portfolios.Properties.VariableNames;

% Convert table to matrix for processing
portfolio_data = table2array(portfolios);

% Prepare for regression results storage
numPortfolios = size(portfolio_data, 2);
results_MKTB = struct();
results_MKTDB = struct();

% Plot individual portfolio time-series data
figure;
for i = 1:numPortfolios
    subplot(7, 8, i);
    plot(portfolio_data(:, i));
    title(portfolioNames{i});
    xlabel('Time');
    ylabel('Returns');
end
sgtitle('Time-Series Plots of Individual Portfolios');

% Plot MKTB vs. MKTDB comparison
figure;
plot(mktb, 'b');
hold on;
plot(mktdb, 'r');
hold off;
legend('MKTB', 'MKTDB');
xlabel('Time');
ylabel('Factor Values');
title('Comparison of MKTB and MKTDB Over Time');

% Addition: Prepare data for heteroscedasticity testing
% Extracting date variable for tracking purposes
dates = portfolio_returns.Date;

% Adding a placeholder for heteroscedasticity test results
heteroscedasticity_results = struct();

fprintf('Section 1 completed: Data imported, plotted, and prepared for further analysis.\n');
%% Section 2: Time-Series Regressions for Single-Factor CAPM (Question 2.1)
% Perform OLS Regressions for each Portfolio against MKTB and MKTDB
% Store alpha, beta, p-values, and R-squared values

% Initialize arrays for storing OLS results
alphas_MKTB = zeros(numPortfolios, 1);
betas_MKTB = zeros(numPortfolios, 1);
alpha_pvals_MKTB = zeros(numPortfolios, 1);
beta_pvals_MKTB = zeros(numPortfolios, 1);
r_squared_MKTB = zeros(numPortfolios, 1);

alphas_MKTDB = zeros(numPortfolios, 1);
betas_MKTDB = zeros(numPortfolios, 1);
alpha_pvals_MKTDB = zeros(numPortfolios, 1);
beta_pvals_MKTDB = zeros(numPortfolios, 1);
r_squared_MKTDB = zeros(numPortfolios, 1);

% Initialize arrays to store heteroscedasticity test results
heteroscedasticity_MKTB = zeros(numPortfolios, 1);
heteroscedasticity_MKTDB = zeros(numPortfolios, 1);

% Initialize arrays to store portfolio betas and average returns
portfolio_betas = zeros(numPortfolios, 1);
average_returns = zeros(numPortfolios, 1);

for i = 1:numPortfolios
    % MKTB Regression
    mdl_MKTB = fitlm(mktb, portfolio_data(:, i));
    alphas_MKTB(i) = mdl_MKTB.Coefficients.Estimate(1);
    betas_MKTB(i) = mdl_MKTB.Coefficients.Estimate(2);
    alpha_pvals_MKTB(i) = mdl_MKTB.Coefficients.pValue(1);
    beta_pvals_MKTB(i) = mdl_MKTB.Coefficients.pValue(2);
    r_squared_MKTB(i) = mdl_MKTB.Rsquared.Ordinary;
    
    % Heteroscedasticity Test for MKTB
    residuals_MKTB = mdl_MKTB.Residuals.Raw;
    [~, pval] = archtest(residuals_MKTB); % ARCH test for heteroscedasticity
    heteroscedasticity_MKTB(i) = (pval < 0.05); % Store 1 if heteroscedasticity is detected

    % MKTDB Regression
    mdl_MKTDB = fitlm(mktdb, portfolio_data(:, i));
    alphas_MKTDB(i) = mdl_MKTDB.Coefficients.Estimate(1);
    betas_MKTDB(i) = mdl_MKTDB.Coefficients.Estimate(2);
    alpha_pvals_MKTDB(i) = mdl_MKTDB.Coefficients.pValue(1);
    beta_pvals_MKTDB(i) = mdl_MKTDB.Coefficients.pValue(2);
    r_squared_MKTDB(i) = mdl_MKTDB.Rsquared.Ordinary;

    % Heteroscedasticity Test for MKTDB
    residuals_MKTDB = mdl_MKTDB.Residuals.Raw;
    [~, pval] = archtest(residuals_MKTDB); % ARCH test for heteroscedasticity
    heteroscedasticity_MKTDB(i) = (pval < 0.05); % Store 1 if heteroscedasticity is detected
end

% Store significant alphas (p < 0.05)
significant_alphas_MKTB = find(alpha_pvals_MKTB < 0.05);
significant_alphas_MKTDB = find(alpha_pvals_MKTDB < 0.05);


fprintf('Significant alphas - MKTB: %d portfolios\n', length(significant_alphas_MKTB));
fprintf('Significant alphas - MKTDB: %d portfolios\n', length(significant_alphas_MKTDB));

% Display heteroscedasticity detection results
fprintf('Portfolios with significant heteroskedasticity - MKTB: %d portfolios\n', sum(heteroscedasticity_MKTB));
fprintf('Portfolios with significant heteroskedasticity - MKTDB: %d portfolios\n', sum(heteroscedasticity_MKTDB));

% Calculate Market Risk Premiums (gamma1 values)
market_risk_premium_MKTB = mean(mktb); % Average market excess return for MKTB
market_risk_premium_MKTDB = mean(mktdb); % Average market excess return for MKTDB

% Calculate SML Points
gamma0_MKTB = mean(alphas_MKTB); % Intercept (average alpha)
gamma0_MKTDB = mean(alphas_MKTDB); % Intercept (average alpha)

% Security Market Line Equations
SML_MKTB = gamma0_MKTB + market_risk_premium_MKTB * betas_MKTB;
SML_MKTDB = gamma0_MKTDB + market_risk_premium_MKTDB * betas_MKTDB;

% Calculate Portfolio Average Returns (Observed)
average_returns = mean(portfolio_data, 1);

% Plot SML for MKTB
figure;
scatter(betas_MKTB, average_returns, 'filled'); % Portfolio returns vs. betas
hold on;
plot(betas_MKTB, SML_MKTB, 'r', 'LineWidth', 2); % SML line
title('Security Market Line (SML) - MKTB');
xlabel('Beta (\beta)');
ylabel('Expected Return (E(R_p))');
legend('Portfolios', 'SML', 'Location', 'Northwest');
grid on;

% Plot SML for MKTDB
figure;
scatter(betas_MKTDB, average_returns, 'filled'); % Portfolio returns vs. betas
hold on;
plot(betas_MKTDB, SML_MKTDB, 'r', 'LineWidth', 2); % SML line
title('Security Market Line (SML) - MKTDB');
xlabel('Beta (\beta)');
ylabel('Expected Return (E(R_p))');
legend('Portfolios', 'SML', 'Location', 'Northwest');
grid on;

%% Section 3: GLS Regression for Robustness Comparison (Question 2.1 and 2.2)
% Perform Generalized Least Squares (GLS) Regressions for MKTB and MKTDB
% Account for potential heteroscedasticity in residuals

% Calculate residual variances for each portfolio from OLS regressions
residual_variances_MKTB = zeros(numPortfolios, 1);
residual_variances_MKTDB = zeros(numPortfolios, 1);
heteroskedasticity_MKTB = zeros(numPortfolios, 1);
heteroskedasticity_MKTDB = zeros(numPortfolios, 1);

for i = 1:numPortfolios
    % Residuals from MKTB regression
    residuals_MKTB = portfolio_data(:, i) - (mktb * betas_MKTB(i) + alphas_MKTB(i));
    residual_variances_MKTB(i) = mean(residuals_MKTB .^ 2); % Calculate variance for each portfolio
    
    % Breusch-Pagan test for heteroskedasticity - MKTB
    X = [ones(length(mktb), 1), mktb];
    [~, ~, ~, ~, stats] = regress(residuals_MKTB .^ 2, X);
    pValue_BP_MKTB = stats(3); % p-value from the Breusch-Pagan test
    heteroskedasticity_MKTB(i) = pValue_BP_MKTB < 0.05;
    
    % Residuals from MKTDB regression
    residuals_MKTDB = portfolio_data(:, i) - (mktdb * betas_MKTDB(i) + alphas_MKTDB(i));
    residual_variances_MKTDB(i) = mean(residuals_MKTDB .^ 2); % Calculate variance for each portfolio
    
    % Breusch-Pagan test for heteroskedasticity - MKTDB
    X = [ones(length(mktdb), 1), mktdb];
    [~, ~, ~, ~, stats] = regress(residuals_MKTDB .^ 2, X);
    pValue_BP_MKTDB = stats(3); % p-value from the Breusch-Pagan test
    heteroskedasticity_MKTDB(i) = pValue_BP_MKTDB < 0.05;
end

% Count portfolios with significant heteroskedasticity
num_heteroskedastic_MKTB = sum(heteroskedasticity_MKTB);
num_heteroskedastic_MKTDB = sum(heteroskedasticity_MKTDB);


% Perform GLS for each portfolio individually
alphas_gls_MKTB = zeros(numPortfolios, 1);
betas_gls_MKTB = zeros(numPortfolios, 1);
alpha_pvals_gls_MKTB = zeros(numPortfolios, 1);
beta_pvals_gls_MKTB = zeros(numPortfolios, 1);

alphas_gls_MKTDB = zeros(numPortfolios, 1);
betas_gls_MKTDB = zeros(numPortfolios, 1);
alpha_pvals_gls_MKTDB = zeros(numPortfolios, 1);
beta_pvals_gls_MKTDB = zeros(numPortfolios, 1);

for i = 1:numPortfolios
    % Weights for GLS (inverse of residual variances)
    weights_MKTB = 1 ./ residual_variances_MKTB(i) * ones(size(mktb));
    weights_MKTDB = 1 ./ residual_variances_MKTDB(i) * ones(size(mktdb));
    
    % GLS for MKTB
    mdl_gls_MKTB = fitglm(mktb, portfolio_data(:, i), 'Weights', weights_MKTB);
    alphas_gls_MKTB(i) = mdl_gls_MKTB.Coefficients.Estimate(1);
    betas_gls_MKTB(i) = mdl_gls_MKTB.Coefficients.Estimate(2);
    alpha_pvals_gls_MKTB(i) = mdl_gls_MKTB.Coefficients.pValue(1);
    beta_pvals_gls_MKTB(i) = mdl_gls_MKTB.Coefficients.pValue(2);
    
    % GLS for MKTDB
    mdl_gls_MKTDB = fitglm(mktdb, portfolio_data(:, i), 'Weights', weights_MKTDB);
    alphas_gls_MKTDB(i) = mdl_gls_MKTDB.Coefficients.Estimate(1);
    betas_gls_MKTDB(i) = mdl_gls_MKTDB.Coefficients.Estimate(2);
    alpha_pvals_gls_MKTDB(i) = mdl_gls_MKTDB.Coefficients.pValue(1);
    beta_pvals_gls_MKTDB(i) = mdl_gls_MKTDB.Coefficients.pValue(2);
end

% Store significant alphas (p < 0.05) for GLS regression of MKTB and MKTDB
gls_significant_alphas_MKTB = find(alpha_pvals_gls_MKTB < 0.05);
gls_significant_alphas_MKTDB = find(alpha_pvals_gls_MKTDB < 0.05);

% Print out the number of significant alphas for both MKTB and MKTDB using GLS
fprintf('GLS Estimation for MKTB and MKTDB completed.\n');
fprintf('Portfolios with significant heteroskedasticity - MKTB: %d portfolios\n', num_heteroskedastic_MKTB);
fprintf('Portfolios with significant heteroskedasticity - MKTDB: %d portfolios\n', num_heteroskedastic_MKTDB);

% Functions for Summary and Analysis

% Detailed portfolio breakdown for significant alphas
fprintf('Portfolios with significant alphas for MKTB (GLS):\n');
for idx = gls_significant_alphas_MKTB'
    fprintf('Portfolio %d: Alpha = %.4f, p-value = %.4f\n', idx, alphas_gls_MKTB(idx), alpha_pvals_gls_MKTB(idx));
end

fprintf('Portfolios with significant alphas for MKTDB (GLS):\n');
for idx = gls_significant_alphas_MKTDB'
    fprintf('Portfolio %d: Alpha = %.4f, p-value = %.4f\n', idx, alphas_gls_MKTDB(idx), alpha_pvals_gls_MKTDB(idx));
end

% Mean and standard deviation of estimated alphas and betas
mean_alpha_MKTB = mean(alphas_gls_MKTB);
std_alpha_MKTB = std(alphas_gls_MKTB);
mean_beta_MKTB = mean(betas_gls_MKTB);
std_beta_MKTB = std(betas_gls_MKTB);

mean_alpha_MKTDB = mean(alphas_gls_MKTDB);
std_alpha_MKTDB = std(alphas_gls_MKTDB);
mean_beta_MKTDB = mean(betas_gls_MKTDB);
std_beta_MKTDB = std(betas_gls_MKTDB);

fprintf('Mean and Standard Deviation of Alphas and Betas (GLS):\n');
fprintf('MKTB - Mean Alpha: %.4f, Std Alpha: %.4f, Mean Beta: %.4f, Std Beta: %.4f\n', mean_alpha_MKTB, std_alpha_MKTB, mean_beta_MKTB, std_beta_MKTB);
fprintf('MKTDB - Mean Alpha: %.4f, Std Alpha: %.4f, Mean Beta: %.4f, Std Beta: %.4f\n', mean_alpha_MKTDB, std_alpha_MKTDB, mean_beta_MKTDB, std_beta_MKTDB);

% Load data from Excel file
filePath = 'C:\Users\nairp\OneDrive\Desktop\Asset pricing Group Project\Portfolio Data.xlsx';

% Load portfolio returns and risk factors
portfolio_returns = readtable(filePath, 'Sheet', 'Portfolio Returns');
risk_factors = readtable(filePath, 'Sheet', 'Corporate Bond Risk Factors');

% Extract MKTB and MKTDB from risk_factors
mktb = risk_factors.MKTB;   % Market factor (MKTB)
mktdb = risk_factors.MKTDB; % Duration-adjusted market factor (MKTDB)

% Extract portfolio return data (excluding the date column)
portfolios = portfolio_returns(:, 2:end); % Assuming first column is Date
portfolio_data = table2array(portfolios); % Convert to matrix for processing

% Number of portfolios
numPortfolios = size(portfolio_data, 2);

% Initialize arrays to store R-squared values for OLS and GLS
r_squared_ols_MKTB = zeros(numPortfolios, 1);
r_squared_ols_MKTDB = zeros(numPortfolios, 1);
r_squared_gls_MKTB = zeros(numPortfolios, 1);
r_squared_gls_MKTDB = zeros(numPortfolios, 1);

% Perform OLS regressions and calculate R-squared values
for i = 1:numPortfolios
    % OLS for MKTB
    mdl_ols_MKTB = fitlm(mktb, portfolio_data(:, i));
    r_squared_ols_MKTB(i) = mdl_ols_MKTB.Rsquared.Ordinary;
    
    % OLS for MKTDB
    mdl_ols_MKTDB = fitlm(mktdb, portfolio_data(:, i));
    r_squared_ols_MKTDB(i) = mdl_ols_MKTDB.Rsquared.Ordinary;
end

% Perform GLS regressions and calculate R-squared values
for i = 1:numPortfolios
    % Residual variances for weights in GLS
    residuals_ols_MKTB = portfolio_data(:, i) - (mktb * mdl_ols_MKTB.Coefficients.Estimate(2) + mdl_ols_MKTB.Coefficients.Estimate(1));
    weights_MKTB = 1 ./ var(residuals_ols_MKTB) * ones(size(mktb));
    
    residuals_ols_MKTDB = portfolio_data(:, i) - (mktdb * mdl_ols_MKTDB.Coefficients.Estimate(2) + mdl_ols_MKTDB.Coefficients.Estimate(1));
    weights_MKTDB = 1 ./ var(residuals_ols_MKTDB) * ones(size(mktdb));
    
    % GLS for MKTB
    mdl_gls_MKTB = fitglm(mktb, portfolio_data(:, i), 'Weights', weights_MKTB);
    r_squared_gls_MKTB(i) = mdl_gls_MKTB.Rsquared.Ordinary;
    
    % GLS for MKTDB
    mdl_gls_MKTDB = fitglm(mktdb, portfolio_data(:, i), 'Weights', weights_MKTDB);
    r_squared_gls_MKTDB(i) = mdl_gls_MKTDB.Rsquared.Ordinary;
end

% Compute average R-squared values
average_r_squared_ols_MKTB = mean(r_squared_ols_MKTB);
average_r_squared_ols_MKTDB = mean(r_squared_ols_MKTDB);
average_r_squared_gls_MKTB = mean(r_squared_gls_MKTB);
average_r_squared_gls_MKTDB = mean(r_squared_gls_MKTDB);

% Display results
fprintf('Average R-squared (OLS) - MKTB: %.4f\n', average_r_squared_ols_MKTB);
fprintf('Average R-squared (OLS) - MKTDB: %.4f\n', average_r_squared_ols_MKTDB);
fprintf('Average R-squared (GLS) - MKTB: %.4f\n', average_r_squared_gls_MKTB);
fprintf('Average R-squared (GLS) - MKTDB: %.4f\n', average_r_squared_gls_MKTDB);


% Count of betas outside expected range (>1 or <0)
betas_outside_range_MKTB = sum(betas_gls_MKTB > 1 | betas_gls_MKTB < 0);
betas_outside_range_MKTDB = sum(betas_gls_MKTDB > 1 | betas_gls_MKTDB < 0);

fprintf('Portfolios with Betas outside expected range (MKTB): %d portfolios\n', betas_outside_range_MKTB);
fprintf('Portfolios with Betas outside expected range (MKTDB): %d portfolios\n', betas_outside_range_MKTDB);


%% Section 4: Cross-Sectional Regression Analysis (Question 2.1 and 2.2)
% Cross-sectional regression of average returns on beta estimates

% Calculate average excess returns for each portfolio
average_returns = mean(portfolio_data, 1)';

% Cross-sectional regression for MKTB betas
cross_sec_model_MKTB = fitlm(betas_MKTB, average_returns);
fprintf('Cross-Sectional Regression Summary - MKTB:\n');
disp(cross_sec_model_MKTB)

% Cross-sectional regression for MKTDB betas
cross_sec_model_MKTDB = fitlm(betas_MKTDB, average_returns);
fprintf('Cross-Sectional Regression Summary - MKTDB:\n');
disp(cross_sec_model_MKTDB)

% Perform Fama-Macbeth Regression to evaluate cross-sectional pricing models

% Initialize arrays to store Fama-Macbeth estimates
fm_alpha_estimates = zeros(numPortfolios, 1);
fm_beta_estimates = zeros(numPortfolios, 2); % For MKTB and MKTDB
fm_pvalues_alpha = zeros(numPortfolios, 1);
fm_pvalues_beta = zeros(numPortfolios, 2);

% First pass: Time-series regressions to estimate betas for each portfolio
betas_first_pass = zeros(numPortfolios, 2);
for i = 1:numPortfolios
    mdl_first_pass = fitlm([mktb, mktdb], portfolio_data(:, i));
    betas_first_pass(i, :) = mdl_first_pass.Coefficients.Estimate(2:3)';
end

% Second pass: Cross-sectional regression of average returns on betas
average_returns = mean(portfolio_data, 1)';
mdl_second_pass = fitlm(betas_first_pass, average_returns);
fm_alpha_estimates = mdl_second_pass.Coefficients.Estimate(1);
fm_beta_estimates = mdl_second_pass.Coefficients.Estimate(2:end);
fm_pvalues_alpha = mdl_second_pass.Coefficients.pValue(1);
fm_pvalues_beta = mdl_second_pass.Coefficients.pValue(2:end);

% Print Fama-Macbeth regression results
fprintf('Fama-Macbeth Regression Summary:\n');
fprintf('Alpha Estimate: %.4f, p-value: %.4f\n', fm_alpha_estimates, fm_pvalues_alpha);
for j = 1:length(fm_beta_estimates)
    fprintf('Beta Estimate %d: %.4f, p-value: %.4f\n', j, fm_beta_estimates(j), fm_pvalues_beta(j));
end

% Additional diagnostics: R-squared, adjusted R-squared, F-statistics
fprintf('Cross-Sectional Regression Diagnostics:\n');
fprintf('R-squared - MKTB: %.4f\n', cross_sec_model_MKTB.Rsquared.Ordinary);
fprintf('Adjusted R-squared - MKTB: %.4f\n', cross_sec_model_MKTB.Rsquared.Adjusted);

% Extract F-statistic and p-value using ANOVA for MKTB
anova_MKTB = anova(cross_sec_model_MKTB);
FStat_MKTB = anova_MKTB.F(1);
pValue_F_MKTB = anova_MKTB.pValue(1);
fprintf('F-statistic - MKTB: %.4f, p-value: %.4e\n', FStat_MKTB, pValue_F_MKTB);

fprintf('R-squared - MKTDB: %.4f\n', cross_sec_model_MKTDB.Rsquared.Ordinary);
fprintf('Adjusted R-squared - MKTDB: %.4f\n', cross_sec_model_MKTDB.Rsquared.Adjusted);

% Extract F-statistic and p-value using ANOVA for MKTDB
anova_MKTDB = anova(cross_sec_model_MKTDB);
FStat_MKTDB = anova_MKTDB.F(1);
pValue_F_MKTDB = anova_MKTDB.pValue(1);
fprintf('F-statistic - MKTDB: %.4f, p-value: %.4e\n', FStat_MKTDB, pValue_F_MKTDB);

%% Section 5: Momentum Factor Construction (Question 2.3)
% Construct the Momentum (MOM) Factor Using Decile Portfolios
% Sort portfolios based on past 12-month returns and create long-short portfolios

% Extract MKTB, MKTDB, and portfolio data (excluding Date column)
mktb = risk_factors.MKTB;
mktdb = risk_factors.MKTDB;
portfolios = portfolio_returns(:, 2:end);
portfolioNames = portfolios.Properties.VariableNames;
portfolio_data = table2array(portfolios);

% Generate 12-month lagged returns for each portfolio
numMonths = size(portfolio_data, 1);
lagged_returns = nan(numMonths - 12, size(portfolio_data, 2));

for i = 13:numMonths
    lagged_returns(i - 12, :) = mean(portfolio_data(i-12:i-1, :), 1);
end

% Calculate the average lagged returns for each portfolio
avg_lagged_returns = nanmean(lagged_returns);

[~, ranks] = sort(avg_lagged_returns);
ranked_deciles = ceil(ranks / (size(portfolio_data, 2) / 10));

% Create MOM factor by longing the top decile and shorting the bottom decile
mom_factor = nan(numMonths, 1);  % Change length of MOM factor to be equal to numMonths
for t = 13:numMonths
    top_decile = mean(portfolio_data(t, ranked_deciles == 10));
    bottom_decile = mean(portfolio_data(t, ranked_deciles == 1));
    mom_factor(t) = top_decile - bottom_decile;
end
fprintf('Momentum factor (MOM) constructed successfully.\n');

% Print summary statistics for the MOM factor
mean_mom = mean(mom_factor, 'omitnan');
std_mom = std(mom_factor, 'omitnan');
fprintf('Momentum Factor (MOM) Summary:\n');
fprintf('Mean: %.4f\n', mean_mom);
fprintf('Standard Deviation: %.4f\n', std_mom);

% Additional analysis for momentum trends (Pre-crisis vs Post-crisis)
mean_pre_crisis = mean(mom_factor(1:round(end/2)), 'omitnan');
mean_post_crisis = mean(mom_factor(round(end/2)+1:end), 'omitnan');
fprintf('Momentum Trend Analysis:\n');
fprintf('Mean Pre-Crisis: %.4f\n', mean_pre_crisis);
fprintf('Mean Post-Crisis: %.4f\n', mean_post_crisis);


%% Section 6
% Perform regressions for each portfolio including MKTB, MKTDB, and MOM

aligned_mktb = mktb;
aligned_mktdb = mktdb;

alphas_multi = zeros(size(portfolio_data, 2), 1);
betas_MKTB_multi = zeros(size(portfolio_data, 2), 1);
betas_MKTDB_multi = zeros(size(portfolio_data, 2), 1);
betas_MOM_multi = zeros(size(portfolio_data, 2), 1);

alpha_pvals_multi = zeros(size(portfolio_data, 2), 1);
beta_MKTB_pvals_multi = zeros(size(portfolio_data, 2), 1);
beta_MKTDB_pvals_multi = zeros(size(portfolio_data, 2), 1);
beta_MOM_pvals_multi = zeros(size(portfolio_data, 2), 1);

r_squared_multi = zeros(size(portfolio_data, 2), 1);  % Array to store R-squared values for each portfolio

% Loop through each portfolio and run multi-factor regression
for i = 1:size(portfolio_data, 2)
    mdl_multi = fitlm([aligned_mktb, aligned_mktdb, mom_factor], portfolio_data(:, i));
    alphas_multi(i) = mdl_multi.Coefficients.Estimate(1);
    betas_MKTB_multi(i) = mdl_multi.Coefficients.Estimate(2);
    betas_MKTDB_multi(i) = mdl_multi.Coefficients.Estimate(3);
    betas_MOM_multi(i) = mdl_multi.Coefficients.Estimate(4);
    
    alpha_pvals_multi(i) = mdl_multi.Coefficients.pValue(1);
    beta_MKTB_pvals_multi(i) = mdl_multi.Coefficients.pValue(2);
    beta_MKTDB_pvals_multi(i) = mdl_multi.Coefficients.pValue(3);
    beta_MOM_pvals_multi(i) = mdl_multi.Coefficients.pValue(4);
    
    % Store R-squared value for each portfolio
    r_squared_multi(i) = mdl_multi.Rsquared.Ordinary;

    % Print individual regression diagnostics for each portfolio
    fprintf('Portfolio %d Regression Diagnostics:\n', i);
    fprintf('R-squared: %.4f\n', mdl_multi.Rsquared.Ordinary);
    fprintf('Adjusted R-squared: %.4f\n', mdl_multi.Rsquared.Adjusted);
    end

% Calculate and print the average R-squared value for the Multi-Factor Model
mean_r_squared_multi = mean(r_squared_multi);
fprintf('Average R-squared - Multi-Factor Model: %.4f\n', mean_r_squared_multi);

% Display number of significant alphas
significant_alphas_multi = find(alpha_pvals_multi < 0.05);
fprintf('Significant alphas - Multi-Factor Model: %d portfolios\n', length(significant_alphas_multi));

% Breusch-Pagan test for heteroscedasticity for multi-factor model residuals
residuals_multi = portfolio_data(13:end, :) - ([aligned_mktb(13:end), aligned_mktdb(13:end), mom_factor(13:end)] * [betas_MKTB_multi'; betas_MKTDB_multi'; betas_MOM_multi']);
heteroskedasticity_multi = zeros(size(portfolio_data, 2), 1);

for i = 1:size(portfolio_data, 2)
    % Breusch-Pagan test for heteroscedasticity - Multi-Factor Model
    X = [ones(length(aligned_mktb(13:end)), 1), aligned_mktb(13:end), aligned_mktdb(13:end), mom_factor(13:end)];
    [~, ~, ~, ~, stats] = regress(residuals_multi(:, i) .^ 2, X);
    pValue_BP_multi = stats(3); % p-value from the Breusch-Pagan test
    testStatistic_BP_multi = stats(1); % Test statistic from the Breusch-Pagan test
    heteroskedasticity_multi(i) = pValue_BP_multi < 0.05;
    fprintf('Breusch-Pagan test for Portfolio %d: Test Statistic = %.4f, p-value = %.4e\n', i, testStatistic_BP_multi, pValue_BP_multi);
end

% Count portfolios with significant heteroscedasticity in Multi-Factor Model
num_heteroskedastic_multi = sum(heteroskedasticity_multi);

fprintf('Portfolios with significant heteroskedasticity - Multi-Factor Model: %d portfolios\n', num_heteroskedastic_multi);

% Perform Fama-MacBeth Regression for Multi-Factor Model
fprintf('Performing Fama-MacBeth Regression for Multi-Factor Model...\n');
X_fmb = [betas_MKTB_multi, betas_MKTDB_multi, betas_MOM_multi];
T = size(lagged_returns, 1);
gamma_estimates = zeros(T, size(X_fmb, 2) + 1);

dates = datetime(2003, 8, 31) + calmonths(0:(T-1)); % Create dates for the output
for t = 1:T
    mdl_fmb = fitlm(X_fmb, lagged_returns(t, :)');
    gamma_estimates(t, :) = mdl_fmb.Coefficients.Estimate';
end

% Calculate the average estimates and their standard errors
fmb_means = mean(gamma_estimates);
fmb_ses = std(gamma_estimates) / sqrt(T);

fprintf('Fama-MacBeth Regression Summary:\n');
for j = 1:length(fmb_means)
    fprintf('Coefficient %d Estimate: %.4f, SE: %.4f\n', j, fmb_means(j), fmb_ses(j));
end

% Print MOM Factor Values with Dates
% Generate 235 dates from August 31, 2003, with monthly intervals
startDate = datetime('31-08-2003', 'InputFormat', 'dd-MM-yyyy');
dates = startDate + calmonths(0:(length(mom_factor) - 1));

% Print MOM factor values along with corresponding dates
fprintf('Momentum Factor (MOM) Values:\n');
for i = 1:length(mom_factor)
    if ~isnan(mom_factor(i))
        fprintf('Date %s: %.4f\n', dates(i), mom_factor(i));
    else
        fprintf('Date %s: 0\n', dates(i));
    end
end

%% Section 6.1: Cross-sectional regression with MKTB and MKTDB
% Perform cross-sectional regression of average portfolio returns on beta estimates for MKTB and MKTDB

% Ensure that beta estimates for MKTB and MKTDB are properly aligned with average returns
numPortfolios = size(portfolio_data, 2);
average_returns = mean(portfolio_data, 1); % Calculate average returns for each portfolio

% Preallocate arrays for MKTB and MKTDB betas
betas_MKTB = zeros(numPortfolios, 1);
betas_MKTDB = zeros(numPortfolios, 1);

% Perform time-series regression for each portfolio to estimate betas for MKTB and MKTDB
for i = 1:numPortfolios
    mdl = fitlm([aligned_mktb, aligned_mktdb], portfolio_data(:, i));
    betas_MKTB(i) = mdl.Coefficients.Estimate(2); % Beta for MKTB
    betas_MKTDB(i) = mdl.Coefficients.Estimate(3); % Beta for MKTDB
end

% Perform cross-sectional regression of average returns on MKTB and MKTDB betas
mdl_cross_section = fitlm([betas_MKTB, betas_MKTDB], average_returns');

% Print the summary of the cross-sectional regression
fprintf('Cross-Sectional Regression Summary - MKTB and MKTDB:\n');
disp(mdl_cross_section);

% Extract key statistics
fprintf('Estimated Coefficients:\n');
coefficients = mdl_cross_section.Coefficients;
disp(coefficients);

fprintf('R-squared: %.4f\n', mdl_cross_section.Rsquared.Ordinary);
fprintf('Adjusted R-squared: %.4f\n', mdl_cross_section.Rsquared.Adjusted);

anova_stats = anova(mdl_cross_section, 'summary');
F_stat = anova_stats.F(2);
p_value = anova_stats.pValue(2);

fprintf('F-statistic: %.4f, p-value: %.4e\n', F_stat, p_value);

% Highlight significant coefficients
fprintf('Significant Factors:\n');
significant = coefficients.pValue < 0.05;
for i = 1:height(coefficients)
    if significant(i)
        fprintf('%s is significant with p-value = %.4e\n', coefficients.Properties.RowNames{i}, coefficients.pValue(i));
    end
end

% Residual Diagnostics for Heteroscedasticity and Normality
% Extract residuals from the cross-sectional regression
residuals = mdl_cross_section.Residuals.Raw;

% Perform Breusch-Pagan test for heteroscedasticity
X = [ones(length(betas_MKTB), 1), betas_MKTB, betas_MKTDB];
residuals_squared = residuals.^2;
mdl_bp = fitlm(X, residuals_squared);
bp_test_stat = mdl_bp.Rsquared.Ordinary * length(residuals);
bp_p_value = 1 - chi2cdf(bp_test_stat, size(X, 2) - 1);

fprintf('\nBreusch-Pagan Test for Heteroscedasticity:\n');
fprintf('Test Statistic: %.4f\n', bp_test_stat);
fprintf('p-value: %.4e\n', bp_p_value);
if bp_p_value < 0.05
    fprintf('The residuals exhibit significant heteroscedasticity.\n');
else
    fprintf('No significant heteroscedasticity detected in the residuals.\n');
end

% Perform Jarque-Bera test for normality
residuals_mean = mean(residuals);
residuals_std = std(residuals);
skewness = mean(((residuals - residuals_mean) / residuals_std).^3);
kurtosis = mean(((residuals - residuals_mean) / residuals_std).^4);

jb_test_stat = length(residuals) * ((skewness^2 / 6) + ((kurtosis - 3)^2 / 24));
jb_p_value = 1 - chi2cdf(jb_test_stat, 2);

fprintf('\nJarque-Bera Test for Normality:\n');
fprintf('Test Statistic: %.4f\n', jb_test_stat);
fprintf('p-value: %.4e\n', jb_p_value);
if jb_p_value < 0.05
    fprintf('The residuals deviate significantly from normality.\n');
else
    fprintf('The residuals are approximately normally distributed.\n');
end

%% Section 6.2: Cross-sectional regression with MKTDB and MOM
% Perform cross-sectional regression of average portfolio returns on beta estimates for MKTDB and MOM

% Ensure that beta estimates for MKTDB and MOM are properly aligned with average returns
numPortfolios = size(portfolio_data, 2);
average_returns = mean(portfolio_data, 1); % Calculate average returns for each portfolio

% Preallocate arrays for MKTDB and MOM betas
betas_MKTDB = zeros(numPortfolios, 1);
betas_MOM = zeros(numPortfolios, 1);

% Perform time-series regression for each portfolio to estimate betas for MKTDB and MOM
for i = 1:numPortfolios
    mdl = fitlm([aligned_mktdb, mom_factor], portfolio_data(:, i));
    betas_MKTDB(i) = mdl.Coefficients.Estimate(2); % Beta for MKTDB
    betas_MOM(i) = mdl.Coefficients.Estimate(3); % Beta for MOM
end

% Perform cross-sectional regression of average returns on MKTDB and MOM betas
mdl_cross_section = fitlm([betas_MKTDB, betas_MOM], average_returns');

% Print the summary of the cross-sectional regression
fprintf('Cross-Sectional Regression Summary - MKTDB and MOM:\n');
disp(mdl_cross_section);

% Extract key statistics
fprintf('Estimated Coefficients:\n');
coefficients = mdl_cross_section.Coefficients;
disp(coefficients);

fprintf('R-squared: %.4f\n', mdl_cross_section.Rsquared.Ordinary);
fprintf('Adjusted R-squared: %.4f\n', mdl_cross_section.Rsquared.Adjusted);

anova_stats = anova(mdl_cross_section, 'summary');
F_stat = anova_stats.F(2);
p_value = anova_stats.pValue(2);

fprintf('F-statistic: %.4f, p-value: %.4e\n', F_stat, p_value);

% Highlight significant coefficients
fprintf('Significant Factors:\n');
significant = coefficients.pValue < 0.05;
for i = 1:height(coefficients)
    if significant(i)
        fprintf('%s is significant with p-value = %.4e\n', coefficients.Properties.RowNames{i}, coefficients.pValue(i));
    end
end

% Residual Diagnostics for Heteroscedasticity and Normality
% Extract residuals from the cross-sectional regression
residuals = mdl_cross_section.Residuals.Raw;

% Perform Breusch-Pagan test for heteroscedasticity
X = [ones(length(betas_MKTDB), 1), betas_MKTDB, betas_MOM];
residuals_squared = residuals.^2;
mdl_bp = fitlm(X, residuals_squared);
bp_test_stat = mdl_bp.Rsquared.Ordinary * length(residuals);
bp_p_value = 1 - chi2cdf(bp_test_stat, size(X, 2) - 1);

fprintf('\nBreusch-Pagan Test for Heteroscedasticity:\n');
fprintf('Test Statistic: %.4f\n', bp_test_stat);
fprintf('p-value: %.4e\n', bp_p_value);
if bp_p_value < 0.05
    fprintf('The residuals exhibit significant heteroscedasticity.\n');
else
    fprintf('No significant heteroscedasticity detected in the residuals.\n');
end

% Perform Jarque-Bera test for normality
residuals_mean = mean(residuals);
residuals_std = std(residuals);
skewness = mean(((residuals - residuals_mean) / residuals_std).^3);
kurtosis = mean(((residuals - residuals_mean) / residuals_std).^4);

jb_test_stat = length(residuals) * ((skewness^2 / 6) + ((kurtosis - 3)^2 / 24));
jb_p_value = 1 - chi2cdf(jb_test_stat, 2);

fprintf('\nJarque-Bera Test for Normality:\n');
fprintf('Test Statistic: %.4f\n', jb_test_stat);
fprintf('p-value: %.4e\n', jb_p_value);
if jb_p_value < 0.05
    fprintf('The residuals deviate significantly from normality.\n');
else
    fprintf('The residuals are approximately normally distributed.\n');
end

%% Section 6.3: Cross-sectional regression with MOM and MKTB
% Perform cross-sectional regression of average portfolio returns on beta estimates for MOM and MKTB

% Ensure that beta estimates for MOM and MKTB are properly aligned with average returns
numPortfolios = size(portfolio_data, 2);
average_returns = mean(portfolio_data, 1); % Calculate average returns for each portfolio

% Preallocate arrays for MOM and MKTB betas
betas_MOM = zeros(numPortfolios, 1);
betas_MKTB = zeros(numPortfolios, 1);

% Perform time-series regression for each portfolio to estimate betas for MOM and MKTB
for i = 1:numPortfolios
    mdl = fitlm([mom_factor, aligned_mktb], portfolio_data(:, i));
    betas_MOM(i) = mdl.Coefficients.Estimate(2); % Beta for MOM
    betas_MKTB(i) = mdl.Coefficients.Estimate(3); % Beta for MKTB
end

% Perform cross-sectional regression of average returns on MOM and MKTB betas
mdl_cross_section = fitlm([betas_MOM, betas_MKTB], average_returns');

% Print the summary of the cross-sectional regression
fprintf('Cross-Sectional Regression Summary - MOM and MKTB:\n');
disp(mdl_cross_section);

% Extract key statistics
fprintf('Estimated Coefficients:\n');
coefficients = mdl_cross_section.Coefficients;
disp(coefficients);

fprintf('R-squared: %.4f\n', mdl_cross_section.Rsquared.Ordinary);
fprintf('Adjusted R-squared: %.4f\n', mdl_cross_section.Rsquared.Adjusted);

anova_stats = anova(mdl_cross_section, 'summary');
F_stat = anova_stats.F(2);
p_value = anova_stats.pValue(2);

fprintf('F-statistic: %.4f, p-value: %.4e\n', F_stat, p_value);

% Highlight significant coefficients
fprintf('Significant Factors:\n');
significant = coefficients.pValue < 0.05;
for i = 1:height(coefficients)
    if significant(i)
        fprintf('%s is significant with p-value = %.4e\n', coefficients.Properties.RowNames{i}, coefficients.pValue(i));
    end
end

% Residual Diagnostics for Heteroscedasticity and Normality
% Extract residuals from the cross-sectional regression
residuals = mdl_cross_section.Residuals.Raw;

% Perform Breusch-Pagan test for heteroscedasticity
X = [ones(length(betas_MOM), 1), betas_MOM, betas_MKTB];
residuals_squared = residuals.^2;
mdl_bp = fitlm(X, residuals_squared);
bp_test_stat = mdl_bp.Rsquared.Ordinary * length(residuals);
bp_p_value = 1 - chi2cdf(bp_test_stat, size(X, 2) - 1);

fprintf('\nBreusch-Pagan Test for Heteroscedasticity:\n');
fprintf('Test Statistic: %.4f\n', bp_test_stat);
fprintf('p-value: %.4e\n', bp_p_value);
if bp_p_value < 0.05
    fprintf('The residuals exhibit significant heteroscedasticity.\n');
else
    fprintf('No significant heteroscedasticity detected in the residuals.\n');
end

% Perform Jarque-Bera test for normality
residuals_mean = mean(residuals);
residuals_std = std(residuals);
skewness = mean(((residuals - residuals_mean) / residuals_std).^3);
kurtosis = mean(((residuals - residuals_mean) / residuals_std).^4);

jb_test_stat = length(residuals) * ((skewness^2 / 6) + ((kurtosis - 3)^2 / 24));
jb_p_value = 1 - chi2cdf(jb_test_stat, 2);

fprintf('\nJarque-Bera Test for Normality:\n');
fprintf('Test Statistic: %.4f\n', jb_test_stat);
fprintf('p-value: %.4e\n', jb_p_value);
if jb_p_value < 0.05
    fprintf('The residuals deviate significantly from normality.\n');
else
    fprintf('The residuals are approximately normally distributed.\n');
end


%% Section 6.4: Cross-sectional regression with all three factors (MKTB, MKTDB, and MOM) - Multivariate Model
% Perform cross-sectional regression of average portfolio returns on beta estimates for MKTB, MKTDB, and MOM

% Cross-sectional regression of average returns on beta estimates from the multivariate model

% Ensure the Multivariate Model Betas and Average Returns are properly aligned
cross_sec_model_multi = fitlm([betas_MKTB_multi, betas_MKTDB_multi, betas_MOM_multi], average_returns);

% Display Cross-Sectional Regression Summary
fprintf('Cross-Sectional Regression Summary - Multivariate Model:\n');
disp(cross_sec_model_multi);

% Perform Fama-Macbeth Regression for the Multivariate Model
fprintf('Performing Fama-Macbeth Regression for the Multivariate Model...\n');

% First pass: Time-series regressions to estimate betas for each portfolio (Multivariate Model)
numPortfolios = size(portfolio_data, 2);
betas_first_pass_multi = zeros(numPortfolios, 3); % For MKTB, MKTDB, and MOM

for i = 1:numPortfolios
    mdl_first_pass_multi = fitlm([aligned_mktb, aligned_mktdb, mom_factor], portfolio_data(:, i));
    betas_first_pass_multi(i, :) = mdl_first_pass_multi.Coefficients.Estimate(2:4)';
end

% Second pass: Cross-sectional regression of average returns on Multivariate Model betas
mdl_second_pass_multi = fitlm(betas_first_pass_multi, average_returns);

% Extract Fama-Macbeth estimates and p-values
fm_alpha_estimates_multi = mdl_second_pass_multi.Coefficients.Estimate(1);
fm_beta_estimates_multi = mdl_second_pass_multi.Coefficients.Estimate(2:end);
fm_pvalues_alpha_multi = mdl_second_pass_multi.Coefficients.pValue(1);
fm_pvalues_beta_multi = mdl_second_pass_multi.Coefficients.pValue(2:end);

% Print Fama-Macbeth regression results for the Multivariate Model
fprintf('Fama-Macbeth Regression Summary for the Multivariate Model:\n');
fprintf('Alpha Estimate: %.4f, p-value: %.4f\n', fm_alpha_estimates_multi, fm_pvalues_alpha_multi);
for j = 1:length(fm_beta_estimates_multi)
    fprintf('Beta Estimate %d: %.4f, p-value: %.4f\n', j, fm_beta_estimates_multi(j), fm_pvalues_beta_multi(j));
end

% Additional diagnostics: R-squared, Adjusted R-squared, F-statistics
fprintf('Cross-Sectional Regression Diagnostics for the Multivariate Model:\n');
fprintf('R-squared: %.4f\n', cross_sec_model_multi.Rsquared.Ordinary);
fprintf('Adjusted R-squared: %.4f\n', cross_sec_model_multi.Rsquared.Adjusted);

% Extract F-statistic and p-value using ANOVA for the Multivariate Model
anova_multi = anova(cross_sec_model_multi);
FStat_multi = anova_multi.F(1);
pValue_F_multi = anova_multi.pValue(1);
fprintf('F-statistic: %.4f, p-value: %.4e\n', FStat_multi, pValue_F_multi);

% Residual Diagnostics for Heteroscedasticity and Normality
% Extract residuals from the cross-sectional regression
residuals = cross_sec_model_multi.Residuals.Raw;

% Perform Breusch-Pagan test for heteroscedasticity
X = [ones(length(betas_MKTB_multi), 1), betas_MKTB_multi, betas_MKTDB_multi, betas_MOM_multi];
residuals_squared = residuals.^2;
mdl_bp = fitlm(X, residuals_squared);
bp_test_stat = mdl_bp.Rsquared.Ordinary * length(residuals);
bp_p_value = 1 - chi2cdf(bp_test_stat, size(X, 2) - 1);

fprintf('\nBreusch-Pagan Test for Heteroscedasticity:\n');
fprintf('Test Statistic: %.4f\n', bp_test_stat);
fprintf('p-value: %.4e\n', bp_p_value);
if bp_p_value < 0.05
    fprintf('The residuals exhibit significant heteroscedasticity.\n');
else
    fprintf('No significant heteroscedasticity detected in the residuals.\n');
end

% Perform Jarque-Bera test for normality
residuals_mean = mean(residuals);
residuals_std = std(residuals);
skewness = mean(((residuals - residuals_mean) / residuals_std).^3);
kurtosis = mean(((residuals - residuals_mean) / residuals_std).^4);

jb_test_stat = length(residuals) * ((skewness^2 / 6) + ((kurtosis - 3)^2 / 24));
jb_p_value = 1 - chi2cdf(jb_test_stat, 2);

fprintf('\nJarque-Bera Test for Normality:\n');
fprintf('Test Statistic: %.4f\n', jb_test_stat);
fprintf('p-value: %.4e\n', jb_p_value);
if jb_p_value < 0.05
    fprintf('The residuals deviate significantly from normality.\n');
else
    fprintf('The residuals are approximately normally distributed.\n');
end


%% Section 7: GRS Test for Model Evaluation (Question 2.1 and 2.2)
% Conduct GRS Test for evaluating whether the alphas are jointly zero

% Calculate the covariance matrix of residuals
residuals_matrix = zeros(numMonths, numPortfolios);
for i = 1:numPortfolios
    residuals_matrix(:, i) = portfolio_data(:, i) - (mktb * betas_MKTB(i) + alphas_MKTB(i));
end

cov_matrix = cov(residuals_matrix);

% Print the covariance matrix for validation
fprintf('Covariance Matrix of Residuals:\n');
disp(cov_matrix);

% Number of observations and portfolios
T = size(portfolio_data, 1);
N = numPortfolios;
K = 1; % Number of factors (MKTB)

% Calculate GRS statistic
alpha_vector = alphas_MKTB;
inv_cov_matrix = inv(cov_matrix);
grs_numerator = (T - N - K) / N * (alpha_vector' * inv_cov_matrix * alpha_vector);
grs_denominator = 1 + (mean(mktb) * mean(mktb'));
GRS_statistic = grs_numerator / grs_denominator;

% Calculate the p-value for the GRS statistic
p_value_grs = 1 - fcdf(GRS_statistic, N, T - N - K);

% Display GRS test result
fprintf('GRS Test Statistic: %.4f\n', GRS_statistic);
fprintf('GRS Test p-value: %.4e\n', p_value_grs);
if p_value_grs < 0.05
    fprintf('The GRS test is significant at the 5%% level, indicating that the alphas are jointly different from zero.\n');
else
    fprintf('The GRS test is not significant at the 5%% level, indicating that the alphas are not jointly different from zero.\n');
end


%% Step 7.1 GRS Test for MKTDB (Duration-Adjusted Risk Factor)
% calculate the GRS test statistic and p-value for the MKTDB model to determine whether the alphas are jointly zero.

% Step 1: Calculate Residuals for Each Portfolio under MKTDB
residuals_matrix_mktdb = zeros(numMonths, numPortfolios);
for i = 1:numPortfolios
    residuals_matrix_mktdb(:, i) = portfolio_data(:, i) - (mktdb * betas_MKTDB(i) + alphas_MKTDB(i));
end

% Step 2: Covariance Matrix of Residuals
cov_matrix_mktdb = cov(residuals_matrix_mktdb);



% Step 3: Parameters for GRS Test
T = size(portfolio_data, 1); % Number of time periods
N = numPortfolios;          % Number of portfolios
K = 1;                      % Number of factors (MKTDB)

% Step 4: Calculate GRS Statistic
alpha_vector_mktdb = alphas_MKTDB; % Vector of alphas for MKTDB
inv_cov_matrix_mktdb = inv(cov_matrix_mktdb); % Inverse of covariance matrix

% Numerator of GRS Test
grs_numerator_mktdb = (T - N - K) / N * (alpha_vector_mktdb' * inv_cov_matrix_mktdb * alpha_vector_mktdb);

% Denominator of GRS Test
mean_mktdb = mean(mktdb);
grs_denominator_mktdb = 1 + (mean_mktdb * mean_mktdb');

% GRS Test Statistic
GRS_statistic_mktdb = grs_numerator_mktdb / grs_denominator_mktdb;

% Step 5: Calculate p-value for the GRS Test
p_value_grs_mktdb = 1 - fcdf(GRS_statistic_mktdb, N, T - N - K);

% Step 6: Display GRS Test Results
fprintf('GRS Test Statistic (MKTDB): %.4f\n', GRS_statistic_mktdb);
fprintf('GRS Test p-value (MKTDB): %.4e\n', p_value_grs_mktdb);
if p_value_grs_mktdb < 0.05
    fprintf('The GRS test is significant at the 5%% level, indicating that the alphas are jointly different from zero.\n');
else
    fprintf('The GRS test is not significant at the 5%% level, indicating that the alphas are not jointly different from zero.\n');
end

%% 7.2: Generalized Least Squares (GLS) Estimation and Diagnostic Checks

% Objective: To perform GLS regressions for MKTB, MKTDB, and MOM factors to ensure robustness against heteroskedasticity.
% Re-estimate the factor loadings using GLS for each portfolio.
% Perform a White test or another diagnostic to check whether GLS estimation addresses heteroskedasticity issues.
% Compare GLS results to previous OLS results and evaluate improvements.

fprintf('Performing GLS estimation for robustness comparison...\n');

alphas_gls_multi = zeros(size(portfolio_data, 2), 1);
betas_MKTB_gls_multi = zeros(size(portfolio_data, 2), 1);
betas_MKTDB_gls_multi = zeros(size(portfolio_data, 2), 1);
betas_MOM_gls_multi = zeros(size(portfolio_data, 2), 1);

for i = 1:size(portfolio_data, 2)
    weights = 1 ./ (aligned_mktb(13:end) .^ 2);  % Example weighting scheme for GLS
    mdl_gls = fitlm([aligned_mktb(13:end), aligned_mktdb(13:end), mom_factor(13:end)], portfolio_data(13:end, i), 'Weights', weights);
    alphas_gls_multi(i) = mdl_gls.Coefficients.Estimate(1);
    betas_MKTB_gls_multi(i) = mdl_gls.Coefficients.Estimate(2);
    betas_MKTDB_gls_multi(i) = mdl_gls.Coefficients.Estimate(3);
    betas_MOM_gls_multi(i) = mdl_gls.Coefficients.Estimate(4);
    
    % Print R-squared and Adjusted R-squared for GLS regression
    fprintf('Portfolio %d GLS Regression Diagnostics:\n', i);
    fprintf('R-squared: %.4f\n', mdl_gls.Rsquared.Ordinary);
    fprintf('Adjusted R-squared: %.4f\n', mdl_gls.Rsquared.Adjusted);
end

% Perform White test for heteroskedasticity in GLS residuals
fprintf('Performing White test for heteroskedasticity in GLS residuals...\n');
heteroskedasticity_gls = zeros(size(portfolio_data, 2), 1);
for i = 1:size(portfolio_data, 2)
    residuals_gls = portfolio_data(13:end, i) - ([aligned_mktb(13:end), aligned_mktdb(13:end), mom_factor(13:end)] * [betas_MKTB_gls_multi(i); betas_MKTDB_gls_multi(i); betas_MOM_gls_multi(i)]);
    [~, ~, ~, ~, stats] = regress(residuals_gls .^ 2, [ones(length(residuals_gls), 1), aligned_mktb(13:end), aligned_mktdb(13:end), mom_factor(13:end)]);
    pValue_White_gls = stats(3); % p-value from the White test
    testStatistic_White_gls = stats(1); % Test statistic from the White test
    heteroskedasticity_gls(i) = pValue_White_gls < 0.05;
    fprintf('White test for Portfolio %d: Test Statistic = %.4f, p-value = %.4e\n', i, testStatistic_White_gls, pValue_White_gls);
end

% Count portfolios with significant heteroskedasticity in GLS residuals
num_heteroskedastic_gls = sum(heteroskedasticity_gls);

fprintf('Portfolios with significant heteroskedasticity after GLS: %d portfolios\n', num_heteroskedastic_gls);

% Compare GLS results with OLS results
fprintf('Comparing GLS results with OLS results...\n');
num_significant_ols = sum(alpha_pvals_multi < 0.05);
num_significant_gls = sum(heteroskedasticity_gls);

% Print alpha and beta coefficients for both OLS and GLS side by side for each portfolio
for i = 1:numPortfolios
    fprintf('Portfolio %d Comparison (OLS vs GLS):\n', i);
    fprintf('OLS Alpha: %.4f, GLS Alpha: %.4f\n', alphas_multi(i), alphas_gls_multi(i));
    fprintf('OLS Beta MKTB: %.4f, GLS Beta MKTB: %.4f\n', betas_MKTB_multi(i), betas_MKTB_gls_multi(i));
    fprintf('OLS Beta MKTDB: %.4f, GLS Beta MKTDB: %.4f\n', betas_MKTDB_multi(i), betas_MKTDB_gls_multi(i));
    fprintf('OLS Beta MOM: %.4f, GLS Beta MOM: %.4f\n', betas_MOM_multi(i), betas_MOM_gls_multi(i));
end

fprintf('Number of portfolios with significant alphas (OLS): %d\n', num_significant_ols);
fprintf('Number of portfolios with significant alphas (GLS): %d\n', num_significant_gls);

%% Step 7.3 REMAKING MOM factors for GRS for multivariate 
% Extract MKTB, MKTDB, and portfolio data (excluding Date column)
mktb = risk_factors.MKTB;
mktdb = risk_factors.MKTDB;
portfolios = portfolio_returns(:, 2:end);
portfolioNames = portfolios.Properties.VariableNames;
portfolio_data = table2array(portfolios);

% Generate 12-month lagged returns for each portfolio
numMonths = size(portfolio_data, 1);
lagged_returns = nan(numMonths - 12, size(portfolio_data, 2));

for i = 13:numMonths
    lagged_returns(i - 12, :) = mean(portfolio_data(i-12:i-1, :), 1);
end

% Calculate the average lagged returns for each portfolio
avg_lagged_returns = nanmean(lagged_returns);

[~, ranks] = sort(avg_lagged_returns);
ranked_deciles = ceil(ranks / (size(portfolio_data, 2) / 10));

% Create MOM factor by longing the top decile and shorting the bottom decile
mom_factor = nan(numMonths, 1);  % Change length of MOM factor to be equal to numMonths
for t = 13:numMonths
    top_decile = mean(portfolio_data(t, ranked_deciles == 10));
    bottom_decile = mean(portfolio_data(t, ranked_deciles == 1));
    mom_factor(t) = top_decile - bottom_decile;
end
fprintf('Momentum factor (MOM) constructed successfully.\n');

% Print summary statistics for the MOM factor
mean_mom = mean(mom_factor, 'omitnan');
std_mom = std(mom_factor, 'omitnan');
fprintf('Momentum Factor (MOM) Summary:\n');
fprintf('Mean: %.4f\n', mean_mom);
fprintf('Standard Deviation: %.4f\n', std_mom);

% Additional analysis for momentum trends (Pre-crisis vs Post-crisis)
mean_pre_crisis = mean(mom_factor(1:round(end/2)), 'omitnan');
mean_post_crisis = mean(mom_factor(round(end/2)+1:end), 'omitnan');
fprintf('Momentum Trend Analysis:\n');
fprintf('Mean Pre-Crisis: %.4f\n', mean_pre_crisis);
fprintf('Mean Post-Crisis: %.4f\n', mean_post_crisis);

% Print MOM Factor Values with Dates
% Generate 235 dates from August 31, 2003, with monthly intervals
startDate = datetime('31-08-2003', 'InputFormat', 'dd-MM-yyyy');
dates = startDate + calmonths(0:(length(mom_factor) - 1));

% Print MOM factor values along with corresponding dates
fprintf('Momentum Factor (MOM) Values:\n');
for i = 1:length(mom_factor)
    if ~isnan(mom_factor(i))
        fprintf('Date %s: %.4f\n', dates(i), mom_factor(i));
    else
        fprintf('Date %s: 0\n', dates(i));
    end
end
% Section: GRS Test for Multivariate Model
% Step 1: Align factors and portfolio returns
aligned_mktb = mktb(13:end);
aligned_mktdb = mktdb(13:end);
aligned_mom = mom_factor(13:end); % Corrected MOM factor
aligned_portfolio_returns = portfolio_data(13:end, :); % Align portfolio returns

% Initialize variables
numPortfolios = size(aligned_portfolio_returns, 2);
numMonths = size(aligned_portfolio_returns, 1);
numFactors = 3; % MKTB, MKTDB, MOM
alphas = zeros(numPortfolios, 1);
betas = zeros(numPortfolios, numFactors);
residuals_matrix = zeros(numMonths, numPortfolios);

% Step 2: Perform time-series regressions for each portfolio
for i = 1:numPortfolios
    % Construct design matrix with factors and intercept
    X = [ones(numMonths, 1), aligned_mktb, aligned_mktdb, aligned_mom];
    y = aligned_portfolio_returns(:, i);

    % Estimate coefficients using least squares
    coeffs = X \ y;
    alphas(i) = coeffs(1); % Intercept (alpha)
    betas(i, :) = coeffs(2:end)'; % Betas for MKTB, MKTDB, MOM

    % Calculate residuals
    residuals_matrix(:, i) = y - X * coeffs;
end

% Step 3: Calculate the covariance matrix of residuals
cov_matrix_residuals = cov(residuals_matrix);

% Step 4: Calculate GRS test statistic
T = numMonths; % Number of time periods
N = numPortfolios; % Number of portfolios
K = numFactors; % Number of factors

% Inverse of covariance matrix of residuals
inv_cov_matrix_residuals = inv(cov_matrix_residuals);

% Mean and covariance matrix of factors
mean_factors = mean([aligned_mktb, aligned_mktdb, aligned_mom]);
factor_cov_matrix = cov([aligned_mktb, aligned_mktdb, aligned_mom]);
inv_factor_cov_matrix = inv(factor_cov_matrix);

% Calculate GRS numerator and denominator
GRS_numerator = (T - N - K) / N * (alphas' * inv_cov_matrix_residuals * alphas);
GRS_denominator = 1 + mean_factors * inv_factor_cov_matrix * mean_factors';

% Final GRS statistic
GRS_statistic = GRS_numerator / GRS_denominator;

% Step 5: Calculate p-value for the GRS statistic
p_value_grs = 1 - fcdf(GRS_statistic, N, T - N - K);

% Display GRS test results
fprintf('GRS Test Statistic (Multivariate Model): %.4f\n', GRS_statistic);
fprintf('GRS Test p-value (Multivariate Model): %.4e\n', p_value_grs);
if p_value_grs < 0.05
    fprintf('The GRS test is significant at the 5%% level, indicating that the alphas are jointly different from zero.\n');
else
    fprintf('The GRS test is not significant at the 5%% level, indicating that the alphas are not jointly different from zero.\n');
end
%% Section 7.4: White's Test for Multivariate Model (Section 6.4)
% Perform White's test for heteroskedasticity on residuals from the multivariate model in Section 6.4.

fprintf('Performing White''s Test for the Multivariate Model...\n');

% Extract residuals from the cross-sectional regression in Section 6.4
residuals_multi = cross_sec_model_multi.Residuals.Raw;
residuals_squared = residuals_multi.^2;

% Construct the design matrix for the White's test
X_white_multi = [ones(size(betas_MKTB_multi)), ...
                 betas_MKTB_multi, betas_MKTDB_multi, betas_MOM_multi, ...
                 betas_MKTB_multi.^2, betas_MKTDB_multi.^2, betas_MOM_multi.^2, ...
                 betas_MKTB_multi.*betas_MKTDB_multi, ...
                 betas_MKTB_multi.*betas_MOM_multi, ...
                 betas_MKTDB_multi.*betas_MOM_multi];

% Perform the regression for White's test
mdl_white_multi = fitlm(X_white_multi, residuals_squared);

% Extract test statistic and p-value
bp_test_stat_multi = mdl_white_multi.Rsquared.Ordinary * length(residuals_multi);
bp_p_value_multi = 1 - chi2cdf(bp_test_stat_multi, size(X_white_multi, 2) - 1);

% Display White's test results
fprintf('White''s Test for Multivariate Model:\n');
fprintf('Test Statistic: %.4f\n', bp_test_stat_multi);
fprintf('P-value: %.4e\n', bp_p_value_multi);
if bp_p_value_multi < 0.05
    fprintf('The residuals exhibit significant heteroskedasticity.\n');
else
    fprintf('No significant heteroskedasticity detected in the residuals.\n');
end

% Additional Diagnostic Printout
fprintf('\nDetailed Regression Output for White''s Test:\n');
disp(mdl_white_multi);


%% Section 8: Graphs
% Load Data (Make sure to replace the filename and sheet name with your actual data file)
data = readtable('C:\Users\nairp\OneDrive\Desktop\Asset pricing Group Project\Portfolio Data.xlsx', 'Sheet', 'Portfolio Returns');

% Calculate Average Returns for Each Portfolio
average_returns = mean(data{:, 2:end}, 1);  % Assuming the first column is the date and the rest are portfolio returns

% Plotting the Bar Chart for Average Returns Across Portfolios
figure;
bar(average_returns);

title('Average Returns Across Portfolios');
xlabel('Portfolio Number');
ylabel('Average Return');

% Customize X-ticks to represent portfolio numbers
xticks(1:length(average_returns));
xticklabels(data.Properties.VariableNames(2:end));

% Display grid for better readability
grid on;



% Time-Series Plot of MKTB Factor Returns (Question 2.3)
% Plot the time-series of MKTB factor returns to analyze trends and volatility

figure;
plot(dates, mktb, 'b', 'LineWidth', 1.5);

% Add labels, title, and grid
xlabel('Time');
ylabel('MKTB Factor Return');
title('Time-Series Plot of MKTB Factor Returns');
grid on;

% Customize x-axis to display dates properly
datetick('x', 'yyyy-mm', 'keepticks', 'keeplimits');

% Add horizontal line at zero to indicate positive/negative returns
hold on;
yline(0, 'k--', 'LineWidth', 1.2);

% Adjust the limits of the y-axis to properly visualize the returns
ylim([min(mktb) - 0.01, max(mktb) + 0.01]);

% Time-Series Line Plot of MKTDB Factor Returns

% Assuming 'dates' contains the date values and 'mktdb' contains the MKTDB factor returns.
figure;
plot(dates, mktdb, 'b-', 'LineWidth', 1.5);

% Add labels, title, and grid
xlabel('Date');
ylabel('MKTDB Returns');
title('Time-Series Plot of MKTDB Factor Returns');
grid on;

datacursormode on; % Allow data cursor mode for interactive exploration of values

% Customize axes
xlim([min(dates), max(dates)]);
dateFormat = 'mmm-yyyy';
datetick('x', dateFormat, 'keeplimits', 'keepticks');
ylim([min(mktdb) - 0.01, max(mktdb) + 0.01]);

% Calculate the average return across all portfolios for each time period
average_portfolio_returns = mean(portfolio_data, 2);

% Plot the time-series line plot of returns for each individual portfolio
figure;
hold on;
for i = 1:size(portfolio_data, 2)
    plot(dates, portfolio_data(:, i), 'LineWidth', 1.2);
end
hold off;

% Add labels, title, and grid
xlabel('Date');
ylabel('Portfolio Return');
title('Time-Series Line Plot of Individual Portfolio Returns');
grid on;

% Customize axes to improve readability
datetick('x', 'yyyy', 'keepticks'); % Display years on x-axis
xlim([dates(1), dates(end)]);
ylim([min(portfolio_data, [], 'all') - 0.01, max(portfolio_data, [], 'all') + 0.01]);

% Add a legend to identify individual portfolios
legend(arrayfun(@(x) sprintf('Portfolio %d', x), 1:size(portfolio_data, 2), 'UniformOutput', false), 'Location', 'bestoutside');


%% Section 9: Expanded Analysis with GLS Results Integration into Fama-MacBeth
% Incorporate GLS Results into the Fama-MacBeth Analysis
% Conduct Time-Series Analysis of GLS-based Alphas

fprintf('Starting Expanded Analysis focusing on GLS integration into Fama-MacBeth and time-series analysis of GLS-based alphas...\n');

% Perform Fama-MacBeth Regression for GLS-based Multi-Factor Model
fprintf('Performing Fama-MacBeth Regression using GLS results for Multi-Factor Model...\n');
X_fmb_gls = [betas_MKTB_multi, betas_MKTDB_multi, betas_MOM_multi];
T = size(lagged_returns, 1);
gamma_estimates_gls = zeros(T, size(X_fmb_gls, 2) + 1);

for t = 1:T
    mdl_fmb_gls = fitlm(X_fmb_gls, lagged_returns(t, :)');
    gamma_estimates_gls(t, :) = mdl_fmb_gls.Coefficients.Estimate';
end

% Calculate the average estimates and their standard errors for GLS-based model
fmb_means_gls = mean(gamma_estimates_gls);
fmb_ses_gls = std(gamma_estimates_gls) / sqrt(T);

fprintf('Fama-MacBeth Regression Summary for GLS-based Model:\n');
for j = 1:length(fmb_means_gls)
    fprintf('Coefficient %d Estimate: %.4f, SE: %.4f\n', j, fmb_means_gls(j), fmb_ses_gls(j));
end

% Time-Series Analysis of GLS-based Alphas
fprintf('Performing time-series analysis of GLS-based alphas...\n');
gls_alphas_time_series = alphas_multi;
mean_gls_alpha = mean(gls_alphas_time_series);
std_gls_alpha = std(gls_alphas_time_series);

fprintf('Time-Series Analysis of GLS-based Alphas:\n');
fprintf('Mean of GLS-based Alphas: %.4f\n', mean_gls_alpha);
fprintf('Standard Deviation of GLS-based Alphas: %.4f\n', std_gls_alpha);

fprintf('Section 9: Expanded Analysis completed successfully.\n');
