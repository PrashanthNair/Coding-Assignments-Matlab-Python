%% MATLAB Code for Fama-French Factor Model Analysis

% Part (a): Load data, construct excess returns, and perform analysis
% This code combines the strengths of both versions for efficiency, readability, and robustness.

% Clear the workspace and close all previous figures to start fresh
clear; close all; clc;

%% Load the data files into MATLAB
% Load the 42 portfolios data (FF42.txt)
% This matrix has 733 rows and 42 columns
%Load RF.txt which contains the monthly risk-free rate
% This is a vector with 733 rows
% Load FF6.txt which contains the six Fama-French factors
% This matrix has 733 rows and 6 columns
load FF42.txt 
load FF6.txt 
load RF.txt 

% Step 2: Construct Excess Returns (R^e)
% The excess returns for each portfolio are calculated by subtracting the risk-free rate from the returns.
% We'll construct a matrix called 'Re' which has the same number of rows as 'FF42' (733 rows) and 42 columns.
% Construct the excess returns matrix Re
Re = FF42 - RF; % Broadcasting the risk-free rate across each portfolio return
% Define the number of time periods (T)
T = size(Re, 1);
% Explanation:
% 'RF' is broadcasted across each of the 42 portfolios for each month, meaning the risk-free rate is subtracted
% from every value in the respective row of 'FF42'. The result is an excess returns matrix 'Re'
% Step 3: Extract the Six Fama-French Factors (f)
% These factors are extracted directly from FF6.txt, as it already contains the six relevant factors.
f = FF6;
% Define the number of factors (k)
k = size(f, 2); % Number of columns in the factor matrix

% Step 4: Testing for Stationarity
% To test whether the excess returns (R^e) and the factors (f) are stationary, we use the Augmented Dickey-Fuller (ADF) test.
% MATLAB provides the function 'adftest' which performs this test.
% Vectorized ADF test for Re and f to improve efficiency
p_values_Re = arrayfun(@(i) adftest(Re(:, i)), 1:size(Re, 2));
p_values_f = arrayfun(@(i) adftest(f(:, i)), 1:size(f, 2));
% Step 5: Analyze the Results of the ADF Test
% ADF test null hypothesis: Data has a unit root (i.e., non-stationary).
% If p-value < 0.05, we reject the null hypothesis, implying stationarity.
% Display results for Re
fprintf('ADF Test Results for Excess Returns (R^e):\n');
for i = 1:42
    if p_values_Re(i) < 0.05
        fprintf('Portfolio %d: Stationary (p-value = %.4f)\n', i, p_values_Re(i));
    else
        fprintf('Portfolio %d: Non-stationary (p-value = %.4f)\n', i, p_values_Re(i));
    end
end
% Display results for Fama-French factors
fprintf('\nADF Test Results for Fama-French Factors (f):\n');
for i = 1:6
    if p_values_f(i) < 0.05
        fprintf('Factor %d: Stationary (p-value = %.4f)\n', i, p_values_f(i));
    else
        fprintf('Factor %d: Non-stationary (p-value = %.4f)\n', i, p_values_f(i));
    end
end

% Step 6: Summary of Stationarity Test
% The output of this code will indicate for each of the 42 portfolios and the 6 factors whether they are stationary or not.
% If the p-value is less than 0.05, we conclude that the series is stationary; otherwise, we consider it non-stationary.
% The analysis here helps us understand the time-series properties of the excess returns and the factors. Stationary time series
% are crucial for many econometric models as they ensure consistency in statistical inference.

% Part b - Estimating Alpha and Beta for Multiple Linear Regression
% Step 7: Estimating Alpha and Beta for Multiple Linear Regression (Part b)
% We now proceed to estimate the parameters (alpha and beta) for the regression model Re_t = alpha + beta * f_t + epsilon_t
% using Ordinary Least Squares (OLS). Initialize matrices to store alpha and beta estimates

betahat = []; % Matrix to store beta estimates for each portfolio (6 factors)
uhat = []; % Matrix to store residuals for each portfolio
R2 = []; % Vector to store R^2 values for each portfolio

% Perform OLS regression for each of the 42 portfolios
for i = 1:42
    % Set up the regression model
    Y = Re(:, i); % Dependent variable: excess returns for portfolio i
    F = [ones(size(f, 1), 1), f]; % Independent variables: constant term and the six factors
    
    % Estimate OLS coefficients
    b = inv(F' * F) * F' * Y; % OLS formula: (X'X)^(-1) X'Y
    
    % Store beta estimates
    betahat(end+1, :) = b'; % Append the beta estimates
    
    % Compute residuals
    uhat(:, end+1) = Y - F * b; % Residuals = actual values - fitted values
    
    % Compute R-squared
    SS_total = sum((Y - mean(Y)).^2); % Total sum of squares
    SS_residual = uhat(:, end)' * uhat(:, end); % Residual sum of squares
    R2(end+1, 1) = 1 - (SS_residual / SS_total); % R^2 calculation
end

% Compute and display the average R-squared across all portfolios
avgR2 = mean(R2);

% Step 8: Report the Results
% Display the beta estimates and R-squared values for each portfolio
fprintf('\nOLS Regression Results:\n');
for i = 1:42
    fprintf('Portfolio %d:\n', i);
    fprintf(' Beta: ');
    fprintf('%.4f ', betahat(i, :));
    fprintf('\n R-squared: %.4f\n\n', R2(i));
end

fprintf('Average R-squared across all portfolios: %.4f\n', avgR2);

% Step 9: Interpretation of Results
% The R-squared values indicate how well the six Fama-French factors explain the time-series variation in the excess returns.
% A higher R-squared suggests a better fit, meaning that the factors explain a larger portion of the variability in returns.
% The average R-squared value gives an overall measure of the explanatory power of the model across all portfolios.

%% Part c %%
% Cross-sectional Regression to Estimate the Zero-beta Rate (γ0) and Factor Risk Premia (γ1)

% Define N and K
N = 42; % Number of portfolios
K = 6;  % Number of factors

% Step 10: Calculate Average Excess Returns (Rebar)
% Average excess return for each of the 42 portfolios
Rebar = []; 
for i = 1:N 
    Rebar(end+1, 1) = mean(Re(:, i)); 
end 

% Step 11: Coefficient Estimates from Time-series Regression
% Estimated betas for the six factors
B = betahat(:, 2:end); % Exclude the constant term from betahat

% Step 12: Construct Xhat Matrix
% Xhat includes a column of ones and the betas from time-series regression
Xhat = [ones(N, 1) B];

% Step 13: Estimate Gamma (gamahat)
% Using the formula: gamahat = (Xhat' * Xhat)^(-1) * Xhat' * Rebar
gamahat = inv(Xhat' * Xhat) * Xhat' * Rebar;

% Step 14: Calculate Ahat Matrix
% Ahat is used in the covariance estimation
Ahat = inv(Xhat' * Xhat) * Xhat';

% Step 15: Covariance of gamahat
% Extract gamahat1 (factor risk premia estimates)
gamahat1 = gamahat(2:end, 1);  % Exclude the zero-beta rate estimate

% Estimate residual covariance (Ehat) from time-series regression residuals
Ehat = cov(uhat); % Covariance of residuals

% Define zero vector (zerok) and Vf (covariance of factors)
zerok = zeros(K, 1); % Zero vector of length K
Vf = cov(f); % Covariance matrix of factors

% Step 16: Calculate Variance-Covariance Matrix (VS)
% Formula: VS = (1 + gamahat1' * inv(Vf) * gamahat1) * Ahat * Ehat * Ahat' + [0 zerok'; zerok Vf]
m = [0 zerok'; zerok Vf]; 
VS = (1 + gamahat1' * inv(Vf) * gamahat1) * Ahat * Ehat * Ahat' + m;

% Step 17: Calculate t-statistics for Gamma Estimates
% Standard errors of gamahat (square root of diagonal elements of VS)
segamahat = sqrt(diag(VS) / size(Re, 1)); % Divide by number of observations (T)

t_stats = gamahat ./ segamahat; % t-statistics for each gamma estimate

% Step 18: Report Gamma Estimates and t-statistics
fprintf('\nGamma Estimates and t-statistics:\n');
for i = 1:length(gamahat)
    fprintf('Gamma %d Estimate: %.4f, t-statistic: %.4f\n', i - 1, gamahat(i), t_stats(i));
end

% Step 19: Critical Values for Significance
% Calculate critical values for 5% and 10% significance levels
c5 = tinv(0.975, N - K - 1); % Critical Value at 5% significance level
c10 = tinv(0.95, N - K - 1); % Critical Value at 10% significance level

% Step 20: Time Series Sample Mean of Factors (Bbar)
% Compare time series sample means with cross-sectional estimates
Bbar = [];
for i = 1:K
    Bbar(end+1, 1) = mean(B(:, i));
end

% Report Comparison
fprintf('\nComparison between Time Series Sample Means and Cross-sectional Estimates:\n');
for i = 1:K
    fprintf('Factor %d Time Series Mean: %.4f, Cross-sectional Estimate: %.4f\n', i, Bbar(i), gamahat1(i));
end


%Part D  Cross-sectional R^2 Calculation
% Step 20 Calculate the cross-sectional R^2 to evaluate how well the six factors explain the variation in expected excess returns.
% Step 20.1: Calculate mean excess returns (Rebar)
Rebar = mean(Re)'; % Mean of excess returns for each of the 42 portfolios (column vector of size 42x1)

% Step 20.2: Calculate sample pricing errors (ehat)
e_hat = Rebar - Xhat * gamahat;

% Step 20.3: Calculate e_0
e_0 = Rebar - mean(Rebar); % e_0 represents deviation from average (size 42x1)

% Step 20.4: Calculate Cross-sectional R^2 (R_cs^2)
R_cs_squared = 1 - (e_hat' * e_hat) / (e_0' * e_0); % Cross-sectional R^2

% Step 21: Report Cross-sectional R^2
fprintf('\nCross-sectional R^2 (R_cs^2): %.4f\n', R_cs_squared);

% Step 22: Interpretation of Cross-sectional R^2
% The cross-sectional R^2 (R_cs^2) provides an indication of how well the six factors explain the cross-sectional variation
% in the expected excess returns across the 42 portfolios. A higher R_cs^2 suggests that the factors have strong explanatory power
% for the differences in expected returns among the portfolios.

% Part e GRS Test
%Step 23 - To formally test whether the six-factor model is rejected by the data, we use the Gibbons, Ross, and Shanken (GRS) test.
% The grs.m function is provided, and we use it here to perform the GRS test.

% Run the GRS test using the grs function
% Inputs for GRS: Excess returns (Re), factor returns (f), risk-free rate (RF)
r1 = f; 
r2 = Re; 
[stat,pval] = GRS(r1,r2); 

% Step 24: Report GRS Test Results
fprintf('\nGRS Test Results:\n');
fprintf('GRS Test Statistic: %.4f\n', stat);
fprintf('p-value: %.4f\n', pval);

% Step 25: Interpretation of GRS Test Results
% The GRS test helps determine if the six-factor model is rejected by the data.
% If the p-value is less than 0.05, we reject the null hypothesis, indicating that the factors are not sufficient
% to explain the variation in the data. Conversely, if the p-value is greater than 0.05, we fail to reject the null,
% suggesting that the model adequately captures the variation in returns.

%Step 26: Part f in Summary file

% Residual Plots
% Plot residuals for Portfolio 1 over time
figure;

% Time series residual plot for Portfolio 1
subplot(2, 1, 1);
plot(1:T, uhat(:, 1), 'LineWidth', 1.2); % Use residuals of the first portfolio
title('Residuals over Time (Portfolio 1)');
xlabel('Time');
ylabel('Residuals');
grid on;

% Histogram of residuals for Portfolio 1
subplot(2, 1, 2);
histogram(uhat(:, 1), 20); % Use 20 bins for the histogram
title('Residual Distribution (Portfolio 1)');
xlabel('Residuals');
ylabel('Frequency');
grid on;

% Plot residuals for all portfolios over time
figure;
for i = 1:N
    subplot(ceil(N / 5), 5, i); % Divide into multiple subplots
    plot(1:T, uhat(:, i), 'LineWidth', 1.2);
    title(['Residuals for Portfolio ', num2str(i)]);
    xlabel('Time');
    ylabel('Residuals');
    grid on;
end
% Calculate Variance Inflation Factors (VIF)
vif_values = zeros(1, k);
for i = 1:k
    % Exclude the ith factor
    Xi = f(:, [1:i-1, i+1:end]); % All factors except the ith
    Yi = f(:, i); % The ith factor
    % Perform OLS regression for Yi on Xi
    beta_i = (Xi' * Xi) \ (Xi' * Yi); % OLS formula
    residual_i = Yi - Xi * beta_i; % Residuals
    % Calculate R-squared
    R2_i = 1 - sum(residual_i.^2) / sum((Yi - mean(Yi)).^2); 
    % Calculate VIF
    vif_values(i) = 1 / (1 - R2_i);
end
% Display VIF results
disp('Variance Inflation Factors (VIF):');
disp(vif_values);