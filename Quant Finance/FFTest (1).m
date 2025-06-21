clear all 
load FF42.txt 
load FF6.txt 
load RF.txt 

%% Part A
%%Build the structure of excess return Re and fatcors f 
Rf = RF(:,:); %% Risk free rate for each month
Rm = FF42(:,:); %% Market return on 42 portfolios
Re = Rm - Rf; %% Excess Return (Return on market - Return on risk free rate)

T = size(Re,1); %% Define Number of samples for each portfolios   
N = 42; %% Define Number of Portfoliois
k = 6;  %% Define Number of Factors

%%%Factors columns Set Up
MKT = FF6(:,1); 
SMB = FF6(:,2); 
HML = FF6(:,3); 
RMW = FF6(:,4); 
CMA = FF6(:,5); 
MOM = FF6(:,6);

f = [MKT SMB HML RMW CMA MOM]; %%Six Factors Matrix
Vf = cov(f,1); %%Population covariance matrix Vf
F = [ones(T,1) f];  %Factors; add ones as intercept to capture the Ave return not capturing by factors.

disp(Vf)

%We use Augmented Dickey-Fuller (ADF) test on the Re(:,i) and f(:,i).
%adftest function provided by Matlab
% Non-stationarity test for Re 
h1 = []; %%Matrix to store the result of the test for excess return
for i = 1:N 
h1(1, end+1) = adftest(Re(:,i)); 
end 

% Non-stationarity test for f 
h2 = []; %%Matrix to store the result of the test for factors
for i = 1:k
h2(1, end+1) = adftest(f(:,i)); 
end 

disp(h1);
disp(h2);


%% Part b %% 
% OLS estimators %%Then we can generate estimated alpha and beta
betahat = []; %% Store the alpha and beta estimates
for i = 1:N 
    betahat(end+1,:) = inv(F' * F) * F' * Re(:,i);  %%Formula: OLS estimated beta= (X'X)^(-1)*X'y (here y=Re;X=F)
end 

%uhat is estimated residuals 
uhat = []; %%Store the residual estimates
for i = 1:N 
y = Re(:,i); 
uhat(:, end+1) = y - F * betahat(i,:)';  %%%% Residual = excess return-estimated beta * X; here X=F
end 

% R_squared and avgR2 (average R-squared)
R2 = []; %%Store the R-squared estimates
for i = 1:N 
y = Re(:,i); 
R2(end+1,1) = 1 - uhat(:,i)'*uhat(:,i)/(sum((y - mean(y)).^2)); %% R-squared=1-SSR/SST
end 

avgR2 = mean(R2); 
disp('R-squared:');
disp(R2);
disp('Average R-squared:');
disp (avgR2);

%% part c %% 
%%Average excess return: Rebar--Given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (2)
Rebar = []; 
for i = 1:N 
Rebar(end+1,1) = mean(Re(:,i)); 
end 
 
%Coefficient estimates for the factors from the time-series regression 
B = betahat(:,2:end); %% Estimated beta for six factors
Xhat = [ones(N,1) B];  	%%% Given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (4)
gamahat = inv(Xhat'*Xhat)*Xhat'*Rebar; %%%Given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (3).
Ahat = inv(Xhat'*Xhat)*Xhat'; %%%Given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (4)

% Covariance of gamahat 
gamahat1 = gamahat(2:end,1);  %% Extract estimates without intercept
Ehat = cov(uhat); %%According to shaken(1992).
zerok = zeros(k,1); % k = 6 
%%formula below are given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (4)
m = [0 zerok'; zerok Vf]; 
VS = (1 + gamahat1'*inv(Vf)*gamahat1)*Ahat*Ehat*Ahat' + m;  

% t-statistic
segamahat = sqrt(diag(VS)/T); %%%Given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (4)--Standard errors
[gamahat gamahat./(segamahat)]; %%%given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (4)
disp('Cross-sectional estimates and their t-statistic:');
disp([gamahat gamahat./(segamahat)]);

c5 = tinv(0.025,N-k-1); %Critical Value at 5% significance level
c10 = tinv(0.05,N-k-1); %Critical Value at 10% significance level
disp('Critical Value at 5%:');
disp(c5);
disp('Critical Value at 10%:');
disp(c10);

%%%Time Series sample Mean
Bbar = [];
for i=1:k %%k=6
    Bbar(end+1,1) = mean(B(:,i));
end
[Bbar gamahat1]; %% Comparison between time series sample mean and cross section estimates.

disp('Time-series sample mean and Cross-sectional estimates:');
disp([Bbar gamahat1]);
%% Part d%% 
ehat = Rebar - Xhat*gamahat; %%given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (5)
ehat0 = Rebar - ones(N,1)*(ones(N,1)'*Rebar/N); %%given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (6)
R2cs = 1 - ehat'*ehat/(ehat0'*ehat0);  %%given in IB9X60_QMF_Group_Work_2024.pdf--Instruction (6)
disp('R-squared in cross-sectional:');
disp(R2cs);

%% Part e%% 
%%%According to the application of GRS test; According to Gibbons, Ross, and Shanken (GRS, 1989).

r1 = f; 
r2 = Re; 
[stat,pval] = grs(r1,r2); %%% We save the grs function code in our file, so can direct apply here.

disp('GRS test Intercept and its p-value:');
disp([stat pval])

%% Residual Plot
figure;
% Time-series residual plot
subplot(2, 1, 1);
plot(1:T, uhat(:, 1), 'LineWidth', 1.2); %
title('Residuals over Time (Portfolio 1)');
xlabel('Time');
ylabel('Residuals');
grid on;

% Residual Histogram
subplot(2, 1, 2);
histogram(uhat(:, 1), 20); % Use 20 bins
title('Residual Distribution (Portfolio 1)');
xlabel('Residuals');
ylabel('Frequency');
grid on;

% Plot Time-series of all portfolio's residual
figure;
for i = 1:N
    subplot(ceil(N / 5), 5, i); % Divide plot into multiple subplot
    plot(1:T, uhat(:, i), 'LineWidth', 1.2);
    title(['Residuals for Portfolio ', num2str(i)]);
    xlabel('Time');
    ylabel('Residuals');
    grid on;
end




