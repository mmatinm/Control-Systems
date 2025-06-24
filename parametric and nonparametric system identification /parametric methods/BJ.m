
% Load and preprocess data
data = load('D:\data2.mat'); 
ugt = detrend(data.u1);  
ygt = detrend(data.y1);  
upt = detrend(data.u2);  
ypt = detrend(data.y2);  
Nt = length(ugt);

delay = 1;  
data1 = iddata(ygt, ugt, 1);  
data2 = iddata(ypt, upt, 1);  

na_range = 2:8;  
nk = delay; 

aic_values = zeros(length(na_range), 1);
fpe_values = zeros(length(na_range), 1);
fit_values2 = zeros(length(na_range), 1);
fit_values2_reduced = zeros(length(na_range), 1);

%%%%% GN -------------------------------------
for na = na_range
    nb = 3;
    nc = na;
    nd = na;
    nf = 3;    

    opt = bjOptions;
    opt.SearchMethod = 'gn'; 
    opt.Focus = 'prediction';
    model = bj(data1, [nb nc nd nf nk] , opt);


    y_valid = data2.OutputData;
    y_pred = predict(model, data2);
    [~,fit_ss,~] = compare(data2, model);
    fit_values2(na-1) = fit_ss; %(1 - goodnessOfFit(y_pred.OutputData, y_valid, 'NRMSE') ) * 100;

    residuals_original = y_valid - y_pred.OutputData; 
    p = na + nb ; 
    RSS_original = sum(residuals_original.^2); 
    aic_values(na-1) = Nt * log(RSS_original / Nt) + 2 * p;
    fpe_values(na-1) = (RSS_original / Nt) / (1 - p / Nt)^2;

    if na == 2 
        reduced_model = reduce(model,2);
    else
        reduced_model = reduce(model,2);
    end
    %y_pred_reduced = predict(reduced_model, data2);
    [~,fit_ss,~] = compare(data2, reduced_model);
    fit_values2_reduced(na-1) = fit_ss; %(1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE') ) * 100;
end

% Plotting AIC, FPE, and Fit Percentage

% Plot AIC values
figure;
subplot(3,1,1);
plot(na_range, aic_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
title('AIC Values ', 'FontSize', 12);
xlabel('Order (na)', 'FontSize', 10);
ylabel('AIC', 'FontSize', 10);
grid on;

% Plot FPE values
subplot(3,1,2);
plot(na_range, fpe_values, '-o', 'LineWidth', 2, 'MarkerSize', 6);
title('FPE Values ', 'FontSize', 12);
xlabel('Order (na)', 'FontSize', 10);
ylabel('FPE', 'FontSize', 10);
grid on;

% Plot Fit percentage values
subplot(3,1,3);
plot(na_range, fit_values2, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Original Model');
hold on;
plot(na_range, fit_values2_reduced, '-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reduced Model');
title('Fit Percentage ', 'FontSize', 12);
xlabel('Order (na)', 'FontSize', 10);
ylabel('Fit Percentage', 'FontSize', 10);
legend('show', 'FontSize', 10);
grid on;

% Adjust layout for better readability
sgtitle('Model Evaluation ', 'FontSize', 14);



%% check the results  GN

opt = bjOptions;
opt.SearchMethod = 'gn';
model = bj(data1, [3 2 2 3 1],opt);

% Validate on data2
y_valid = data2.OutputData;  
y_pred_original = predict(model, data2);  
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE') ) * 100;
[~,fit_ss,~] = compare(data2, model);
disp(['Fit percentage for the original model: ', num2str(fit_ss), '%']);

% Reduce the model
reduced_model = reduce(model, 2);
y_pred_reduced = predict(reduced_model, data2);  
%fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE') ) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the reduced model: ', num2str(fit_ss), '%']);

% Residuals for both models
residuals_original = y_valid - y_pred_original.OutputData;
residuals_reduced = y_valid - y_pred_reduced.OutputData;

disp(sum(residuals_original.^2)/Nt)
p = 10 ; 
RSS_original = sum(residuals_original.^2); 
aicc = Nt * log(RSS_original / Nt) + 2 * p;
fpee = (RSS_original / Nt) / (1 - p / Nt)^2;
disp('aic')
disp(aicc)
disp('fpe')
disp(fpee)


% Residual correlation plots
figure(5);
subplot(3, 1, 1);
autocorr(residuals_original, 'NumLags', 40);
title('Residual Autocorrelation (Original Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_original, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_original, y_pred_original.OutputData, 40);
title('Cross-correlation of Residuals with Output (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

figure(6);
subplot(3, 1, 1);
autocorr(residuals_reduced, 'NumLags', 40);
title('Residual Autocorrelation (Reduced Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_reduced, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_reduced, y_pred_reduced.OutputData, 40);
title('Cross-correlation of Residuals with Output (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

% Zero-Pole plot
figure(7);
hold on;
pzmap(model);
pzmap(reduced_model);
legend('Original Model', 'Reduced Model');
title('Zero-Pole Plot: Original vs Reduced Model');
grid on;


figure;
subplot(2,1,1);
resid(data2, model);
subplot(2,1,2);
resid(data2, reduced_model);
figure(9);
compare(data2, model);


%%           
%%%%%%%%%%%%%  RPLR -----------------------------------------------
na_range = 2:4;
nk = 1; 
lambda = 0.98;
AIC_values = zeros(1, size(na_range,2)); 
FPE_values = zeros(1, size(na_range,2)); 
fit_values = zeros(1, size(na_range,2)); 
fit_values_reduced = zeros(1, size(na_range,2));

for na = na_range
    nb = 3;  
    nc = na;  
    nd = 3;  
    nf = na; 

    [theta, y_hat, R, e] = rplr(data1, [0, nb, nc, nd, nf, nk], 'ff', lambda);

    final_theta = theta(end, :);
    B = [zeros(1, nk), final_theta(1:nb)]; 
    C = [1, final_theta(nb+1:nb+nc)];      
    D = [1, final_theta(nb+nc+1:nb+nc+nd)]; 
    F = [1, final_theta(nb+nc+nd+1:nb+nc+nd+nf)]; 
    % 
    % disp(['For na = ', num2str(na)]);
    % disp('Estimated ARMAX Model Parameters:');
    % disp('B:'); disp(B');
    % disp('C:'); disp(C');

    model = idpoly([], B, C, D, F); 
    model.Ts = data1.Ts; 

    y_valid = data2.OutputData; 
    y_pred_original = predict(model, data2); 
    %fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, model);
    fit_values(na - 1) = fit_ss; %fit_value_original;  

    residuals_original = y_valid - y_pred_original.OutputData;
    N = length(y_valid); 
    p = na + nb + nc; 
    RSS_original = sum(residuals_original.^2); 
    AIC_original = N * log(RSS_original / N) + 2 * p;
    FPE_original = (RSS_original / N) / (1 - p / N)^2;

    AIC_values(na - 1) = AIC_original;
    FPE_values(na - 1) = FPE_original;

    % disp(['Fit percentage for the original model: ', num2str(fit_value_original), '%']);
    % disp(['AIC for the original model: ', num2str(AIC_original)]);
    % disp(['FPE for the original model: ', num2str(FPE_original)]);
    
    if na == 2
        reduced_model = reduce(model, 2);
    else
        reduced_model = reduce(model, 3);
    end
    y_pred_reduced = predict(reduced_model, data2); 
    %fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, reduced_model);
    fit_values_redused(na - 1) = fit_ss; %fit_value_reduced;
    %disp(['Fit percentage for the reduced model: ', num2str(fit_value_reduced), '%']);
end


figure;
subplot(3, 1, 1);
plot(na_range, AIC_values, '-o');
title('AIC');
xlabel('na');
ylabel('AIC');

subplot(3, 1, 2);
plot(na_range, FPE_values, '-o');
title('FPE');
xlabel('na');
ylabel('FPE');

subplot(3, 1, 3);
plot(na_range, fit_values, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Original Model');
hold on;
plot(na_range, fit_values_reduced, '-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reduced Model');
title('Fit Percentage ', 'FontSize', 12);
xlabel('Order (na)', 'FontSize', 10);
ylabel('Fit Percentage', 'FontSize', 10);
legend('show', 'FontSize', 10);
grid on;




%%
%%%%%%%  check results for RPLR ---------------------------


nb = 3;
nc = 3;
nd = 3;
nf = 3;
nk = 1; 

lambda = 0.98; 
[theta, y_hat, R, e] = rplr(data1, [0, nb, nc, nd, nf, nk], 'ff', lambda);

final_theta = theta(end, :);
B = [zeros(1, nk), final_theta(1:nb)]; 
C = [1, final_theta(nb+1:nb+nc)];      
D = [1, final_theta(nb+nc+1:nb+nc+nd)]; 
F = [1, final_theta(nb+nc+nd+1:nb+nc+nd+nf)]; 

model = idpoly([], B, C, D, F); 

disp('Estimated ARMAX Model Parameters:');
disp('B:'); disp(B');
disp('C:'); disp(C');
disp('D:'); disp(D');
disp('F:'); disp(F');

model.Ts = data1.Ts; 

y_valid = data2.OutputData; 
y_pred_original = predict(model, data2); 
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the original model: ', num2str(fit_ss), '%']);

residuals_original = y_valid - y_pred_original.OutputData;

N = length(y_valid); 
p = na + nb + nc; 
RSS_original = sum(residuals_original.^2); 
AIC_original = N * log(RSS_original / N) + 2 * p;
FPE_original = (RSS_original / N) / (1 - p / N)^2;
disp(['AIC for the original model: ', num2str(AIC_original)]);
disp(['FPE for the original model: ', num2str(FPE_original)]);

reduced_model = reduce(model, 3); 
y_pred_reduced = predict(reduced_model, data2); 
%fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the reduced model: ', num2str(fit_ss), '%']);

residuals_reduced = y_valid - y_pred_reduced.OutputData;

% Plot residual correlations and cross-correlations
figure(1);
subplot(3, 1, 1);
autocorr(residuals_original, 'NumLags', 40);
title('Residual Autocorrelation (Original Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_original, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_original, y_pred_original.OutputData, 40);
title('Cross-correlation of Residuals with Output (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

figure(2);
subplot(3, 1, 1);
autocorr(residuals_reduced, 'NumLags', 40);
title('Residual Autocorrelation (Reduced Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_reduced, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_reduced, y_pred_reduced.OutputData, 40);
title('Cross-correlation of Residuals with Output (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

% Plot Zero-Pole Map for both models
figure(3);
hold on;
pzmap(model);
pzmap(reduced_model);
legend('Original Model', 'Reduced Model');
title('Zero-Pole Plot: Original vs Reduced Model');
grid on;

% Summarize the results
disp('Summary of Results:');
disp(['Original Model AIC: ', num2str(AIC_original)]);
disp(['Original Model FPE: ', num2str(FPE_original)]);

% Plot each theta parameter over time (iterations)
% Plot the 'na' group (first 3 parameters)
figure(4);
for i = 1:na
    subplot(na, 1, i);
    plot(theta(:, i)); % Plot the i-th theta parameter over time
    title(['Theta for A ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(i), '}']);
end

% Plot the 'nb' group (next 3 parameters)
figure(5);
for i = 1:nb
    subplot(nb, 1, i);
    plot(theta(:, na + i)); % Plot the (na + i)-th theta parameter over time
    title(['Theta for B ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(na + i), '}']);
end

% Plot the 'nc' group (next 3 parameters)
figure(6);
for i = 1:nc
    subplot(nc, 1, i);
    plot(theta(:, na + nb + i)); % Plot the (na + nb + i)-th theta parameter over time
    title(['Theta for C  ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(na + nb + i), '}']);
end


%%
        
%%%%%%%%%%%%%  RPEM -----------------------------------------------
na_range = 2:3;
nk = 1; 
lambda = 0.98;
AIC_values = zeros(1, size(na_range,2)); 
FPE_values = zeros(1, size(na_range,2)); 
fit_values = zeros(1, size(na_range,2)); 
fit_values_reduced = zeros(1, size(na_range,2));

for na = na_range
    nb = 2;  
    nc = na;  
    nd = 2;  
    nf = na; 

    [theta, y_hat, R, e] = rpem(data1, [0, nb, nc, nd, nf, nk], 'ff', lambda);

    final_theta = theta(end, :);
    B = [zeros(1, nk), final_theta(1:nb)]; 
    C = [1, final_theta(nb+1:nb+nc)];      
    D = [1, final_theta(nb+nc+1:nb+nc+nd)]; 
    F = [1, final_theta(nb+nc+nd+1:nb+nc+nd+nf)]; 
    % 
    % disp(['For na = ', num2str(na)]);
    % disp('Estimated ARMAX Model Parameters:');
    % disp('B:'); disp(B');
    % disp('C:'); disp(C');

    model = idpoly([], B, C, D, F); 
    model.Ts = data1.Ts; 

    y_valid = data2.OutputData; 
    y_pred_original = predict(model, data2); 
    %fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, model);
    fit_values(na - 1) = fit_ss; %fit_value_original;  

    residuals_original = y_valid - y_pred_original.OutputData;
    N = length(y_valid); 
    p = na + nb + nc; 
    RSS_original = sum(residuals_original.^2); 
    AIC_original = N * log(RSS_original / N) + 2 * p;
    FPE_original = (RSS_original / N) / (1 - p / N)^2;

    AIC_values(na - 1) = AIC_original;
    FPE_values(na - 1) = FPE_original;

    % disp(['Fit percentage for the original model: ', num2str(fit_value_original), '%']);
    % disp(['AIC for the original model: ', num2str(AIC_original)]);
    % disp(['FPE for the original model: ', num2str(FPE_original)]);
    
    if na == 2
        reduced_model = reduce(model, 2);
    else
        reduced_model = reduce(model, 3);
    end
    y_pred_reduced = predict(reduced_model, data2); 
    %fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, reduced_model);
    fit_values_redused(na - 1) = fit_ss; %fit_value_reduced;
    %disp(['Fit percentage for the reduced model: ', num2str(fit_value_reduced), '%']);
end


figure;
subplot(3, 1, 1);
plot(na_range, AIC_values, '-o');
title('AIC');
xlabel('na');
ylabel('AIC');

subplot(3, 1, 2);
plot(na_range, FPE_values, '-o');
title('FPE');
xlabel('na');
ylabel('FPE');

subplot(3, 1, 3);
plot(na_range, fit_values, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Original Model');
hold on;
plot(na_range, fit_values_reduced, '-x', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'Reduced Model');
title('Fit Percentage ', 'FontSize', 12);
xlabel('Order (na)', 'FontSize', 10);
ylabel('Fit Percentage', 'FontSize', 10);
legend('show', 'FontSize', 10);
grid on;




%%
%%%%%%%  check results for RPLR ---------------------------


nb = 3;
nc = 3;
nd = 3;
nf = 3;
nk = 1; 

lambda = 0.98; 
[theta, y_hat, R, e] = rpem(data1, [0, nb, nc, nd, nf, nk], 'ff', lambda);

final_theta = theta(end, :);
B = [zeros(1, nk), final_theta(1:nb)]; 
C = [1, final_theta(nb+1:nb+nc)];      
D = [1, final_theta(nb+nc+1:nb+nc+nd)]; 
F = [1, final_theta(nb+nc+nd+1:nb+nc+nd+nf)]; 

model = idpoly([], B, C, D, F); 

disp('Estimated ARMAX Model Parameters:');
disp('B:'); disp(B');
disp('C:'); disp(C');
disp('D:'); disp(D');
disp('F:'); disp(F');

model.Ts = data1.Ts; 

y_valid = data2.OutputData; 
y_pred_original = predict(model, data2); 
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the original model: ', num2str(fit_ss), '%']);

residuals_original = y_valid - y_pred_original.OutputData;

N = length(y_valid); 
p = na + nb + nc; 
RSS_original = sum(residuals_original.^2); 
AIC_original = N * log(RSS_original / N) + 2 * p;
FPE_original = (RSS_original / N) / (1 - p / N)^2;
disp(['AIC for the original model: ', num2str(AIC_original)]);
disp(['FPE for the original model: ', num2str(FPE_original)]);

reduced_model = reduce(model, 3); 
y_pred_reduced = predict(reduced_model, data2); 
%fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the reduced model: ', num2str(fit_ss), '%']);

residuals_reduced = y_valid - y_pred_reduced.OutputData;

% Plot residual correlations and cross-correlations
figure(1);
subplot(3, 1, 1);
autocorr(residuals_original, 'NumLags', 40);
title('Residual Autocorrelation (Original Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_original, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_original, y_pred_original.OutputData, 40);
title('Cross-correlation of Residuals with Output (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

figure(2);
subplot(3, 1, 1);
autocorr(residuals_reduced, 'NumLags', 40);
title('Residual Autocorrelation (Reduced Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_reduced, data2.InputData, 40);
title('Cross-correlation of Residuals with Input (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_reduced, y_pred_reduced.OutputData, 40);
title('Cross-correlation of Residuals with Output (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

% Plot Zero-Pole Map for both models
figure(3);
hold on;
pzmap(model);
pzmap(reduced_model);
legend('Original Model', 'Reduced Model');
title('Zero-Pole Plot: Original vs Reduced Model');
grid on;

% Summarize the results
disp('Summary of Results:');
disp(['Original Model AIC: ', num2str(AIC_original)]);
disp(['Original Model FPE: ', num2str(FPE_original)]);

% Plot each theta parameter over time (iterations)
% Plot the 'na' group (first 3 parameters)
figure(4);
for i = 1:na
    subplot(na, 1, i);
    plot(theta(:, i)); % Plot the i-th theta parameter over time
    title(['Theta for A ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(i), '}']);
end

% Plot the 'nb' group (next 3 parameters)
figure(5);
for i = 1:nb
    subplot(nb, 1, i);
    plot(theta(:, na + i)); % Plot the (na + i)-th theta parameter over time
    title(['Theta for B ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(na + i), '}']);
end

% Plot the 'nc' group (next 3 parameters)
figure(6);
for i = 1:nc
    subplot(nc, 1, i);
    plot(theta(:, na + nb + i)); % Plot the (na + nb + i)-th theta parameter over time
    title(['Theta for C  ', num2str(i)]);
    xlabel('Iteration');
    ylabel(['\theta_{', num2str(na + nb + i), '}']);
end

