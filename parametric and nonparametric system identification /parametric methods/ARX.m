% Load and preprocess data
data = load('D:\data2.mat'); adjust this
ugt = detrend(data.u1);  
ygt = detrend(data.y1);  
upt = detrend(data.u2);  
ypt = detrend(data.y2);  
Nt = length(ugt);

delay = 1;  
data1 = iddata(ygt, ugt, 1);  
data2 = iddata(ypt, upt, 1);  

na_range = 1:10; 
nb_range = 1:10;  
nk = delay;      

orders = combvec(na_range, nb_range)';
aic_values = zeros(length(na_range), length(nb_range));
fpe_values = zeros(length(na_range), length(nb_range));

aic_values1 = zeros(length(na_range), length(nb_range));
fpe_values1 = zeros(length(na_range), length(nb_range));
fit_values2 = zeros(length(na_range), length(nb_range)); 

aic_values_reduced = zeros(length(na_range), length(nb_range));
fpe_values_reduced = zeros(length(na_range), length(nb_range));
fit_values2_reduced = zeros(length(na_range), length(nb_range));  

%%%%% LS -------------------------------------
for na = na_range
    for nb = nb_range

        if nb > na 
            aic_values1(na, nb) = 0 ;
            fpe_values1(na, nb) = 0 ;
            fit_values2(na, nb) = 0;
            fit_values2_reduced(na, nb) = 0;
            continue;
        end

        model = arx(data1, [na nb nk]);

        %  final_theta = LS_ARX(ygt,ugt, na ,nb , nk);
        % %final_theta = theta(end, :);
        % A = [1, final_theta(1:na)'];
        % B = [zeros(1, nk), final_theta(na+1:na+nb)'];

        y_valid = data2.OutputData;
        y_pred = predict(model, data2);
        [~,fit_ss,~] = compare(data2, model);
        fit_values2(na, nb) = fit_ss; %(1 - goodnessOfFit(y_pred.OutputData, y_valid, 'NRMSE') ) * 100;
        
        residuals_original = y_valid - y_pred.OutputData;
        p = na + nb ; 
        RSS_original = sum(residuals_original.^2); 
        aic_values1(na, nb) = Nt * log(RSS_original / Nt) + 2 * p;
        fpe_values1(na, nb) = (RSS_original / Nt) / (1 - p / Nt)^2;
        % aic_values1(na, nb) = aic(model);
        % fpe_values1(na, nb) = fpe(model);

        reduced_model = reduce(model, 2);   
        y_pred_reduced = predict(reduced_model, data2);
        [~,fit_ss,~] = compare(data2, reduced_model);
        fit_values2_reduced(na, nb) = fit_ss; %(1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE') ) * 100;
        
    end
end

% Heatmaps for Gaussian input (data1) - Original Models
figure(1);
% subplot(3, 2, 1);
heatmap(na_range(2:10), nb_range(2:10), aic_values1((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('AIC Heatmap (LS)');
xlabel('na');
ylabel('nb');

figure(2);
% subplot(3, 2, 2);
heatmap(na_range(2:10), nb_range(2:10), fpe_values1((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('FPE Heatmap (LS)');
xlabel('na');
ylabel('nb');

figure(3);
% subplot(3, 2, 3);
heatmap(na_range(2:10), nb_range(2:10), fit_values2((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('Fit % Heatmap (Validation)');
xlabel('na');
ylabel('nb');

% Heatmaps for reduced models
% subplot(3, 2, 4);
% heatmap(na_range, nb_range, aic_values_reduced', 'Colormap', jet, 'ColorbarVisible', 'on');
% title('AIC Heatmap (Reduced Model - Gaussian Input)');
% xlabel('na');
% ylabel('nb');
% 
% subplot(3, 2, 5);
% heatmap(na_range, nb_range, fpe_values_reduced', 'Colormap', jet, 'ColorbarVisible', 'on');
% title('FPE Heatmap (Reduced Model - Gaussian Input)');
% xlabel('na');
% ylabel('nb');

figure(4);
% subplot(3, 2, 6);
heatmap(na_range(2:10), nb_range(2:10), fit_values2_reduced((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('Fit % Heatmap (Reduced Model Validation (LS) )');
xlabel('na');
ylabel('nb');


%% check the results  LS

% Fit ARX model
model = arx(data1, [10 10 1]);

% Validate on data2
y_valid = data2.OutputData;  
y_pred_original = predict(model, data2); 
[~,fit_ss,~] = compare(data2, model);
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE') ) * 100;
disp(['Fit percentage for the original model: ', num2str(fit_ss), '%']);

figure;
plot(y_valid , 'k');
hold on;
plot(y_pred_original.OutputData , 'b');
hold off;
grid on;
legend();

% Reduce the model
reduced_model = reduce(model,2);
y_pred_reduced = predict(reduced_model, data2);  
[~,fit_ss,~] = compare(data2, reduced_model);
%fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE') ) * 100;
disp(['Fit percentage for the reduced model: ', num2str(fit_ss), '%']);

% Residuals for both models
residuals_original = y_valid - y_pred_original.OutputData;
residuals_reduced = y_valid - y_pred_reduced.OutputData;

p = 20 ; 
RSS_original = sum(residuals_reduced.^2); 
aicc = Nt * log(RSS_original / Nt) + 2 * p;
fpee = (RSS_original / Nt) / (1 - p / Nt)^2;
disp('aic')
disp(aicc)
disp('fpe')
disp(fpee)

p = 3 ; 
RSS_original = sum(residuals_reduced.^2); 
aicc = Nt * log(RSS_original / Nt) + 2 * p;
fpee = (RSS_original / Nt) / (1 - p / Nt)^2;
disp('aic')
disp(aicc)
disp('fpe')
disp(fpee)

% Residual correlation plots
figure(5);
subplot(3, 1, 1);
autocorr(residuals_original, 'NumLags', 100);
title('Residual Autocorrelation (Original Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_original, data2.InputData, 100);
title('Cross-correlation of Residuals with Input (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_original, y_pred_original.OutputData, 100);
title('Cross-correlation of Residuals with Output (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

figure(6);
subplot(3, 1, 1);
autocorr(residuals_reduced, 'NumLags', 100);
title('Residual Autocorrelation (Reduced Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_reduced, data2.InputData, 100);
title('Cross-correlation of Residuals with Input (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_reduced, y_pred_reduced.OutputData, 100);
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

% figure;
% compare(data1,model);

figure;
compare(data2,model);



%%  IV ------------------------------------------

na_range = 1:10;
nb_range = 1:10;

orders = combvec(na_range, nb_range)';
aic_values = zeros(length(na_range), length(nb_range));
fpe_values = zeros(length(na_range), length(nb_range));

aic_values1 = zeros(length(na_range), length(nb_range));
fpe_values1 = zeros(length(na_range), length(nb_range));
fit_values2 = zeros(length(na_range), length(nb_range)); 

aic_values_reduced = zeros(length(na_range), length(nb_range));
fpe_values_reduced = zeros(length(na_range), length(nb_range));
fit_values2_reduced = zeros(length(na_range), length(nb_range));  

%%%%% IV -------------------------------------

for na = na_range
    for nb = nb_range
        if nb > na 
            aic_values1(na, nb) = 0 ;
            fpe_values1(na, nb) = 0 ;
            fit_values2(na, nb) = 0;
            fit_values2_reduced(na, nb) = 0;
            continue;
        end
        model = iv4(data1, [na nb nk]);
        % aic_values1(na, nb) = aic(model);
        % fpe_values1(na, nb) = fpe(model);

        y_valid = data2.OutputData;
        y_pred = predict(model, data2);
        [~,fit_ss,~] = compare(data2, model);
        if isnan(fit_ss) || fit_ss > 100 || fit_ss < -100
            fit_values2(na, nb) = 0;
        else
            fit_values2(na, nb) = fit_ss; %(1 - goodnessOfFit(y_pred.OutputData, y_valid, 'NRMSE')) * 100;
        end

        residuals_original = y_valid - y_pred.OutputData;
        p = na + nb ; 
        RSS_original = sum(residuals_original.^2); 
        aic_values1(na, nb) = Nt * log(RSS_original / Nt) + 2 * p;
        fpe_values1(na, nb) = (RSS_original / Nt) / (1 - p / Nt)^2;

        reduced_model = reduce(model, 2);
        y_pred_reduced = predict(reduced_model, data2);
        [~,fit_ss,~] = compare(data2, reduced_model);
        if isnan(fit_ss) || fit_ss > 100 || fit_ss < -100
            fit_values2_reduced(na, nb) = 0;
        else
            fit_values2_reduced(na, nb) = fit_ss; %(1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE') ) * 100;
        end
    end
end

% Heatmaps for Gaussian input (data1) - Original Models
figure(8);
% subplot(3, 2, 1);
heatmap(na_range(2:10), nb_range(2:10), aic_values1((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('AIC Heatmap (IV)');
xlabel('na');
ylabel('nb');

figure(9);
% subplot(3, 2, 2);
heatmap(na_range(2:10), nb_range(2:10), fpe_values1((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('FPE Heatmap (IV)');
xlabel('na');
ylabel('nb');

figure(10);
% subplot(3, 2, 3);
heatmap(na_range(2:10), nb_range(2:10), fit_values2((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('Fit % Heatmap (Validation)');
xlabel('na');
ylabel('nb');

% Heatmaps for reduced models
% subplot(3, 2, 4);
% heatmap(na_range, nb_range, aic_values_reduced', 'Colormap', jet, 'ColorbarVisible', 'on');
% title('AIC Heatmap (Reduced Model - Gaussian Input)');
% xlabel('na');
% ylabel('nb');
% 
% subplot(3, 2, 5);
% heatmap(na_range, nb_range, fpe_values_reduced', 'Colormap', jet, 'ColorbarVisible', 'on');
% title('FPE Heatmap (Reduced Model - Gaussian Input)');
% xlabel('na');
% ylabel('nb');

figure(11);
% subplot(3, 2, 6);
heatmap(na_range(2:10), nb_range(2:10), fit_values2_reduced((2:10),(2:10))', 'Colormap', jet, 'ColorbarVisible', 'on');
title('Fit % Heatmap (Reduced Model Validation (IV) )');
xlabel('na');
ylabel('nb');


%% check the results  IV

% Fit ARX model
model = iv4(data1, [7 7 1]);

% Validate on data2
y_valid = data2.OutputData;  
y_pred_original = predict(model, data2);  
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, model);
disp(['Fit percentage for the original model: ', num2str(fit_ss), '%']);

figure;
plot( data2.OutputData );
hold on;
plot( y_pred_original.OutputData);
hold off;
title('OUT');
xlabel('Lag');
ylabel('Cross-correlation');
legend('real' ,'predict');

% Reduce the model
reduced_model = reduce(model,2);
y_pred_reduced = predict(reduced_model, data2);  
%fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, reduced_model);
disp(['Fit percentage for the reduced model: ', num2str(fit_ss), '%']);

% Residuals for both models
residuals_original = y_valid - y_pred_original.OutputData;
residuals_reduced = y_valid - y_pred_reduced.OutputData;

disp(sum(residuals_original.^2)/Nt)
p = 14 ; 
RSS_original = sum(residuals_original.^2); 
aicc = Nt * log(RSS_original / Nt) + 2 * p;
fpee = (RSS_original / Nt) / (1 - p / Nt)^2;
disp('aic')
disp(aicc)
disp('fpe')
disp(fpee)

MM = 40;
% Residual correlation plots
figure(5);
subplot(3, 1, 1);
autocorr(residuals_original, 'NumLags', MM);
title('Residual Autocorrelation (Original Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_original, data2.InputData, MM);
title('Cross-correlation of Residuals with Input (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_original, y_pred_original.OutputData, MM);
title('Cross-correlation of Residuals with Output (Original Model)');
xlabel('Lag');
ylabel('Cross-correlation');

figure(6);
subplot(3, 1, 1);
autocorr(residuals_reduced, 'NumLags', MM);
title('Residual Autocorrelation (Reduced Model)');
xlabel('Lag');
ylabel('Autocorrelation');

subplot(3, 1, 2);
crosscorr(residuals_reduced, data2.InputData, MM);
title('Cross-correlation of Residuals with Input (Reduced Model)');
xlabel('Lag');
ylabel('Cross-correlation');

subplot(3, 1, 3);
crosscorr(residuals_reduced, y_pred_reduced.OutputData, MM);
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
compare(data2,model);

figure;
subplot(2,1,1);
resid(data2, model);
subplot(2,1,2);
resid(data2, reduced_model);

%%
%%%% RLS ---------------------------------------------
na_range = 2:10;
nk = 1; 
lambda = 0.97;
AIC_values = zeros(1, size(na_range,2)); 
FPE_values = zeros(1, size(na_range,2)); 
fit_values = zeros(1, size(na_range,2)); 
fit_values_reduced = zeros(1, size(na_range,2));

for na = na_range
    nb = na;  
    nc = na;  

    % ARMAX Model Estimation
    [theta, y_hat, R, e] = rpem(data1, [na, nb, 0, 0, 0, nk], 'ff', lambda);

    final_theta = theta(end, :);
    A = [1, final_theta(1:na)];
    B = [zeros(1, nk), final_theta(na+1:na+nb)];
    %C = [1, final_theta(na+nb+1:na+nb+nc)];
    % 
    % disp(['For na = ', num2str(na)]);
    % disp('Estimated ARX Model Parameters:');
    % disp('A:'); disp(A');
    % disp('B:'); disp(B');
    % disp('C:'); disp(C');

    model = idpoly(A, B, [], [], []); 
    model.Ts = data1.Ts; 

    y_valid = data2.OutputData; 
    y_pred_original = predict(model, data2); 
    %fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;

    [~,fit_ss,~] = compare(data2, model);
    fit_values(na - 1) = fit_ss;  

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
    

    reduced_model = reduce(model,2);
    y_pred_reduced = predict(reduced_model, data2); 
    %fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, reduced_model);
    fit_values_reduced(na - 1) = fit_ss;
    %disp(['Fit percentage for the reduced model: ', num2str(fit_value_reduced), '%']);
end


figure;
subplot(3, 1, 1);
plot(na_range, AIC_values, '-o');
title('AIC');
xlabel('na');
ylabel('AIC');
grid on;

subplot(3, 1, 2);
plot(na_range, FPE_values, '-o');
title('FPE');
xlabel('na');
ylabel('FPE');
grid on;

subplot(3, 1, 3);
plot(na_range, fit_values, '-o', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', 'Original Model');
hold on;
plot(na_range, fit_values_reduced, '-x', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', 'Reduced Model');
title('Fit Percentage ');
xlabel('Order (na)');
ylabel('Fit Percentage');
legend('show');
grid on;



%%
%%%%%%%  check results for RLS ---------------------------

na = 7;
nb = 7;
nc = 2;
nk = 1; 

lambda = 0.99; 
[theta, y_hat, R, e] = rpem(data1, [na, nb, 0, 0, 0, nk], 'ff', lambda);

final_theta = theta(end, :);
A = [1, final_theta(1:na)];
B = [zeros(1, nk), final_theta(na+1:na+nb)];
%C = [1, final_theta(na+nb+1:na+nb+nc)];

% disp('Estimated ARMAX Model Parameters:');
% disp('A:'); disp(A');
% disp('B:'); disp(B');
% disp('C:'); disp(C');

model = idpoly(A, B, [], [], []);
model.Ts = data1.Ts; 

y_valid = data2.OutputData; 
y_pred_original = predict(model, data2); 
%fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
[~,fit_ss,~] = compare(data2, model);
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

% % Summarize the results
% disp('Summary of Results:');
% disp(['Original Model AIC: ', num2str(AIC_original)]);
% disp(['Original Model FPE: ', num2str(FPE_original)]);
% disp(['Original Model Fit (%): ', num2str(fit_value_original), '%']);
% disp(['Reduced Model Fit (%): ', num2str(fit_value_reduced), '%']);

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

% % Plot the 'nc' group (next 3 parameters)
% figure(6);
% for i = 1:nc
%     subplot(nc, 1, i);
%     plot(theta(:, na + nb + i)); % Plot the (na + nb + i)-th theta parameter over time
%     title(['Theta for C  ', num2str(i)]);
%     xlabel('Iteration');
%     ylabel(['\theta_{', num2str(na + nb + i), '}']);
% end

figure;
compare(data2,model);

figure;
resid(data2, model);

figure;
resid(data2, reduced_model);

%%
%%%% RIV ---------------------------------------------
na_range = 2:10;
nk = 1; 
lambda = 0.99;
AIC_values = zeros(1, size(na_range,2)); 
FPE_values = zeros(1, size(na_range,2)); 
fit_values = zeros(1, size(na_range,2)); 
fit_values_reduced = zeros(1, size(na_range,2));

for na = na_range
    nb = na;  
    nc = na;  
    [theta, y_hat] = RIV_ARX(ygt, ugt, na, nb, nk, lambda );
    %[theta, y_hat, R, e] = rpem(data1, [0, nb, 0, 0, na, nk], 'ff', lambda);
    final_theta = theta(end,:);
    B = [zeros(1, nk), final_theta(1:nb)];
    A = [1, final_theta(nb+1:na+nb)];
    %C = [1, final_theta(na+nb+1:na+nb+nc)];
    % 
    % disp(['For na = ', num2str(na)]);
    % disp('Estimated ARX Model Parameters:');
    % disp('A:'); disp(A');
    % disp('B:'); disp(B');
    % disp('C:'); disp(C');

    model = idpoly(A, B, [], [], []); 
    model.Ts = data1.Ts; 

    y_valid = data2.OutputData; 
    y_pred_original = predict(model, data2); 
    %fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, model);
    fit_values(na - 1) = fit_ss; %fit_value_original;  

    residuals_original = y_valid - y_pred_original.OutputData;
    N = length(y_valid); 
    p = nb + nc; 
    RSS_original = sum(residuals_original.^2); 
    AIC_original = Nt * log(RSS_original / Nt) + 2 * p;
    FPE_original = (RSS_original / Nt) / (1 - p / Nt)^2;

    AIC_values(na - 1) = AIC_original;
    FPE_values(na - 1) = FPE_original;

    % disp(['Fit percentage for the original model: ', num2str(fit_value_original), '%']);
    % disp(['AIC for the original model: ', num2str(AIC_original)]);
    % disp(['FPE for the original model: ', num2str(FPE_original)]);
    

    reduced_model = reduce(model,2);
    y_pred_reduced = predict(reduced_model, data2); 
    %fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
    [~,fit_ss,~] = compare(data2, reduced_model);
    fit_values_reduced(na - 1) = fit_ss; %fit_value_reduced;
    %disp(['Fit percentage for the reduced model: ', num2str(fit_value_reduced), '%']);
end


figure;
subplot(3, 1, 1);
plot(na_range, AIC_values, '-o');
title('AIC');
xlabel('na');
ylabel('AIC');
grid on;

subplot(3, 1, 2);
plot(na_range, FPE_values, '-o');
title('FPE');
xlabel('na');
ylabel('FPE');
grid on;

subplot(3, 1, 3);
plot(na_range, fit_values, '-o', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', 'Original Model');
hold on;
plot(na_range, fit_values_reduced, '-x', 'LineWidth', 1, 'MarkerSize', 4, 'DisplayName', 'Reduced Model');
title('Fit Percentage ');
xlabel('Order (na)');
ylabel('Fit Percentage');
legend('show');
grid on;




%%
%%%%%%%  check results for RIV ---------------------------

na = 2;
nb = 2;
nc = 2;
nk = 1; 

lambda = 0.99; 
[theta, y_hat, R, e] = rpem(data1, [na, nb, nc, 0, 0, nk], 'ff', lambda);

final_theta = theta(end, :);
A = [1, final_theta(1:na)];
B = [zeros(1, nk), final_theta(na+1:na+nb)];
C = [1, final_theta(na+nb+1:na+nb+nc)];

disp('Estimated ARMAX Model Parameters:');
disp('A:'); disp(A');
disp('B:'); disp(B');
disp('C:'); disp(C');

model = idpoly(A, B, C, [], []);
model.Ts = data1.Ts; 

y_valid = data2.OutputData; 
y_pred_original = predict(model, data2); 
fit_value_original = (1 - goodnessOfFit(y_pred_original.OutputData, y_valid, 'NRMSE')) * 100;
disp(['Fit percentage for the original model: ', num2str(fit_value_original), '%']);

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
fit_value_reduced = (1 - goodnessOfFit(y_pred_reduced.OutputData, y_valid, 'NRMSE')) * 100;
disp(['Fit percentage for the reduced model: ', num2str(fit_value_reduced), '%']);

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
disp(['Original Model Fit (%): ', num2str(fit_value_original), '%']);
disp(['Reduced Model Fit (%): ', num2str(fit_value_reduced), '%']);

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

function [theta_all, y_hat] = RIV_ARX(y, u, na, nb, nk, lambda )
    % Inputs:
    % y - Output data vector
    % u - Input data vector
    % na - Order of A(q^-1)
    % nb - Order of B(q^-1)
    % nk - Delay (dead time)
    % lambda - Forgetting factor
    
    % Initialization
    N = length(y);               % Number of data points
    n_params = na + nb;          % Total number of parameters
    theta = zeros(n_params, 1);  % Initial parameter estimates
    theta_all = zeros(n_params, N); % Matrix to store all theta_t
    P = eye(n_params) * 1e6;     % Initial covariance matrix
    phi = zeros(n_params, 1);    % Regressor vector
    zeta = zeros(n_params, 1);   % Instrumental variable vector
    y_hat = zeros(N, 1);         % Predicted output

    % Main RIV loop
    for t = max(na+nk+1, nb + nk):N
        phi(1:nb) = u(t-nk:-1:t-nk-nb+1);
        phi(nb+1:end) = -y_hat(t-1:-1:t-na);

        zeta(1:nb) = u(t-nk:-1:t-nk-nb+1); 
        zeta(nb+1:end) = -y_hat(t-1-nk:-1:t-na-nk); 

        L = P * zeta / (lambda + phi' * P * zeta);
        
        % Update parameter estimates
        e = y(t) - phi' * theta; % Prediction error
        theta = theta + L * e;

        % Store current parameter estimates
        theta_all(:, t) = theta;

        % Update covariance matrix (P)
        P = (1 / lambda) * (P - L * phi' * P);
        
        % Store predicted output
        y_hat(t) = phi' * theta;
    end

    % Trim unused columns in theta_all
    theta_all = theta_all(:, max(na, nb + nk):N);
    theta_all = theta_all';
end

function [theta] = LS_ARX(y, u, na, nb, nk )

    N = length(y);               % Number of data points
    n_params = na + nb;          % Total number of parameters
    theta = zeros(n_params, 1);  % Initial parameter estimates
    %theta_all = zeros(n_params, N); % Matrix to store all theta_t
    %P = eye(n_params) * 1e6;     % Initial covariance matrix
    phi = zeros(n_params,1);
    phii = zeros(n_params, n_params);
    y_hat = zeros(N, 1);         
    

    for t = max(na ,nb)+1:N
        phi(1:na) = -y(t-1:-1:t-na);
        phi(na+1:end) = u(t-nk:-1:t-nb);
        phii = phii + phi*phi';
        F = F + phi*y(t);
    end
    phii = phii / N;
    F = F / N;

    theta = inv(phii)*F;
end

