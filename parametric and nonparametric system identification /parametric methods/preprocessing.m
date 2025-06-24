
% Load and preprocess data
data = load('D:\data2.mat');   %adjust this
ugt = detrend(data.u1);
ygt= detrend(data.y1);
upt= detrend(data.u2);
ypt = detrend(data.y2);
Nt = length(ugt);
%% step
gamma = 800;
g = SPA(ypt, upt, Nt, gamma, @hamm);

num_points = 512;  
g_padded = [g; zeros(num_points - length(g), 1)];
impulse_response = ifft(g_padded, 'symmetric');

step_response = cumsum(impulse_response);

figure;
plot(step_response);
title('Step Response (Non Parametric PRBS)');
xlabel('Time Steps');
ylabel('Amplitude');
grid on;



%%
%%%%%%%%%%  delay --------------------------------------------------
[ccfup, lagsup] = xcorr(upt, upt, 'biased');
% [ccfyup, lagsyup] = xcorr(ypt, upt, 'coeff');

[ccfug, lagsug] = xcorr(ugt, ugt, 'biased');
% [ccfyug, lagsyug] = xcorr(ygt, ugt, 'coeff');


% [~, maxIdxg] = max(ccfyug);
% estimated_lagg = lagsyug(maxIdxg);
% 
% [~, maxIdxp] = max(ccfyup);
% estimated_lagp = lagsyup(maxIdxp);


figure(1);

subplot(2, 1, 1);
% hold on;
plot(lagsug , ccfug, 'k', 'LineWidth', 1.5); % Gaussian autocorrelation
% plot(lagsup , ccfup, 'c', 'LineWidth', 1.5); % PRBS autocorrelation
xlabel('Lag');
ylabel('Autocorrelation');
title('Autocorrelation of Gaussian Input');
% legend('Gaussian Input');
grid on;
% hold off;

subplot(2, 1, 2);
% hold on;
plot(lagsup , ccfup, 'b', 'LineWidth', 1.5); % Gaussian cross-correlation
% plot(lagsyup , ccfyup, 'c', 'LineWidth', 1.5); % PRBS cross-correlation
xlabel('Lag ');
ylabel('Autocorrelation');
title('Autocorrelation of PRBS Input');
% legend('Gaussian Input', 'PRBS Input');
grid on;
% hold off;
% 
% % Display Delay Estimation
% [~, maxIdx1] = max(ccfyug);
% [~, maxIdx2] = max(ccfyup);
% estimated_delay1 = lagsyug(maxIdx1) ; % Delay for Gaussian
% estimated_delay2 = lagsyup(maxIdx2) ; % Delay for PRBS
% 
% fprintf('Estimated Delay for Gaussian Input: %.3f \n', estimated_delay1);
% fprintf('Estimated Delay for PRBS Input: %.3f \n', estimated_delay2);
% 

% N_values = [1024, 2048, 4096, 8192, 16384];
% M = 70;
% figure(10);
% mean_responsew = zeros(1,M);
% for idx = 1:length(N_values)
%     N = 4096;
%     num_groups = floor(Nt / N); 
%     impulse_responses = zeros(num_groups, M); 
% 
%     for group = 1:num_groups
%         u_group = ugt((group-1)*N + 1 : group*N);
%         y_group = ygt((group-1)*N + 1 : group*N);
%         impulse_responses(group, :) = corr(y_group, u_group, M, N);
%     end
% 
%     mean_responsew(1,:) = mean(impulse_responses, 1);
%     hold on;
%     for group = 1:num_groups
%         data_to_plot = impulse_responses(group, :);
%         h_individual = plot(1:length(data_to_plot), data_to_plot,'r.', 'MarkerSize', 5); 
%     end
%     h_mean = plot(mean_responsew(1,:),'k.', 'MarkerSize', 5 ); 
%     hold off;
%     grid on;
% 
%     %legend([h_individual, h_mean], {'Individual Responses', 'Mean Response'}, 'Location', 'northeast');
%     xlabel('Time Index');
%     ylabel('Impulse Response Coefficient');
%     title(sprintf('White Noise, N = %d', N));
% 
% end

figure(4);
subplot(2,1,1);
grid on;
data1 = iddata(ygt, ugt, 1);
hg = impulseest(data1);
index = 0:length(hg.Numerator)-1;
%impulseplot(hg);
stem(index ,hg.Numerator ,'filled', 'k','LineWidth',1);
xlim([0 69]);
xlabel('Index');
ylabel('Impulse Response Coefficient');
title('Impulse Responses for Gaussian Input');
subplot(2,1,2);
grid on;
datap = iddata(ypt, upt, 1);
hp = impulseest(datap);
%impulseplot(hp);
stem(index, hp.Numerator, 'r','filled','LineWidth',1);
xlim([0 69]);
xlabel('Index');
ylabel('Impulse Response Coefficient');
title('Impulse Responses for PRBS Input');

%%%%% Order -----------------------------------------------
% first way -------------------------------------

% [ccfyp, lagsyp] = xcorr(ypt, ypt, 'coeff');
% [ccfyg, lagsyg] = xcorr(ygt, ygt, 'coeff');
% 
% figure(2);
% subplot(2,1,1);
% plot(lagsyg, ccfyg);
% xlabel('Lag');
% ylabel('Output-Correlation');
% title('AutoCorrelation of Gaussian Outputs');
% grid on;
% subplot(2,1,2);
% plot(lagsyp, ccfyp);
% xlabel('Lag');
% ylabel('Output-Correlation');
% title('AutoCorrelation of PRBS Outputs');
% grid on;
% 
% [~, maxIdxg] = max(ccfyg);
% estimated_orderg = lagsyg(maxIdxg);
% 
% [~, maxIdxp] = max(ccfyp);
% estimated_orderp = lagsyp(maxIdxp);
% 
% fprintf('Estimated Order for Gaussian Input: %.3f \n', estimated_orderg);
% fprintf('Estimated Order for PRBS Input: %.3f \n', estimated_orderp);


%% Order Estimation

gamma = 800;
gp = RS(ypt, upt, Nt, gamma, @hamm); 
gg = RS(ygt, ugt, Nt, gamma, @hamm); 

% gg = SPA(ygt, ugt, Nt, gamma, @hamm); 
% gp = SPA(ypt, upt, Nt, gamma, @hamm); 

delta_omega = 2*pi / (2*gamma + 1);
var_gg = (1 / (2*pi)) * sum(abs(gg)) * delta_omega;
var_gp = (1 / (2*pi)) * sum(abs(gp)) * delta_omega;
fprintf('Variance of noise (Gaussian input): %.9f\n', var_gg);
fprintf('Variance of noise (PRBS input): %.9f\n', var_gp);


max_order = 10; % Maximum order to test
threshold = 1e-6; % Threshold for singularity
for s = 1:max_order
    R_s = zeros(2*s, 2*s);
    v = [eye(s),zeros(s,s);zeros(s,s),zeros(s,s)];
    Phi_s = [];
    for t = s+1:Nt
        phi_s = [-ygt(t-1:-1:t-s)', ugt(t-1:-1:t-s)'];
        R_s = R_s + (phi_s' * phi_s);
    end
    R_s = R_s / Nt;
    R_s = R_s- (var_gg)*v;
    % Check Rank (Singular if rank < size)
    if rank((R_s)^s) < size(R_s, 1)
        fprintf('Estimated system order: %d\n', s-1);
        break;
    end
end


freq = 2*pi*(0:2*gamma)/(2*gamma+1);
freq = freq(1:gamma+1);

figure(3);
subplot(2,1,1);
plot(freq, 20*log10(abs(gg)),'k', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title(sprintf('Noise Spectrum for Gaussian Input '));

subplot(2,1,2);
plot(freq, 20*log10(abs(gp)),'b', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title(sprintf('Noise Spectrum for PRBS Input '));

figure(4);
subplot(2,1,1);
plot(freq, 20*log10(abs(gg)),'k', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title(sprintf('Bode Diagram for Gaussian Input '));

subplot(2,1,2);
plot(freq, angle(gg)*180/pi,'k', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
%title(sprintf('Noise Spectrum for PRBS Input '));

figure(5);
subplot(2,1,1);
plot(freq, 20*log10(abs(gp)),'b', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title(sprintf('Bode Diagram for PRBS Input '));

subplot(2,1,2);
plot(freq, angle(gp)*180/pi,'b', 'LineWidth', 1); 
grid on;
set(gca, 'XScale', 'log'); 
%ylim([-80,20]);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
%title(sprintf('Noise Spectrum for PRBS Input '));

%%
%%%%%%% Energy -----------------------------------

R_gauss = xcorr(ugt, 'biased'); 
R_prbs = xcorr(upt, 'biased');   

S_gauss = abs(fft(R_gauss)); 
S_prbs = abs(fft(R_prbs));   

f =2*pi*(0:Nt-1) / (Nt); 

S_gauss_dB = 20 * log10(S_gauss ); 
S_prbs_dB = 20 * log10(S_prbs );

% Plot Spectrum of Gaussian Signal
figure;
subplot(2,1,1);
semilogx(f, S_gauss_dB(1:Nt), 'k', 'LineWidth', 1.5); % Plot positive frequencies
xlabel('\omega (rad/s)'); ylabel('Magnitude (dB)');
title('Gaussian Signal');
grid on;

% Plot Spectrum of PRBS Signal
subplot(2,1,2);
semilogx(f, S_prbs_dB(1:Nt), 'b', 'LineWidth', 1.5);
xlabel('\omega (rad/s)'); ylabel('Magnitude (dB)');
title('PRBS Signal');
grid on;


%% order

data = iddata(ygt, ugt, 1); % output, input, and sample time Ts
orders = 1:10;
criteria = zeros(length(orders), 1);

for i = 1:length(orders)
    model = n4sid(data1, orders(i), 'Focus', 'prediction');
    criteria1(i) = aic(model);
    criteria2(i) = fpe(model);
end
figure;
subplot(2,1,1);
plot(orders, criteria1, '-o');
xlabel('Model Order');
ylabel('AIC ');
title('Order Selection');

subplot(2,1,2);
plot(orders, criteria2, '-o');
xlabel('Model Order');
ylabel('FPE ');
title('Order Selection');


%%

function w = cons(N, gamma, location)
    w = zeros(1, N);
    start_idx = max(1, location - gamma);
    end_idx = min(N, location + gamma);
    w(start_idx:end_idx) = 1;
end

function w = parz(N, gamma, location)
    w = zeros(1, N);
    start_idx = max(1, location - gamma);
    end_idx = min(N, location + gamma);
    for k = start_idx:end_idx
        dist = abs(k - location);
        if dist <= gamma / 2
            w(k) = 1 - 6 * (dist / gamma)^2 + 6 * (dist / gamma)^3;
        elseif dist <= gamma
            w(k) = 2 * (1 - dist / gamma)^3;
        end
    end
end

function w = hann(N, gamma, location)
    w = zeros(1, N);
    start_idx = max(1, location - gamma);
    end_idx = min(N, location + gamma);
    for k = start_idx:end_idx
        dist = abs(k - location);
        w(k) = 0.5 * (1 + cos(pi * dist / gamma));
    end
end

function w = bart(N, gamma, location)
    w = zeros(1, N);
    start_idx = max(1, location - gamma);
    end_idx = min(N, location + gamma);
    for k = start_idx:end_idx
        w(k) = 1 - abs(k - location) / gamma;
    end
end

function w = hamm(N, gamma, location)
    w = zeros(1, N);
    start_idx = max(1, location - gamma);
    end_idx = min(N, location + gamma);
    for k = start_idx:end_idx
        dist = abs(k - location);
        w(k) = 0.5 + 0.5 * cos(pi * dist / gamma);
    end
end



function rs = RS(y, u, N, gamma, window_func)
    y = y(1:N);
    u = u(1:N);
    Ryu = xcorr(y, u, N, 'biased'); 
    Ruu = xcorr(u, N, 'biased');
    Ryy = xcorr(y, N, 'biased');

    Ryu = Ryu'; 
    Ruu = Ruu';
    Ryy = Ryy';

    Ryu = Ryu(N-gamma:N+gamma);
    Ruu = Ruu(N-gamma:N+gamma);
    Ryy = Ryy(N-gamma:N+gamma);
    

    window = window_func(2*gamma + 1 , gamma , gamma +1); 
    Ryu_windowed = Ryu .* window; 
    Ruu_windowed = Ruu .* window; 
    Ryy_windowed = Ryy .* window;

    S_yu = fft(Ryu_windowed, 2*gamma + 1); 
    S_uu = fft(Ruu_windowed, 2*gamma + 1); 
    S_yy = fft(Ryy_windowed, 2*gamma + 1);

    rs = S_yy - (abs(S_yu).^2 ./ S_uu);
    
    rs=rs(1:gamma+1);
    
end

function g = SPA(y, u, N, gamma, window_func)
    y = y(1:N);
    u = u(1:N);
    Ryu = xcorr(y, u, N, 'biased'); 
    Ruu = xcorr(u, N, 'biased'); 
    Ryu = Ryu'; 
    Ruu = Ruu';
    Ryu = Ryu(N-gamma:N+gamma);
    Ruu = Ruu(N-gamma:N+gamma);

    window = window_func(2*gamma + 1 , gamma , gamma +1); 
    Ryu_windowed = Ryu .* window; 
    Ruu_windowed = Ruu .* window; 

    S_yu = fft(Ryu_windowed, 2*gamma + 1); 
    S_uu = fft(Ruu_windowed, 2*gamma + 1); 
    
    g = S_yu ./ S_uu;
    
    g=g(1:gamma+1);
    
end

function theta = corr(y,u,M,N)
    
    R_yu = xcorr(y, u, 'biased'); 
    R_uu = xcorr(u, 'biased'); 

    Phi = zeros(N, M);
    temp = zeros(1, M);
    for j = 1:N
        for i = 1:M
            temp(i) = R_uu(N+j-i); 
        end
        Phi(j,:) = temp;
    end

    R_yu_vector = R_yu(N:end); 

    theta = (Phi' * Phi) \ (Phi' * R_yu_vector);
end

