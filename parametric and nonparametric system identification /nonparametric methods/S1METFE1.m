
% Load and preprocess data
data = load('D:\Edu\System Identification\sim 1\data1.mat'); 
uct = detrend(data.uc);
yct= detrend(data.yc);
uwt= detrend(data.uw);
ywt = detrend(data.yw);
Nt = length(uct); 

N_values = [1024, 2048, 4096, 8192, 16384];


% system with white noise -----------------------------------------------


mean_gw = zeros(length(N_values),Nt/2);
for idx = 1:length(N_values)
    N = N_values(idx);
    num_groups = floor(Nt / N); 
    gh = zeros(num_groups, N/2); 
    
    for group = 1:num_groups
        u_group = uwt((group-1)*N + 1 : group*N);
        y_group = ywt((group-1)*N + 1 : group*N);
        gh(group, :) = ETFE(y_group, u_group,N);
    end
   
    mean_gw(idx,1:N/2) = mean(gh, 1);
    freq = 2*pi*(0:N-1)/N;
    freq = freq(1:floor(N/2));

    % magnitude ----------------------------------------------------
    figure(2);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            h_individual = plot(freq, 20*log10(abs(gh(group, :))),'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(freq,20*log10(abs(mean_gw(idx,1:N/2))),'k', 'LineWidth', 1 ); 
        hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        legend([h_individual(1), h_mean], {'Individual', 'Mean $\beta_r = 1$ '}, 'Location', 'southwest','Interpreter', 'latex');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Magnitude (dB)');
        title(sprintf('White Noise, N = %d', N));
        
    end
    if idx == length(N_values)
        mean_gw(length(N_values),1:N/2) = ETFE(ywt,uwt,N);
    end

    % phase  ----------------------------------------------------------
    figure(4);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            h_individual = plot(freq, angle(gh(group, :))*180/pi ,'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(freq, angle(mean_gw(idx,1:N/2))*180/pi ,'k', 'LineWidth', 1 ); 
        hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        legend([h_individual(1), h_mean], {'Individual', 'Mean $\beta_r = 1$ '}, 'Location', 'southwest','Interpreter', 'latex');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Phase (deg)');
        title(sprintf('White Noise , N = %d', N));
        
    end
 
end


freq = 2*pi*(0:Nt-1)/Nt;
freq = freq(1:floor(Nt/2));

figure(1);
subplot(1,2,1);
semilogx(freq, 20*log10(abs(mean_gw(length(N_values),:))),'k', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title('G Mean ETFE White noise');
grid on;

figure(3);
subplot(1,2,1);
semilogx(freq, angle(mean_gw(length(N_values),:))*180/pi,'k', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
title('G Mean ETFE White noise');
grid on;


% system with colored noise -----------------------------------------------


mean_gc = zeros(length(N_values),Nt/2);
for idx = 1:length(N_values)
    N = N_values(idx);
    num_groups = floor(Nt / N); 
    gh = zeros(num_groups, N/2); 
    
    for group = 1:num_groups
        u_group = uct((group-1)*N + 1 : group*N);
        y_group = yct((group-1)*N + 1 : group*N);
        gh(group, :) = ETFE(y_group, u_group,N);
    end
   
    mean_gc(idx,1:N/2) = mean(gh, 1);
    freq = 2*pi*(0:N-1)/N;
    freq = freq(1:floor(N/2));

    % magnitude ----------------------------------------------------
    figure(5);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            h_individual = plot(freq, 20*log10(abs(gh(group, :))),'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(freq,20*log10(abs(mean_gc(idx,1:N/2))),'b', 'LineWidth', 1 ); 
        hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        legend([h_individual(1), h_mean], {'Individual', 'Mean $\beta_r = 1$ '}, 'Location', 'southwest','Interpreter', 'latex');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Magnitude (dB)');
        title(sprintf('Colored Noise , N = %d', N));
        
    end
    if idx == length(N_values)
        mean_gc(length(N_values),1:N/2) = ETFE(ywt,uwt,N);
    end

    % phase  ----------------------------------------------------------
    figure(6);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            h_individual = plot(freq, angle(gh(group, :))*180/pi ,'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(freq, angle(mean_gc(idx,1:N/2))*180/pi ,'b', 'LineWidth', 1 ); 
        hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        legend([h_individual(1), h_mean], {'Individual', 'Mean $\beta_r = 1$ '}, 'Location', 'southwest','Interpreter', 'latex');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Phase (deg)');
        title(sprintf('Colored Noise , N = %d', N));
        
    end
 
end


freq = 2*pi*(0:Nt-1)/Nt;
freq = freq(1:floor(Nt/2));

figure(1);
subplot(1,2,2);
semilogx(freq, 20*log10(abs(mean_gc(length(N_values),:))),'b', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title('G ETFE Colored noise (Magnitude)');
grid on;

figure(3);
subplot(1,2,2);
semilogx(freq, angle(mean_gc(length(N_values),:))*180/pi,'b', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
title('G ETFE Colored noise (Phase)');
grid on;


% auto correlation of impulse responses for different N -----------------

% system with white noise -----------------------------------------------

figure(7);
subplot(2,2,1);
hold on;   
for idx = 1:length(N_values)
    N = N_values(idx);
    plot(xcorr(detrend(20*log10(abs(mean_gw(idx,1:N/2))),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Frequency (Rad/s)');
ylabel('Auto Correlation Coefficient');
title('G Magnitude Consistensy White noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;

subplot(2,2,3);
hold on;   
for idx = 1:length(N_values)
    N = N_values(idx);
    plot(xcorr(detrend(angle(mean_gw(idx,1:N/2)),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Frequency (Rad/s)');
ylabel('Auto Correlation Coefficient');
title('G Phase Consistensy White noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;

% system with colored noise -----------------------------------------------

subplot(2,2,2);
hold on;

for idx = 1:length(N_values)
    N = N_values(idx);
    plot(xcorr(detrend(20*log10(abs(mean_gc(idx,1:N/2))),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Lag');
ylabel('Auto Correlation Coefficient');
title('G Magnitude Consistensy Colored noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;

subplot(2,2,4);
hold on;

for idx = 1:length(N_values)
    N = N_values(idx);
    plot(xcorr(detrend(angle(mean_gc(idx,1:N/2)),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Lag');
ylabel('Auto Correlation Coefficient');
title('G Phase Consistensy Colored noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;


% function X = computeDFT(x, N)
%     X = zeros(N, 1);  
%     for k = 0:N-1
%         for n = 0:N-1
%             X(k+1) = X(k+1) + x(n+1) * exp(-1j * 2 * pi * k * n / N);
%         end
%     end
%     X = X / sqrt(N); 
% end

function g = ETFE(y,u,N)
    U = fft(u, N);  
    Y = fft(y, N);      
    g = Y ./ U;
    g = g(1:floor(N/2));
end



