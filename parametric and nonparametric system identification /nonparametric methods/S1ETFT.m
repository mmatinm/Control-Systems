
% Load and preprocess data
data = load('D:\data1.mat');   % change this
uct = detrend(data.uc);
yct= detrend(data.yc);
uwt= detrend(data.uw);
ywt = detrend(data.yw);
Nt = length(uct); 

N_values = [1024, 2048, 4096, 8192, 16384];
colors = ['r', 'c', 'g', 'b','k']; 

% system with white noise -----------------------------------------------

for idx = 1:length(N_values)
    N = N_values(idx);
    uw = detrend(data.uw(1:N));
    yw = detrend(data.yw(1:N));
    
    freq = 2*pi*(0:N-1)/N;
    freq = freq(1:floor(N/2));

    g = ETFE(yw,uw,N);

    % magnitude ----------------------------------------------------
    figure(2);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        %hold on;
        h_mean = plot(freq,20*log10(abs(g)),'k', 'LineWidth', 1 ); 
        %hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        %legend([h_individual(1), h_mean], {'Individual', 'Mean'}, 'Location', 'southwest');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Magnitude (dB)');
        title(sprintf('White Noise, N = %d', N));
        
    end

    % phase  ----------------------------------------------------------
    figure(4);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        %hold on;
        h_mean = plot(freq, angle(g)*180/pi ,'k', 'LineWidth', 1 ); 
        %hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        %legend([h_individual(1), h_mean], {'Individual', 'Mean'}, 'Location', 'southwest');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Phase (deg)');
        title(sprintf('White Noise , N = %d', N));
        
    end
    % auto correlation of impulse responses for different N -----------------
    figure(7);
    subplot(2,2,1);
    plot(xcorr(detrend(20*log10(abs(g)),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
    hold on;
    xlabel('Lag');
    ylabel('Auto Correlation Coefficient');
    title('G Magnitude Consistensy White noise');
    legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;

    subplot(2,2,3);
    hold on;   
    plot(xcorr(detrend(angle(g),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
    xlabel('Lag');
    ylabel('Auto Correlation Coefficient');
    title('G Phase Consistensy White noise');
    legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    hold off;
    

end


freq = 2*pi*(0:Nt-1)/Nt;
freq = freq(1:floor(Nt/2));

figure(1);
subplot(1,2,1);
semilogx(freq, 20*log10(abs(ETFE(ywt,uwt,Nt))),'k', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title('G ETFE White noise (Magnitude)');
grid on;

figure(3);
subplot(1,2,1);
semilogx(freq, angle(ETFE(ywt,uwt,Nt))*180/pi,'k', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
title('G ETFE White noise (Phase)');
grid on;


% system with colored noise -----------------------------------------------

for idx = 1:length(N_values)
    N = N_values(idx);
    uc = detrend(data.uc(1:N));
    yc = detrend(data.yc(1:N));
    freq = 2*pi*(0:N-1)/N;
    freq = freq(1:floor(N/2));
    g = ETFE(yc,uc,N);

    % magnitude ----------------------------------------------------
    figure(5);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        %hold on;
        h_mean = plot(freq,20*log10(abs(g)),'b', 'LineWidth', 1 ); 
        %hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        %legend([h_individual(1), h_mean], {'Individual', 'Mean'}, 'Location', 'southwest');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Magnitude (dB)');
        title(sprintf('Colored Noise , N = %d', N));
        
    end

    % phase  ----------------------------------------------------------
    figure(6);
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        %hold on;
        h_mean = plot(freq, angle(g)*180/pi ,'b', 'LineWidth', 1 ); 
        %hold off;
        grid on;
        set(gca, 'XScale', 'log'); 
        %legend([h_individual(1), h_mean], {'Individual', 'Mean'}, 'Location', 'southwest');
        %legend('Individual','Mean');
        xlabel('Frequency (Rad/s)');
        ylabel('Phase (deg)');
        title(sprintf('Colored Noise , N = %d', N));
        
    end
    % auto correlation of impulse responses for different N -----------------
    figure(7);
    subplot(2,2,2);
    plot(xcorr(detrend(20*log10(abs(g)),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
    hold on;
    xlabel('Lag');
    ylabel('Auto Correlation Coefficient');
    title('G Magnitude Consistensy Colored noise');
    legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;

    subplot(2,2,4);
    hold on;   
    plot(xcorr(detrend(angle(g),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
    xlabel('Lag');
    ylabel('Auto Correlation Coefficient');
    title('G Phase Consistensy Colored noise');
    legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
    grid on;
    hold off;

end


freq = 2*pi*(0:Nt-1)/Nt;
freq = freq(1:floor(Nt/2));

figure(1);
subplot(1,2,2);
semilogx(freq, 20*log10(abs(ETFE(yct,uct,Nt))),'b', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Magnitude (dB)');
title('G ETFE Colored noise (Magnitude)');
grid on;

figure(3);
subplot(1,2,2);
semilogx(freq, angle(ETFE(yct,uct,Nt))*180/pi,'b', 'LineWidth', 1);
xlabel('Frequency (Rad/s)');
ylabel('Phase (deg)');
title('G ETFE Colored noise (Phase)');
grid on;


% auto correlation of impulse responses for different N -----------------


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



