
% Load and preprocess data
data = load('D:\Edu\System Identification\sim 1\data1.mat'); 
uct = detrend(data.uc);
yct= detrend(data.yc);
uwt= detrend(data.uw);
ywt = detrend(data.yw);
Nt = length(uct);

N_values = [1024, 2048, 4096, 8192, 16384];
gamma_values = [200, 400, 800 , 1600];

% system with white noise -----------------------------------------------

N = 8192; 

for idx = 1:length(gamma_values)
    gamma = gamma_values(idx);

    %hamming ---------------------------------------
    g = SPA(ywt, uwt, N, gamma, @hamm); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(2);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'k', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(4);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'k', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');

    %constant ---------------------------------------
    g = SPA(ywt, uwt, N, gamma, @cons); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(2);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'c', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(4);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'c', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');


    %parzen ---------------------------------------
    g = SPA(ywt, uwt, N, gamma, @parz); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(2);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'g', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(4);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'g', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');

    %bartlet ---------------------------------------
    g = SPA(ywt, uwt, N, gamma, @bart); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(2);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'r', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southwest');

    % Phase Plot --------------------------------
    figure(4);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'r', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('White, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southwest');
end

% system with Colored noise -----------------------------------------------

for idx = 1:length(gamma_values)
    gamma = gamma_values(idx);

    %hamming ---------------------------------------
    g = SPA(yct, uct, N, gamma, @hamm); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(1);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'k', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(3);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'k', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');

    %constant ---------------------------------------
    g = SPA(yct, uct, N, gamma, @cons); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(1);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'c', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(3);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'c', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');


    %parzen ---------------------------------------
    g = SPA(yct, uct, N, gamma, @parz); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(1);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'g', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    
    % Phase Plot --------------------------------
    figure(3);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'g', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');

    %bartlet ---------------------------------------
    g = SPA(yct, uct, N, gamma, @bart); 
   
    freq = 2*pi*(0:2*gamma)/(2*gamma+1);
    freq = freq(1:gamma+1);
    % Magnitude Plot ----------------------------
    figure(1);
    subplot(2, 2, idx);
    hold on;
    plot(freq, 20*log10(abs(g)),'r', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Magnitude (dB)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southwest');

    % Phase Plot --------------------------------
    figure(3);

    subplot(2, 2, idx);
    hold on;
    plot(freq, angle(g)*180/pi,'r', 'LineWidth', 1); 
    hold off;
    grid on;
    set(gca, 'XScale', 'log'); 
    xlabel('Frequency (Rad/s)');
    ylabel('Phase (deg)');
    title(sprintf('Colored, N = %d, $\\gamma = %d$', N, gamma), 'Interpreter', 'latex');
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southwest');
end


% windows ----------------------------------------------
w_constant = cons(101, 50, 51);
w_parzen = parz(101, 50, 51);
w_bartlett = bart(101, 50, 51);
w_hamming = hamm(101, 50, 51); %N, gamma, location

figure(30);
hold on;
plot(w_constant,'c', 'LineWidth', 1.5,'DisplayName', 'Constant');
plot(w_parzen,'g', 'LineWidth', 1.5, 'DisplayName', 'Parzen');
plot(w_bartlett,'r', 'LineWidth', 1.5, 'DisplayName', 'Bartlett');
plot(w_hamming,'k', 'LineWidth', 1.5, 'DisplayName', 'Hamming');
hold off;
xlabel('Index');
xlim([1,101]);
ylabel('Amplitude');
title('Windows Comparison');
legend show;
grid on;


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






