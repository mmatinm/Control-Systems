

% Load and preprocess data
%data = load('D:\Edu\System Identification\sim 1\data403208017.mat'); 
uct = detrend(data.uc);
yct= detrend(data.yc);
uwt= detrend(data.uw);
ywt = detrend(data.yw);
Nt = length(uct);

N_values = [1024, 2048, 4096, 8192, 16384];
gamma_values = [ 200, 400, 800 , 1600];

% system with white noise -----------------------------------------------

N = 8192; 

for idx = 1:length(gamma_values)
    gamma = gamma_values(idx);

    %hamming ---------------------------------------
    g = RS(ywt, uwt, N, gamma, @hamm); 
   
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
    g = RS(ywt, uwt, N, gamma, @cons); 
   
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
    g = RS(ywt, uwt, N, gamma, @parz); 
   
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
    g = RS(ywt, uwt, N, gamma, @bart); 
   
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
    g = RS(yct, uct, N, gamma, @hamm); 
   
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
    g = RS(yct, uct, N, gamma, @cons); 
   
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
    g = RS(yct, uct, N, gamma, @parz); 
   
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
    g = RS(yct, uct, N, gamma, @bart); 
   
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
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southeast');

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
    legend('Hamming','Constant','Parzen','Bartlet','Location', 'southeast');
end



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








