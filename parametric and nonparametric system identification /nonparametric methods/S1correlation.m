
% Load and preprocess data
data = load('D:\data1.mat');  % change this
uct = detrend(data.uc);
yct = detrend(data.yc);
uwt = detrend(data.uw);
ywt = detrend(data.yw);

Nt = length(uct);
N_values = [1024, 2048, 4096, 8192, 16384];
M = 100;
colors = ['r', 'c', 'g', 'b','k']; 

% system with white noise -----------------------------------------------

figure(2);
mean_responsew = zeros(length(N_values),M);
for idx = 1:length(N_values)
    N = N_values(idx);
    num_groups = floor(Nt / N); 
    impulse_responses = zeros(num_groups, M); 
    
    for group = 1:num_groups
        u_group = uwt((group-1)*N + 1 : group*N);
        y_group = ywt((group-1)*N + 1 : group*N);
        impulse_responses(group, :) = corr(y_group, u_group, M, N);
    end
    
    mean_responsew(idx,:) = mean(impulse_responses, 1);
    
    if idx ~= 5
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            data_to_plot = impulse_responses(group, :);
            h_individual = plot(1:length(data_to_plot), data_to_plot,'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(mean_responsew(idx,:),'k.', 'MarkerSize', 5 ); 
        hold off;
        grid on;

        legend([h_individual(1), h_mean], {'Individual Responses', 'Mean Response'}, 'Location', 'northeast');
        xlabel('Time Index');
        ylabel('Impulse Response Coefficient');
        title(sprintf('White Noise, N = %d', N));
        
    end
    if idx == length(N_values)
        mean_responsew(length(N_values),:) = corr(ywt,uwt,M,Nt);
    end
end

figure(1);
subplot(1,2,1);
plot(mean_responsew(length(N_values),:),'b.', 'MarkerSize', 5);
xlabel('Time Index');
ylabel('Impulse Response Coefficient');
title('I.R White Noise N = 16384');
grid on;
hold off;


% system with colored noise -----------------------------------------------

figure(3);
mean_responsec = zeros(length(N_values),M);
for idx = 1:length(N_values)
    N = N_values(idx);
    num_groups = floor(Nt / N); 
    impulse_responses = zeros(num_groups, M); 
    
    for group = 1:num_groups
        u_group = uct((group-1)*N + 1 : group*N);
        y_group = yct((group-1)*N + 1 : group*N);
        impulse_responses(group, :) = corr(y_group, u_group, M, N);
    end
    
    mean_responsec(idx,:) = mean(impulse_responses, 1);
    
    if idx ~= length(N_values)
        subplot(2, 2, idx);
        hold on;
        for group = 1:num_groups
            data_to_plot = impulse_responses(group, :);
            h_individual = plot(1:length(data_to_plot), data_to_plot,'r.', 'MarkerSize', 5); 
        end
        h_mean = plot(mean_responsec(idx,:),'k.', 'MarkerSize', 5 ); 
        hold off;
        grid on;

        legend([h_individual(1), h_mean], {'Individual Responses', 'Mean Response'}, 'Location', 'northeast');
        xlabel('Time Index');
        ylabel('Impulse Response Coefficient');
        title(sprintf('Colored Noise, N = %d', N));
        
    end
    if idx == length(N_values)
        mean_responsec(length(N_values),:) = corr(yct,uct,M,Nt);
    end
end

figure(1);
subplot(1,2,2);
plot(mean_responsec(length(N_values),:),'b.', 'MarkerSize', 5);
xlabel('Time Index');
ylabel('Impulse Response Coefficient');
title('I.R Colored Noise N = 16384');
grid on;
hold off;


% compare --------------------------------------------------------------

% figure(4);
% subplot(1,2,2);
% grid on;
% data1 = iddata(yct, uct, 1);
% h = impulseest(data1);
% impulseplot(h);
% hold on 
% xlabel('Time Index');
% ylabel('Impulse Response Coefficient');
% title('I.R Comparsion (system with colored noise)');
% plot(corr(yct,uct,M,Nt), '.','k','MarkerSize', 5);
% legend('impulseest','correlation method (N=16384)');
% hold off;
% 
% 
% subplot(1,2,1);
% data1 = iddata(ywt, uwt, 1);
% h = impulseest(data1);
% impulseplot(h);
% hold on ;
% grid on;
% xlabel('Time Index');
% ylabel('Impulse Response Coefficient');
% title('I.R Comparison (system with white noise)');
% plot(corr(ywt,uwt,M,Nt), '.','k','MarkerSize', 5);
% legend('impulseest','correlation method(N=16384)');
% hold off;


%%%%% auto correlation of impulse responses for different N -------------

% system with white noise -----------------------------------------------

figure(4);
subplot(1,2,1);
hold on;   
for idx = 1:length(N_values)
    plot(xcorr(detrend(mean_responsew(idx,:),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Time Index');
ylabel('Auto Correlation Coefficient');
title('I.R Consistensy White noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;

% system with colored noise -----------------------------------------------

subplot(1,2,2);
hold on;

for idx = 1:length(N_values)
    plot(xcorr(detrend(mean_responsec(idx,:),0), 'biased'), 'Color', colors(idx), 'LineWidth', 1.5);
end

xlabel('Time Index');
ylabel('Auto Correlation Coefficient');
title('I.R Consistensy Colored noise');
legend(arrayfun(@(x) sprintf('N = %d', x), N_values, 'UniformOutput', false), 'Location', 'northeast');
grid on;
hold off;

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

