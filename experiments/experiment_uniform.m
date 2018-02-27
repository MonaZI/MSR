% MSR with uniform shift pmf
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code:

clear all
close all
clc

% list of parameters
% signal and observations
d = sort([[11:10:101],[15:10:95]],'ascend');
n = 1e5;
sigma = 0;
pmf_type = 'uniform';

% optimization
lambda_mu = 1;
lambda_C = 1;
lambda_T = 1;
lamda = [lambda_mu, lambda_C, lambda_T];
T_gen = 1;

th = 1e-3;
num_repeats = 50;

p_th = zeros(length(d),max(d));
MSE_x = zeros(length(d), max(d), num_repeats);
fval = zeros(length(d), max(d), num_repeats);

if isempty(gcp('nocreate'))
    parpool('local',4);
end

for i = 1:length(d)
    % generating a signal of length d(i)
    x_true = rand(d(i),1);
    
    % generating the shifts based on the distribution (which is uniform here)
    [p_true, X] = sig_shifter(d(i), n, x_true, pmf_type);
    
    for m = 2:1:d(i)
        [mu_est, C_est, T_est] = generate_invariants(X, m, sigma, T_gen);
        
        mse_x_epoch = zeros(num_repeats,1);
        fval_epoch = zeros(num_repeats,1);
        
        % use parfor to run the iterations in parallel
        for iter = 1:num_repeats
            [ x_est, fval_epoch(iter), ~ ] = ...
                uniform_p(d(i), mu_est, C_est, T_est);
            x_align = align_to_ref(x_est, x_true);
            mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
        end
        
        fprintf('d = %d, m = %d, mse_x = %f, fval = %f\n',...
            d(i),m,mean(mse_x_epoch),mean(fval_epoch))
        p_th(i,m) = length(find(mse_x_epoch < th))/num_repeats;
        MSE_x(i,m,:) = mse_x_epoch;
        fval(i,m,:) = fval_epoch;
        
    end
    
end
% save('experiment_uniform', 'MSE_x', 'fval')

figure
imagesc(d,[1:max(d)],p_th')
xlabel('d')
ylabel('m')
colorbar
title(['p_{th}\{MSE < ',num2str(th),'\}'])
