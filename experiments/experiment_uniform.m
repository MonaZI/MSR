clear all
close all

% delete(gcp('nocreate'))
% parpool('local',16);

th = 1e-3;
d = [3:9];
sigma = 0;
n = 1e5;
num_repeats = 50;
lambda_mu = 1;
lambda_C = 1;
lambda_T = 1;
pmf_type = 'uniform';

p_th = zeros(length(d),max(d));
MSE_x = zeros(length(d), max(d), num_repeats);
fval = zeros(length(d), max(d), num_repeats);
for i = 1:length(d)
    % generating a signal of length d
    x_true = rand(d(i),1);
    
    % generating the shifts based on the distribution (which is uniform here)
    [p_true, X] = sig_shifter(d(i), n, x_true, pmf_type);
    
    for m = 1:1:d(i)
        [mu_est, C_est, T_est] = generate_invariants(X, m, sigma, 1);
        
        mse_x_epoch = zeros(num_repeats,1);
        fval_epoch = zeros(num_repeats,1);
        for iter = 1:num_repeats
            [ x_est, fval_epoch(iter) ] = uniform_p(d(i), mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T);
            x_align = align_to_ref(x_est, x_true);
            mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
        end
        
        fprintf('d = %d, m = %d, mse_x = %f, fval = %f\n',d(i),m,mean(mse_x_epoch),mean(fval_epoch))
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
title('p_{th}\{MSE < th\}')