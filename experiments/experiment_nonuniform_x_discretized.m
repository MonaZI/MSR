clear all
close all

% delete(gcp('nocreate'))
% parpool('local',16);

c = 4;
th = 1e-3;
d = 7;
sigma = 0;
n = 1e5;
num_repeats = 50;
lambda_mu = 1;
lambda_C = 1;
lambda_T = 1;
pmf_type = 'nonuniform';

p_th = zeros(length(d),max(d));
MSE_x = zeros(length(d), max(d), num_repeats);
MSE_p = zeros(length(d), max(d), num_repeats);
fval = zeros(length(d), max(d), num_repeats);
for i = 1:length(d)
    % generating a signal of length d with discretized values in [0:c-1]
    % interval
    x_true = randi(c,[d(i),1])-1;
    
    % generating the shifts based on the distribution (which is uniform here)
    [p_true, X] = sig_shifter(d, n, x_true, pmf_type);
    
    for m = 2:1:d(i)
        [mu_est, C_est, T_est] = generate_invariants(X, m, sigma, 1);
        
        mse_x_epoch = zeros(num_repeats,1);
        mse_p_epoch = zeros(num_repeats,1);
        fval_epoch = zeros(num_repeats,1); 
        for iter = 1:num_repeats
            % bispectrum included in the objective
            [ x_est, p_est, fval_epoch(iter) ] = nonuniform_p(d, mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T);
            x_est = round(x_est);
            x_align = align_to_ref(x_est, x_true);
            p_align = align_to_ref(p_est, p_true);
            mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
            mse_p_epoch(iter) = (norm(p_align-p_true,'fro'))^2;
        end
        
        fprintf('d = %d, m = %d, mse_x = %f, mse_p = %f, fval = %f\n',d(i),m,mean(mse_x_epoch),mean(mse_p_epoch),mean(fval_epoch))
        p_th(i,m) = length(find(mse_x_epoch < th))/num_repeats;
        MSE_x(i,m,:) = mse_x_epoch;
        MSE_p(i,m,:) = mse_p_epoch;
        fval(i,m,:) = fval_epoch;

    end

end
% save('experiment_nonuniform_x_discretized', 'MSE_x', 'MSE_p', 'fval')

figure
imagesc(d,[1:max(d)],p_th')
xlabel('d')
ylabel('m')
title('p_{th}\{MSE < th\}')