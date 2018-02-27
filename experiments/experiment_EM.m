% Comparing our approach in solving MSR with EM
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code: https://github.com/MonaZI/MSR

clear all
close all
clc

% list of parameters
% signal and observations
d = 11;
m = 7;
n = 1e5;
sigma = 10.^[-0.3:0.05:log10(2)];
pmf_type = 'nonuniform';

% optimization
lambda_mu = 1;
lambda_C = 1;
lambda = [lambda_mu, lambda_C];
T_gen = 0;

% EM
tol = 1e-5;
max_iter = 5e5;

num_repeats = 1;

% initialization
mse_x_EM = zeros(length(sigma),1);
mse_p_EM = zeros(length(sigma),1);
mse_x = zeros(length(sigma),num_repeats);
mse_p = zeros(length(sigma),num_repeats);

rel_EM = zeros(length(sigma),1);
rel_cov = zeros(length(sigma),num_repeats);

p_error_EM = zeros(length(sigma),1);
p_error_cov = zeros(length(sigma),num_repeats);

time_EM = zeros(length(sigma),1);
time_cov = zeros(length(sigma),num_repeats);

fval_cov = zeros(length(sigma),num_repeats);
fval_EM = zeros(length(sigma),num_repeats);

% generating the signal, the shift pmf and the shifts
x_true = rand(d,1);
[p_true, X] = sig_shifter(d, n, x_true, pmf_type);
X = X(1:m,:);

if isempty(gcp('nocreate'))
    parpool('local',4);
end

for sigma_ind = 1:length(sigma)
    fprintf(['\nsigma = ',num2str(sigma(sigma_ind)),'\n'])

    % initialization, method of moments
    rel_cov_epoch = zeros(num_repeats,1); time_cov_epoch = zeros(num_repeats,1);
    mse_x_epoch = zeros(num_repeats,1); mse_p_epoch = zeros(num_repeats,1);
    fval_epoch = zeros(num_repeats,1);
    
    % initizalization, EM
    rel_EM_epoch = zeros(num_repeats,1); time_EM_epoch = zeros(num_repeats,1);
    mse_x_EM_epoch = zeros(num_repeats,1); mse_p_EM_epoch = zeros(num_repeats,1);
    fval_EM_epoch = zeros(num_repeats,1);
    
    % use parfor to run the iterations in parallel
    for iter = 1:num_repeats 
        fprintf(['iteration = ',num2str(iter),'\n'])
        
        % generaing the invariants
        tic
        [mu_est, C_est, T_est] = generate_invariants(X, m, sigma(sigma_ind), T_gen);
        time_cov_epoch(iter) = toc;
        
        % our approach
        fprintf('Our approach \n')
        [ x_est, p_est, fval_cov(iter), time ] = ...
            nonuniform_p_no_bispec(d, mu_est, C_est, lambda);
        
        time_cov_epoch(iter) = time_cov_epoch(iter) + time;
        x_align = align_to_ref(x_est, x_true);
        p_align = align_to_ref(p_est, p_true);
        mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
        mse_p_epoch(iter) = (norm(p_align-p_true,'fro'))^2;
        rel_cov_epoch(iter) = (norm(x_align-x_true,'fro'))^2 / (norm(x_true,'fro'))^2;
        
        % EM
        fprintf('EM \n')
        obs = X(1:m,:) + sigma(sigma_ind) * randn(m,n);
        [x_EM,p_EM,time_EM_epoch(iter)] = ...
            EM_solver(obs, d, sigma(sigma_ind), tol, max_iter);
        
        x_align_EM = align_to_ref(x_EM, x_true);
        p_align_EM = align_to_ref(p_EM, p_true);
        mse_x_EM_epoch(iter) = (norm(x_align_EM-x_true,'fro'))^2;
        mse_p_EM_epoch(iter) = (norm(p_align_EM-p_true,'fro'))^2;
        rel_EM_epoch(iter) = (norm(x_align_EM-x_true,'fro'))^2 / (norm(x_true,'fro'))^2;
        fval_EM_epoch(iter) = ...
            likelihood_func( x_align_EM, p_align_EM, obs, sigma(sigma_ind) );
    end
    
    time_cov(sigma_ind,:) = time_cov_epoch;
    mse_x(sigma_ind,:) = mse_x_epoch;
    rel_cov(sigma_ind,:) = rel_cov_epoch;
    fval_cov(sigma_ind,:) = fval_epoch;
    
    time_EM(sigma_ind,:) = time_EM_epoch;
    mse_x_EM(sigma_ind,:) = mse_x_EM_epoch;
    mse_p_EM(sigma_ind,:) = mse_p_EM_epoch;
    rel_EM(sigma_ind,:) = rel_EM_epoch;
    fval_EM(sigma_ind,:) = fval_EM_epoch;
    
    fprintf('sigma = %d, MSE_our-approach = %f, MSE_EM = %f\n\n', ...
        sigma(sigma_ind),min(mse_x(sigma_ind,:)),min(mse_x_EM(sigma_ind,:)))
    
end
% save('experiment_EM', 'mse_x', 'mse_p', 'fval_cov', 'mse_x_EM', 'mse_p_EM')

% plotting the results
figure
subplot(2,1,1)
loglog(sigma, min(mse_x_EM,[],2), 'k-*', 'LineWidth', 2, 'MarkerSize',4)
hold on
loglog(sigma, min(mse_x,[],2), 'kx-', 'LineWidth', 1)
legend({'EM','our approach'}, 'Location', 'southeast', 'FontSize', 9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'MSE'},'FontSize',10)
grid on


subplot(2,1,2)
loglog(sigma, mean(time_EM,2), 'k-*', 'LineWidth', 2, 'MarkerSize',4)
hold on
loglog(sigma, mean(time_cov,2), 'kx-', 'LineWidth', 1)
legend({'EM','our approach'}, 'Location', 'southeast', 'FontSize', 9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'Computation time'},'FontSize',10)
grid on
