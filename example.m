% Quick example
%
%February 2018
%paper: https://arxiv.org/abs/1802.08950
%code: https://github.com/MonaZI/MSR

clear all
close all
clc

% list of parameters
% signal and observations
d = 41;
m = 25;
n = 1e5;
sigma = 0;
pmf_type = 'nonuniform';

% optimization
lambda_mu = 1;
lambda_C = 1;
lambda_T = 1;
lambda = [lambda_mu, lambda_C, lambda_T];

assert((strcmp(pmf_type,'nonuniform')) | (strcmp(pmf_type,'uniform')), ...
    'Wrong pmf type! The choices for pmf_type are 1) uniform, 2) nonuniform')

% generating a signal of length d
x_true = rand(d,1);

% generating the shifts based on the pmf_type
[p_true, X] = sig_shifter(d, n, x_true, pmf_type);

% estimating the invariants
if isempty(gcp('nocreate'))
    parpool('local',4);
end
fprintf('Generating the invariants ...\n')
T_gen = 1; % determines whether T is generated by generate_invariants function
tic
[mu_est, C_est, T_est] = generate_invariants(X, m, sigma, T_gen);
time = toc;

figure(1)
if strcmp(pmf_type,'nonuniform')
    fprintf('Solving MSR for nonuniform pmf case ...')
    [ x_est, p_est, fval ] = nonuniform_p(d, mu_est, C_est, T_est, lambda);
    fprintf('Done!\n')
    x_align = align_to_ref(x_est, x_true);
    mse_x = (norm(x_align-x_true,'fro'))^2;
    rel_x = mse_x / (norm(x_true,'fro'))^2;
    
    p_align = align_to_ref(p_est, p_true);
    mse_p = (norm(p_align-p_true,'fro'))^2;
    rel_p = mse_p / (norm(p_true,'fro'))^2;
    
    % comparing the recovered and the true signals and pmfs
    subplot(1,2,1)
    plot([0:d-1],x_align,'b*-')
    hold on
    plot([0:d-1],x_true,'ko--')
    legend('rec. sig.','true sig.')
    xlabel('n'); ylabel('x[n]')
    grid on; xlim([0 d-1])
    title(['|| x_{est} - x ||^2 / || x ||^2 = ',num2str(rel_x)])
    
    subplot(1,2,2)
    plot([0:d-1],p_align,'b*-')
    hold on
    plot([0:d-1],p_true,'ko--')
    legend('Recovered pmf','true pmf')
    xlabel('n'); ylabel('p[n]')
    grid on; xlim([0 d-1])
    title(['|| p_{est} - p ||^2 / || p ||^2 = ',num2str(rel_p)])
    
elseif strcmp(pmf_type,'uniform')
    fprintf('Solving MSR for uniform pmf...')
    [ x_est, fval ] = uniform_p(d, mu_est, C_est, T_est, lambda);
    fprintf('Done!\n')
    x_align = align_to_ref(x_est, x_true);
    mse_x = (norm(x_align-x_true,'fro'))^2;
    rel_x = mse_x / (norm(x_true,'fro'))^2;
    
    % comparing the recovered and the true signals and pmfs
    plot([0:d-1],x_align,'b*-')
    hold on
    plot([0:d-1],x_true,'ko--')
    legend('rec. sig.','true sig.')
    xlabel('n'); ylabel('x[n]')
    grid on; xlim([0 d-1])
    title(['|| x_{est} - x ||^2 / || x ||^2 = ',num2str(rel_x)])
    
end
