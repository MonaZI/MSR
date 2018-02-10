% jobscript comparing our moment matching method with EM for small mask
% lengths

clear all
% close all

% delete(gcp('nocreate'))
% parpool('local',4);

sigma = 10.^-0.3;
m = [11:-1:6];
d = 21;
n = 1e5;
tol = 1e-5; % the tolerance corresponding to the EM solver
num_repeats = 50;
lambda_mu = 1;
lambda_C = 1;
lambda_T = 1;
pmf_type = 'uniform';

% initialization
mse_x_EM = zeros(length(m),num_repeats);
mse_p_EM = zeros(length(m),num_repeats);
mse_x = zeros(length(sigma),num_repeats);

rel_EM = zeros(length(sigma),num_repeats);
rel_cov = zeros(length(sigma),num_repeats);

p_error_EM = zeros(length(sigma),num_repeats);
p_error_cov = zeros(length(sigma),num_repeats);

time_EM = zeros(length(sigma),num_repeats);
time_cov = zeros(length(sigma),num_repeats);

fval_cov = zeros(length(sigma),num_repeats);
fval_EM = zeros(length(sigma),num_repeats);

% generating the signal, the shift pmf and the shifted versions of the
% signal
x_true = rand(d,1);
[p_true, X] = sig_shifter(d, n, x_true, pmf_type);

for m_ind = 1:length(m)
    
    % initialization, method of moments
    rel_cov_epoch = zeros(num_repeats,1); time_cov_epoch = zeros(num_repeats,1);
    mse_x_epoch = zeros(num_repeats,1); fval_epoch = zeros(num_repeats,1);
    
    % initizalization, EM
    rel_EM_epoch = zeros(num_repeats,1); time_EM_epoch = zeros(num_repeats,1);
    mse_x_EM_epoch = zeros(num_repeats,1); mse_p_EM_epoch = zeros(num_repeats,1);
    fval_EM_epoch = zeros(num_repeats,1);
    
    
    parfor iter = 1:num_repeats
        iter
        [mu_est, C_est, T_est] = generate_invariants(X, m(m_ind), sigma, 1);
        
        % bispectrum included in the objective, method of moments
        [ x_est, fval_epoch(iter), time_cov_epoch(iter) ] = uniform_p(d, mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T);
        x_align = align_to_ref(x_est, x_true);
        mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
        rel_cov_epoch(iter) = (norm(x_align-x_true,'fro'))^2 / (norm(x_true,'fro'))^2;
        
        % EM
        obs = X(1:m(m_ind),:) + sigma * randn(m(m_ind),n);
        % initializing the EM with the results from the method of moments
        [x_EM,p_EM,time_EM_epoch(iter)] = EM_solver(obs,d,sigma,tol,5e5,rand(d,1),p_true);
        x_align_EM = align_to_ref(x_EM, x_true);
        p_align_EM = align_to_ref(p_EM, p_true);
        mse_x_EM_epoch(iter) = (norm(x_align_EM-x_true,'fro'))^2;
        mse_p_EM_epoch(iter) = (norm(p_align_EM-p_true,'fro'))^2;
        rel_EM_epoch(iter) = (norm(x_align_EM-x_true,'fro'))^2 / (norm(x_true,'fro'))^2;
        fval_EM_epoch(iter) = likelihood_func( x_align_EM, p_align_EM, obs, sigma );
    end
    time_cov(m_ind,:) = time_cov_epoch;
    mse_x(m_ind,:) = mse_x_epoch;
    rel_cov(m_ind,:) = rel_cov_epoch;
    fval_cov(m_ind,:) = fval_epoch;
    
    time_EM(m_ind,:) = time_EM_epoch;
    mse_x_EM(m_ind,:) = mse_x_EM_epoch;
    mse_p_EM(m_ind,:) = mse_p_EM_epoch;
    rel_EM(m_ind,:) = rel_EM_epoch;
    fval_EM(m_ind,:) = fval_EM_epoch;  

    fprintf('m = %d, opt_cov = %f, EM = %f\n',m(m_ind),min(mse_x(m_ind,:)),min(mse_x_EM(m_ind,:)))
    save('experiment_unifrom_EM_2', 'mse_x', 'rel_cov', 'fval_cov', 'time_cov', 'mse_x_EM', 'mse_p_EM', 'rel_EM', 'fval_EM','time_EM')

end

%% plotting the results
width = 4;     % Width in inches
height = 4;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 2;      % LineWidth
msz = 8;       % MarkerSize

% The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);


ind = 9;
figure
subplot(2,1,1)
loglog(sigma_vec(1:ind),mse_EM(1:ind),'k-*','LineWidth',2,'MarkerSize',4);hold on
loglog(sigma_vec(1:ind),min(mse_cov(1:ind,:),[],2),'kx-','LineWidth',1)
legend({'EM','our approach'},'Location','southeast','FontSize',9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'MSE'},'FontSize',10)
grid on


subplot(2,1,2)
loglog(sigma_vec(1:ind),time_EM(1:ind),'k-*','LineWidth',2,'MarkerSize',4);hold on
loglog(sigma_vec(1:ind),mean(time_cov(1:ind,:),2),'kx-','LineWidth',1)
legend({'EM','our approach'},'Location','southeast','FontSize',9)
xlabel({'\sigma'},'FontSize',10)
ylabel({'MSE'},'FontSize',10)
grid on
