% jobscript comparing our alg with covariance with EM

clear all
close all

% delete(gcp('nocreate'))
% parpool('local',4);

sigma = 10.^[0:0.05:log10(2)];%-0.3
d = 45;
m = 25;
n = 1e5;
tol = 1e-6; % the tolerance corresponding to the EM solver
num_repeats = 5;
lambda_mu = 1;
lambda_C = 1;
pmf_type = 'nonuniform';

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

% generating the signal, the shift pmf and the shifts
x_true = rand(d,1);
[p_true, X] = sig_shifter(d, n, x_true, pmf_type);
X = X(1:m,:);

for sigma_ind = 1:length(sigma)
    
    rel_cov_epoch = zeros(num_repeats,1); time_cov_epoch = zeros(num_repeats,1);
    mse_x_epoch = zeros(num_repeats,1); mse_p_epoch = zeros(num_repeats,1);
    fval_epoch = zeros(num_repeats,1);
    
    for iter = 1:num_repeats
        
        [mu_est, C_est, ~] = generate_invariants(X, m, sigma(sigma_ind), 0);
        
        % bispectrum not included in the objective
        [ x_est, p_est, fval_epoch(iter) ] = nonuniform_p_no_bispec(d, mu_est, C_est, lambda_mu, lambda_C);
        x_align = align_to_ref(x_est, x_true);
        p_align = align_to_ref(p_est, p_true);
        mse_x_epoch(iter) = (norm(x_align-x_true,'fro'))^2;
        mse_p_epoch(iter) = (norm(p_align-p_true,'fro'))^2;
    end
    time_cov(sigma_ind,:) = time_cov_epoch;
    mse_x(sigma_ind,:) = mse_x_epoch;
    mse_p(sigma_ind,:) = mse_p_epoch;
    fval_cov(sigma_ind,:) = fval_epoch;
    
    [x_EM,p_EM,time_EM(sigma_ind)] = EM_solver(X + sigma(sigma_ind) * randn(size(X)),d,sigma(sigma_ind),tol);
    x_align_EM = align_to_ref(x_EM, x_true);
    p_align_EM = align_to_ref(p_EM, p_true);
    mse_x_EM(sigma_ind) = (norm(x_align_EM-x_true,'fro'))^2;
    mse_p_EM(sigma_ind) = (norm(p_align_EM-p_true,'fro'))^2;
        
    fprintf('sigma_ind = %d, opt_cov = %f, EM = %f\n',sigma_index,min(mse_cov(sigma_index,:)),mse_x_EM(sigma_index))
    
end
% save('experiment_EM', 'mse_x', 'mse_p', 'fval_cov', 'mse_x_EM', 'mse_p_EM')

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
