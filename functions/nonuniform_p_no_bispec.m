function [ rec_sig, p_est, fval, time ] = nonuniform_p_no_bispec(d, mu_est, C_est, lambda)
%Estimating the signal x and shift's pmf p from invariants mu, C, T when 
%p is a non-uniform distribution
%input: 
%       d: signal length
%       mu_est: estimated mean
%       C_est: estimated 2nd order correlation
%       lambda: the vector of weights corresponding to each term in the
%       objective
%output:
%       rec_sig: estimated signal
%       est_p: estimated shifts pmf
%       p: true distribution
%       fval: value of the objective function at the final point
%       time: time (s) that takes for the optimization to finish
%
% ! Note that based on the MATLAB version, the options set in the 
% "optimoptions" might have different names. We used MATLAB R2015b.
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code: https://github.com/MonaZI/MSR

if ~exist('lambda','var') || isempty(lambda)
    lambda = ones(2,1);
end

lambda_mu = lambda(1);
lambda_C = lambda(2);

% Initialization
xinit = rand(d, 1);
p0 = rand(d, 1);
p0 = p0/sum(p0);
z0 = [xinit; p0(2:end)];

% Defining the optimization problem
% constraints on p
A = [zeros(d-1,d),-eye(d-1);[ zeros(1, d), ones(1, d-1)]];
b = [zeros(d-1, 1); 1];
F = @(z)objfun(z, mu_est, C_est, lambda_mu, lambda_C);
options = optimoptions('fmincon', 'Display','off','Algorithm','sqp',...
    'GradObj','on', 'TolFun', 1e-16, 'MaxIter', 4e3, 'MaxFunEvals', 1e4,...
    'DerivativeCheck', 'off');
tic
[z, fval] = fmincon(F, z0, A, b, [], [], [], [], [], options);
time = toc;

p_est = z(d+1:end);
p_est = [1-sum(p_est); p_est];
rec_sig = z(1:d);
end



function [ cost, G ] = objfun( f, mu0, C0, lambda_mu, lambda_C )
%lambda_T x ||  T0 - T(x,p) ||^2 + ...
%lambda_C x ||  C0 - C(x,p) ||^2 + ...
%lambda_mu x || mu0 - mu(x,p) ||^2

m = size(C0, 2);
d = length(f);
d = ceil(d/2);

p = f(d+1:end);
p = [ 1 - sum(p); p ];
f = f(1:d);

mu = zeros(m, 1);
C = zeros(m, m);
G1 = zeros(d, 1); G2 = zeros(d, 1);
Gp1 = zeros(d, 1); Gp2 = zeros(d, 1);

for i = 1:m
    tmp1 = circshift(f, -(i-1));
    P_i = circshift(p, mod(i-1, d));
    mu(i) = tmp1'*p;
    G1 = G1 - 2*(mu0(i) - mu(i))*circshift(p, mod(i-1, d));
    Gp1 = Gp1 - 2*(mu0(i) - mu(i))*tmp1;
    for j = 1:m
        tmp2 = circshift(f, -(j-1));
        fij = circshift(f, mod((i-j), d));
        C(i, j)= bsxfun(@times, tmp1, p).'*tmp2;
        Ctmp = C0(i, j) - C(i, j);
        G2 = G2 - 4*Ctmp*P_i .* fij;
        Gp2 = Gp2 - 2*Ctmp*(tmp1.*tmp2); 
    end;
end;

cost = lambda_C * norm(C0 - C, 'fro')^2 + ...
       lambda_mu * norm(mu - mu0, 'fro').^2;
Gx = lambda_C * G2 + lambda_mu * G1;
tmp = lambda_C * Gp2 + lambda_mu * Gp1;

Gp = tmp(2:end) - tmp(1);
G = [Gx; Gp];
end
