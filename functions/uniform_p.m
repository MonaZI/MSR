function [ x_est, fval, time ] = uniform_p(d, mu_est, C_est, T_est, lambda)
%Estimating the signal x and shift's pmf p from invariants mu, C, T when
%p is a uniform distribution
%input:
%       d: signal length
%       mu_est: estimated mean
%       C_est: estimated 2nd order correlation
%       T_est: estimated 3rd oredr correlation (bispectrum)
%       lambda: the vector of weights corresponding to each term in the
%       objective
%output:
%       x_est: estimated signal
%       fval: value of the objective function at the final point
%       time: amount of required time for the computations
%
% ! Note that based on the MATLAB version, the options set in the 
% "optimoptions" might have different names. We used MATLAB R2015b.
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code: https://github.com/MonaZI/MSR


if ~exist('lambda','var') || isempty(lambda)
    lambda = ones(3,1);
end

lambda_mu = lambda(1);
lambda_C = lambda(2);
lambda_T = lambda(3);

% Initialization
xinit = rand(d, 1);
z0 = xinit;

% Defining the optimization problem
F = @(z)objfun_unif(z, mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T);
options = optimoptions('fmincon', 'Display','off','Algorithm','sqp',...
    'GradObj','on', 'TolFun', 1e-16, 'MaxIter', 4e3, 'MaxFunEvals', 1e4,...
    'DerivativeCheck', 'off');
tic
[z, fval] = fmincon(F, z0, [], [], [], [], [], [], [], options);
time = toc;

x_est = z(1:d);
end



function [ cost, G ] = objfun_unif( f, mu0, C0, T0, lambda_mu, lambda_C, lambda_T )
%lambda_T x ||  T0 - T(x,p) ||^2 + ...
%lambda_C x ||  C0 - C(x,p) ||^2 + ...
%lambda_mu x || mu0 - mu(x,p) ||^2

m = size(C0, 2);
d = length(f);
p = 1/d * ones(d, 1);

C = zeros(m, 1);
T = zeros(m, m);
G1 = zeros(d, 1); G2 = zeros(d, 1); G3 = zeros(d, 1);

mu0 = mu0(1);
C0 = C0(:,1);
T0 = T0(:,:,1);

mu = mean(f);
G1 = G1 - 2*(mu0 - mu)*p;
for i = 1:m
    tmp1 = circshift(f, -(i-1));
    tmp2 = circshift(f, i-1);
    C(i) = 1/d * (f' * tmp1);
    Ctmp = C0(i) - C(i);
    G2 = G2 - (2/d)*Ctmp*(tmp1+tmp2);
    for j = 1:m
        tmp3_1 = circshift(f, -(j-1));
        T(i,j) = 1/d * sum(f.*tmp1.*tmp3_1);
        Ttmp = T0(i,j) - T(i,j);
        
        G3 = G3 - 2/d*(circshift(f, -(j-1)) .* circshift(f, -(i-1))+...
            circshift(f, (j-1)) .* circshift(f, j-i)+...
            circshift(f, (i-1)) .* circshift(f, i-j))*Ttmp;
        
    end
    
end

cost = lambda_T * norm(T0(:) - T(:), 'fro')^2 + ...
    lambda_C * norm(C0 - C, 'fro')^2 + ...
    lambda_mu * norm(mu - mu0, 'fro').^2;
Gx = lambda_T * G3 + lambda_C * G2 + lambda_mu * G1;

G = [Gx];
end
