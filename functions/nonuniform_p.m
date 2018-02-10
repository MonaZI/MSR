function [ rec_sig, p_est, fval ] = nonuniform_p(d, mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T)
%Estimating the signal x and shift's pmf p from invariants mu, C, T when 
%p is a non-uniform distribution
%input: 
%       d: signal length
%       mu_est: estimated mean
%       C_est: estimated 2nd order correlation
%       T_est: estimated 3rd oredr correlation (bispectrum)
%       lambda_mu: the weight corresponding to the mean term
%       lambda_C: the weight corresponding to the 2nd order moment term
%       lambda_T: the weight corresponding to the 3rd order moment term
%output: 
%       rec_sig: estimated signal
%       est_p: estimated shifts pmf
%       p: true distribution
%       fval: value of the objective function at the final point

% Initialization
xinit = rand(d, 1);
p0 = rand(d, 1);
p0 = p0/sum(p0);
z0 = [xinit; p0(2:end)];

% Defining the optimization problem
A = -eye(2*d-1);
A = [A; [ zeros(1, d), ones(1, d-1)]];
b = [zeros(2*d-1, 1); 1];
F = @(z)objfun(z, mu_est, C_est, T_est, lambda_mu, lambda_C, lambda_T);
% options = optimoptions('fmincon', 'Display','off','Algorithm','sqp', 'SpecifyObjectiveGradient',true, 'FunctionTolerance', 1e-16, 'StepTolerance', 1e-15, 'MaxIterations', 4000, 'CheckGradient', false);
options = optimoptions('fmincon', 'Display','off','Algorithm','sqp', 'GradObj','on', 'TolFun', 1e-16, 'MaxIter', 4000, 'MaxFunEvals', 10000, 'DerivativeCheck', 'off');
[z, fval] = fmincon(F, z0, A, b, [], [], [], [], [], options);

p_est = z(d+1:end);
p_est = [1-sum(p_est); p_est];
rec_sig = z(1:d);
end



function [ cost, G ] = objfun( f, mu0, C0, T0, lambda_mu, lambda_C, lambda_T )
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
T = zeros(m, m, m);
G1 = zeros(d, 1); G2 = zeros(d, 1); G3 = zeros(d, 1);
Gp1 = zeros(d, 1); Gp2 = zeros(d, 1); Gp3 = zeros(d, 1);

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
        for l = 1:m
            tmp3 = circshift(f, -(l-1));
            fil = circshift(f, mod((i-l), d));
            T(i, j, l) = sum(tmp1.*tmp2.*tmp3.*p);
            Ttmp = T0(i, j, l) - T(i, j, l);
            G3 = G3 - 6*Ttmp*P_i.*fij.*fil;
            Gp3 = Gp3 - 2*Ttmp*(tmp1.*tmp2.*tmp3);
        end;
    end;
end;

cost = lambda_T * norm(T0(:) - T(:), 'fro')^2 + ...
       lambda_C * norm(C0 - C, 'fro')^2 + ...
       lambda_mu * norm(mu - mu0, 'fro').^2;
Gx = lambda_T * G3 + lambda_C * G2 + lambda_mu * G1;
tmp = lambda_T * Gp3 + lambda_C * Gp2 + lambda_mu * Gp1;

Gp = tmp(2:end) - tmp(1);
G = [Gx; Gp];
end
