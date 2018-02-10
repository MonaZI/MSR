function [mu, C_denoised, T_denoised] = generate_invariants(X, m, sigma, bispec_ind)
% Generating invariant features based on the noisy shifted observations
%input:
%   x_true: the true signal
%   shifts: the set of shifts distributed based on the pmf of shifts
%   m: mask length
%   sigma: the standard deviation of the noise
%   bispec_ind: if 1, bispectrum is computed, else it is not computed
%output:
%   mu: the estimated mean
%   C: the estimated correlation
%   T: the estimated 3rd order correlation

n = size(X, 2);

X = X(1:m,:);
X = X + sigma * randn(size(X));

% compute invariants
mu = mean(X, 2);
C = 1/n * (X*X');
% try tensorlab here !!!!
T = zeros(m, m, m);
if bispec_ind == 1
    for i = 1:m
        for j = 1:m
            for k = 1:m
                T(i, j, k) = 1/n * sum((X(i, :).*X(j, :).*X(k, :)));
            end;
        end;
    end;
end

% denoising the invariants
if sigma ==0 
    C_denoised = C;
    T_denoised = T;
else
    [C_denoised,T_denoised] = C_T_denoiser(mu, C, T, sigma);
end

end