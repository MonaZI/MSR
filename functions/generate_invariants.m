function [mu, C_denoised, T_denoised] = generate_invariants(X, m, sigma, bispec_ind)
% Generating invariant features based on the noisy shifted observations
%input:
%   X: clean shifted observations
%   m: mask length
%   sigma: the standard deviation of the noise
%   bispec_ind: if 1, bispectrum is computed, else it is not computed
%output:
%   mu: the estimated mean
%   C: the estimated correlation
%   T: the estimated 3rd order correlation

n = size(X, 2);

% masking and adding noise
X = X(1:m,:);
X = X + sigma * randn(size(X));

% compute invariants
mu = mean(X, 2);
C = 1/n * (X*X');
T = zeros(m, m, m);
tic;
if bispec_ind == 1
   U = {X, X, X};
   % using built-in functions in tensorlab package
   T = 1/n * cpdgen(U);    
end
t = toc;

% T can be constructed using the following for loop, but it is much slower 
% compared to the above approach
% tic
% if bispec_ind == 1
%     parfor i = 1:m
%         for j = 1:m
%             for k = 1:m
%                 T(i, j, k) = 1/n * sum((X(i, :).*X(j, :).*X(k, :)));
%             end;
%         end;
%     end;
% end
% t2 = toc;

% denoising the invariants
if sigma ==0 
    C_denoised = C;
    T_denoised = T;
else
    [C_denoised,T_denoised] = C_T_denoiser(mu, C, T, sigma);
end

end