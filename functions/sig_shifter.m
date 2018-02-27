function [p_true, X] = sig_shifter(d, n, x_true, pmf_type)
%Generating pmf of the shifts based on the required pmf_type
%input: 
%       d: signal length
%       n: number of samples
%       x_true: the true value of the signal
%       pmf_type: the type of pmf which can be "uniform" or "nonuniform"
%output: 
%       p_true: the true distribution of the shifts
%       X: the clean shifted observations of the signal
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code:

assert((strcmp(pmf_type,'nonuniform')) | (strcmp(pmf_type,'uniform')), ...
    'Wrong pmf type! The choices for pmf_type are 1) uniform, 2) nonuniform')

if strcmp(pmf_type,'uniform')
    p_true = ones(d,1);
elseif strcmp(pmf_type,'nonuniform')
    p_true = rand(d,1);
end

p_true = p_true/sum(p_true);
cdf_shift = cumsum(p_true);

shifts = zeros(1,n);
shifts(1:ceil(cdf_shift(1)*n)) = 0;
for k = 2:d
    shifts(1+ceil(cdf_shift(k-1)*n):ceil(cdf_shift(k)*n)) = k-1;
end

% generating the clean shifted versions of the signal
X = zeros(d, n);
for i = 1:n
    X(:, i) =  circshift(x_true, -shifts(i));
end;

end
