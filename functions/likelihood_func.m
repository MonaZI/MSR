function [ cost ] = likelihood_func( x, p, obs, sigma )
% Computing the likelihood function corresponding to the masking problem
%input:
%   x: the signal
%   p: pmf of the shifts
%   obs: the noisy maksed observations
%output:
%   cost: the negative log likelihood cost
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code:

d = length(x);
m = size(obs,1);
X = zeros(d,d);
for i = 0:d-1
    X(:, i+1) =  circshift(x, -(i));
end;
X = X(1:m,:);
 
n = size(obs, 2);
X_tmp = permute(X,[1,3,2]);
X_tmp = repmat(X_tmp,[1,n]);
y_tmp = repmat(obs,[1,1,d]);
tmp = exp((-1/(2*sigma^2)) * sum((y_tmp-X_tmp).^2, 1));
p_tmp = reshape(p,[1,1,d]);
p_tmp = repmat(p_tmp,[1,n]);
cost = -1*mean(log10(sum(p_tmp .* tmp, 3)));
 
end
