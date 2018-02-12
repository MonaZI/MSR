function [x,p,time_cost] = EM_solver(obs,d,sigma,tol,max_iter)
% Solving the maximum likelihood estimation of the signal and the pmf of
% the shifts using EM
%input:
%   obs: the noisy shifted observations
%   d: length of the signal
%   sigma: noise level
%   tol: distance tolerance (specifies the convergnece tol of EM)
%   max_iter: the maximum number of iterations for EM
%output:
%   x: estimated signal
%   p: estimated shift pmf
%   time_cost: EM computation time

if ~exist('max_iter','var') || isempty(max_iter)
    max_iter = 5e5;
    if ~exist('tol','var') || isempty(tol)
        tol = 1e-5;
    end
end

[m,n] = size(obs);

% initializing x and p randomly
x = rand(d,1);
p = rand(d,1);
p = p/sum(p);

not_conv = 1;
iter = 1;

% iterating over E-step and M-step
tic
while not_conv && (iter<max_iter)
    x_prev = x;
    x_obs = zeros(m,d);
    
    % E-step
    for i = 0:d-1
        tmp =  circshift(x, -(i));
        x_obs(:, i+1) = tmp(1:m);
    end;
    x_obs = repmat(x_obs,[1,1,n]);
    y_obs = repmat(permute(obs,[1,3,2]),[1,d]);
    temp_sub = sum((y_obs-x_obs).^2,1);
    temp_sub = reshape(temp_sub,[size(temp_sub,2),size(temp_sub,3)]);
    temp = exp(-(1/(2*sigma^2))*temp_sub);
    p_rep = repmat(p,[1,n]);
    r = (temp.*p_rep)./repmat(sum(temp.*p_rep,1),[d,1]);
    
    % M-step
    flip_y = obs;
    p = sum(r,2)/n;
    for k = 0:d-1
        if (k-m+1)>=0
            r_temp = r(k+1:-1:k-m+1+1,:);
        else
            r_temp = [r(k+1:-1:1,:);r(end:-1:mod(k-m+1,d)+1,:)];
        end
        x(k+1) = sum(sum(r_temp.*flip_y))/sum(sum(r_temp));   
    end
    
    % align current x and previous x
    x_prev_aligned = align_to_ref(x_prev, x);
    error = norm(x - x_prev_aligned,'fro');
%     fprintf('iter = %d, error = %f \n',iter,error)
    if iter > 1
        not_conv = (abs(error)>tol);
    end
    iter = iter + 1;
end
time_cost = toc;

end