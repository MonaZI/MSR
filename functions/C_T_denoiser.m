function [C_denoised,T_denoised] = C_T_denoiser(mean_sig, C, T, sigma)
%Denoising the estimated second and third order invariants
%input: 
%       mean_sig: estimated mean
%       C_est: estimated 2nd order correlation
%       T_est: estimated 3rd oredr correlation (bispectrum)
%       sigma: noise level
%output:
%       C_denoised: denoised 2nd order invariant
%       T_denoised: denoised 3rd order invariant
%
%February 2018
%paper: http://arxiv.org/abs/1802.08950
%code: https://github.com/MonaZI/MSR

m = size(C,1);
C_denoised = C - (sigma^2) * eye(m);
T_denoised = T;

for k1 = 1:m
    for k2 = 1:m
        for k3 = 1:m
            
            temp = (sigma^2)*(mean_sig(k1)*(k2==k3)+mean_sig(k2)*(k1==k3)+mean_sig(k3)*(k1==k2));
            T_denoised(k1,k2,k3) = T_denoised(k1,k2,k3)-temp;        
            
        end
    end
end

end
