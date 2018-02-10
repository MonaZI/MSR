function [C_denoised,T_denoised] = C_T_denoiser(mean_sig,C,T,sigma)

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