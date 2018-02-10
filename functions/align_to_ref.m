function aligned_sig = align_to_ref(x, x_ref)
% Aligning the signal x to the reference signal x_ref
%input:
%       x: the signal to be aligned
%       x_ref: the reference signal
%output:
%       aligned_sig: aligned signal

x_fft = fft(x);
x_ref_fft = fft(x_ref);
[ ~, id ] = max(real(fft(x_fft.*conj(x_ref_fft))));
aligned_sig = circshift(x, id-1);

end
