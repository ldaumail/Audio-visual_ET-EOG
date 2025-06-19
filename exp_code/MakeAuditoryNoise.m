function [noise] = MakeAuditoryNoise(p)

% Time vector
t = linspace(0,p.noiseDur,p.noiseDur*p.Fs)';

% White noise
y = randn(size(t));

% Fourier Transform
F = complex2real(fft(y),t);

% Band-pass filter in the frequency domain
if strcmp(p.noiseType, 'Gaussian')
    
    Filter = exp(-(F.freq-p.Fc).^2/p.filterWidth^2);     
    
elseif strcmp(p.noiseType, 'Notch')
    
    Filter = double(F.freq > p.lowCutoff-1 & F.freq < p.highCutoff+1); 
    
end

F.amp = F.amp.*Filter;

% Inverse Fourier Transform:
noise = real(ifft(real2complex(F)));

end