function [s] = RiseFall(p,s)

% time vectors
% n_samples = p.dur*p.Fs;
n_samples = length(s);
n_RF_samples = p.RiseFallDur*p.Fs;

t = 0:n_RF_samples-1;

% determine rise and fall rate
filter_min = 0; 
filter_max = 1; 

rate = abs(filter_max - filter_min) / n_RF_samples; 

% create an envelope
t_envelope = ones(n_samples,1); 
rise = t.*rate;
fall = fliplr(rise);
t_envelope(1:n_RF_samples) = rise; 
t_envelope(n_samples-n_RF_samples+1:end) = fall;

% t_envelope = t_envelope';

% apply envelope
s = s.* repmat(t_envelope,1,size(s,2));

end