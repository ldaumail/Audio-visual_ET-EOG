%% make auditory noise - Gaussian
p.Fs = 44100;
p.dur = 1;
p.noiseType = 'Gaussian';
p.Fc = 1000;             % center frequency of band-pass filter in hz
p.width = 200;          % width of band-pass filter in hz         

s = MakeAuditoryNoise(p);