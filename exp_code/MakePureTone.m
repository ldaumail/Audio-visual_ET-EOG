function [tone] = MakePureTone(p)
%Fc is the frequency of the pure tone (Hz)
%Fs is the sampling rate of the audio
%dur is the duration of the tone (s)
tone = cos(2*pi*p.Fc*((0:(1/p.Fs):p.dur-(1/p.Fs))'+p.phi));
%sound(tone)