function [s] = auditoryCueIntegrator(p,s,x,y,t)
% function [s,p,t] = auditoryCueIntegrator(x,y,s,p,t)
% Simulates a moving auditory stimulus allowing for the inclusion of
% specific cues.
%
% Inputs:
%
% x:    time-course of x position
% y:    time-course of y posiition
% s:    time-course of sound wave (can be monaural or binaural)
% p:    structure containing parameters:
%
%   p.doppler:  flag to include Doppler effect (default true)
%   p.itd:      flag to include ITD cue (default true)
%   p.ild:      flag to include ILD cue using simple head shadow model
%               (default true)
%   p.inverseSquareLaw  flag to attenuate sound using inverse square law
%               (default true)
%
%   p.Fs        sampling rate of sound (default 8192)
%   p.c         speed of sound (meters/second)  (default is 345)
%   p.a         width of head (meters)          (default is 0.0875)
%   p.k         scalar 'k' which is the maximum attenuation by the head shadow in dB 
%               (default is 10 dB)
%   p.d0        distance of orignal sound (default 1 meter) used as
%               reference for inverse square law.
%
% t:    time vector  (default set uses sampling rate of sound and length of vectors)
%
% Written Summer 2013 by G.M. Boynton, University of Washington

%% deal with defaults

if ~exist('p','var')
    p = [];
end

if ~isfield(p,'Fs')
    p.Fs = 8192;  %sampling rate of sound (Hz)
end

if ~isfield(p,'c')
    p.c = 345;  %speed of sound (meters/second)
end

% Make time vector if it doesn't exist.
if ~exist('t','var')
    t = (1:length(x))'/p.Fs;
    dt = 1/p.Fs;
else
    t = t(:);
end

if ~isfield(p,'a')
    p.a = .0875;  %with of head (meters)
end

if ~isfield(p,'d0')
    p.d0 = 1; %distance of original sound source (meters)
end

if ~isfield(p,'k')
    p.k = 10;
end

if ~isfield(p,'doppler')
    p.doppler = true;
end

if ~isfield(p,'itd')
    p.itd = true;
end

if ~isfield(p,'inverseSquareLaw')
    p.inverseSquareLaw =true;
end

if ~isfield(p,'ild')
    p.ild = true;
end

%% Other initial stuff

% if s is monaural, make it binaural
if isrow(s) || iscolumn(s)
    s = repmat(s(:),1,2);
end

% be sure x and y are column vectors
x = x(:);
y = y(:);

%distance from observer over time
d = sqrt(x.^2+y.^2);

%% Doppler effect

if p.doppler
    
    v = diff(d)/dt;  %velocity over time
    f = p.c./(p.c+v);  %Doppler shift over time
    
    sampDt = dt*ones(size(f)).*f; %resampled time vector
    sampT = [0;cumsum(sampDt)];
    
    s = interp1(t,s,sampT,'spline');  
end

%% ITD

if p.itd
    itd  = (sqrt((x+p.a).^2+y.^2)-sqrt((x-p.a).^2+y.^2))/p.c;
    
    %resample with delays
    sL = interp1(t,s(:,1),t-itd/2,'spline');
    sR = interp1(t,s(:,2),t+itd/2,'spline');
    
    s = [sL,sR];
end


%% inverse square law
if p.inverseSquareLaw
    s = s.*repmat((p.d0./d).^2,1,2);
end

%% ILD

if p.ild
    %simple cosine head model
    theta = atan2(x,y);
    
    %Attenuation in dB as a function of sound source
    aL = p.k*(sin(theta)+1)/2;
    aR = p.k*(sin(theta+pi)+1)/2;
    
    %Attenuation factor converted from dB
    attFacL = 10.^(-aL/20);
    attFacR = 10.^(-aR/20);
    
    s(:,1) = s(:,1).*attFacL;
    s(:,2) = s(:,2).*attFacR;
end

s = min(max(s,-1),1);




