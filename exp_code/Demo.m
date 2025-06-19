clear all;
close all;
 
%% make a pure tone
p.Fc = 400;             % hz
p.Fs = 44100;           % sampling rate samples/sec
p.dur = 1;              % duration in sec

s = MakePureTone(p);    


%% make auditory noise - Gaussian
p.Fs = 44100;
p.dur = 1;
p.noiseType = 'Gaussian';
p.Fc = 1000;             % center frequency of band-pass filter in hz
p.width = 200;          % width of band-pass filter in hz         

s = MakeAuditoryNoise(p);


%% make auditory noise - Notch 
p.Fs = 44100;
p.dur = 1;
p.noiseType = 'Notch';
p.lowCutoff = 100;
p.highCutoff = 1000;

s = MakeAuditoryNoise(p);


%% generate trajectory
p.startxy = [-10 5];     % [x y] coordinates in meters
p.endxy = [10 5];        % [x y] coordinates in meters
p.orbital = 0;           % 0 or 1 (0 for frontoparallel or looming)

[xx, yy] = MakeTrajectory(p);


%% auditory cue integrator 
p.doppler   = 1;
p.itd       = 1;
p.ild       = 1; 
p.inverseSquareLaw = 1;

[s] = auditoryCueIntegrator(p, s, xx, yy);


%% pinna effects  ** needs to be checked ** 
yyR = pinna(0, 0, p.Fs, s(:,1));
yyL = pinna(0, 0, p.Fs, s(:,2)); 
s = [yyL yyR];


%% temporal envelope (e.g., Rise and Fall)
p.RiseFallDur = 0.2;    % rise and fall time (sec)

s = RiseFall(p,s);

% s(end-1500:end,:) = 0; % to avoid offset pop

%% visualize (if you'd like)
subplot(1,2,1); 
plot1 = plot(s);
xlabel('time'); ylabel('amplitude');
legend('Left','Right','Location','southeast');
axis square;

subplot(1,2,2);
plot(xx,yy); hold on;
plot(xx(1),yy(1),'ro');
plot(xx(end),yy(end),'bo');
plot(0,0,'k+');
xlim([-20 20]); ylim([-20 20]); axis square;
legend('trajectory','start','end','observer','Location','southeast');
xlabel('X (LeftRight)'); ylabel('Y (Depth)');


%% play

Snd('Play', s',p.Fs);

