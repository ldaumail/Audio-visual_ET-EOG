function auditory_visualsync_LD_v4(studyName, subject, session, debug)
%Edited by Loic Daumail on 5/6/2025
% subject = 'sub-00',
% session = 1;
% debug = 1; 
% studyName = 'ADA';

global ex window screenNumber edfFile 

ex.version = 'v04';
% --- DISPLAY SETUP  ---
ex.viewingDist = 57; % Subject to screen distance, can adjust (cm)
ex.screenWidth = 47.5; % Screen width (cm)
ex.backgroundColor = [128 128 128];
if debug == 1
    ET = 0;
    ex.screenParams = SetResolution(max(Screen('Screens')),3024,1964,120);%3024,1964,120
    ex.gammaCorrection = 0;

else
    ET = 1;
    ex.screenParams = SetResolution(max(Screen('Screens')),1920,1080,60);
    ex.gammaCorrection = 1;
end

% --- KEYBOARD ----
[keyboardIndices, productNames, ~] = GetKeyboardIndices ;
deviceNumber = keyboardIndices(1);
responseKeys = zeros(1,256);
responseKeys(KbName('f'))=1; % Key for leftward motion
responseKeys(KbName('j'))=1; % Key for rightward motion

Screen('Preference', 'SkipSyncTests', 0);
% Bring the Command Window to the front if it is already open
% if ~IsOctave; commandwindow; end
ex.runNum = input('Run number :');
screenNumber=max(Screen('Screens'));

ex.subject = subject;
ex.session = session;
ex.studyName = studyName;

%%%% set-up rand
ex.startTime = clock;
rng(sum(100*ex.startTime));
ex.rand = rng;

%%%% files and things
ex.root = pwd;
ex.date = datestr(now,30);

% Psychtoolbox setup
PsychDefaultSetup(2);

% --- AUDIO SETUP ---
InitializePsychSound(1);


% --- EXPERIMENT DESIGN ---

ex.conds = {'LowLeftwardVol','LowRightwardVol',...
            'HighLeftwardVol','HighRightwardVol',...
            'LowLeftwardInvol','LowRightwardInvol',...
            'HighLeftwardInvol','HighRightwardInvol',...
            'VisLeftward', 'VisRightward',...
            }; %Low = low pitch tone, high = High pitch tone
ex.numConds = length(ex.conds);
ex.repsPerRun = 3;
ex.condShuffle = [];
for i =1:ex.repsPerRun 
    ex.condShuffle = [ex.condShuffle, Shuffle([1:4])];
end
for i =1:ex.repsPerRun 
    ex.condShuffle = [ex.condShuffle, Shuffle([5:8])];
end
for i =1:ex.repsPerRun 
    ex.condShuffle = [ex.condShuffle, Shuffle([9:10])];
end
for i =1:ex.numConds
    conditions(i).name = ex.conds(i);
    conditions(i).startTimes = [];
end

%--- Timing ----
ex.dot_displacement_deg = 20; % degrees from center (± 20 is 40 total)
ex.driftingSpeeds = randi([5,20], length(ex.condShuffle),1); %take random drifting speeds between 5 and 20 deg/s 
ex.trialdur = ex.dot_displacement_deg*2./ex.driftingSpeeds;

% ---- STIMULI ---
ex.nrchannels = 2;
ex.Fs = 44100;
ex.pahandle = PsychPortAudio('Open', [], 1, 1, ex.Fs, ex.nrchannels);

% Setup MakeAuditoryNoise
ex.p.Fs = ex.Fs; % since Fs is already made
ex.p.filterWidth = 100; % Width of the filter -> need to adjust most likely
ex.p.noiseType = 'Notch'; 
ex.lowCutofflow = 1000;
ex.highCutofflow = 2000;
ex.lowCutoffhigh = 8000;
ex.highCutoffhigh = 9000;

ex.lowTone = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds)); %5 being the lowest possible speed, so longest possible trial duration
ex.highTone = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds));

for i =1:length(ex.driftingSpeeds)
    ex.p.noiseDur = ex.trialdur(i); % Duration of noise in seconds -> might have to adjust down to 0.5
    ex.p.lowCutoff = ex.lowCutofflow;
    ex.p.highCutoff = ex.highCutofflow;
    ex.lowTone(1:length(MakeAuditoryNoise(ex.p)),i) = MakeAuditoryNoise(ex.p);
    ex.p.lowCutoff = ex.lowCutoffhigh;
    ex.p.highCutoff = ex.highCutoffhigh;
    ex.highTone(1:length(MakeAuditoryNoise(ex.p)),i) = MakeAuditoryNoise(ex.p);
end

%---- Set up auditory motion trajectory ---%
ex.trj.rightStartxy = [-1*ex.screenWidth/200, 0]; % -1 to 1 is left to right, units must be in meters
ex.trj.rightEndxy = [1*ex.screenWidth/200, 0];
ex.trj.leftStartxy = [1*ex.screenWidth/200, 0]; % 1 to -1 is right to left
ex.trj.leftEndxy = [-1*ex.screenWidth/200, 0];
ex.trj.orbital = 0;
ex.trj.Fs = ex.Fs;

% MakeTrajectory
rightx = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds)); %5 being the lowest possible speed, so longest possible trial duration
righty = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds)); %5 being the lowest possible speed, so longest possible trial duration
leftx = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds)); %5 being the lowest possible speed, so longest possible trial duration
lefty = nan(round(ex.Fs*ex.dot_displacement_deg*2/5),length(ex.driftingSpeeds)); %5 being the lowest possible speed, so longest possible trial duration

for i =1:length(ex.driftingSpeeds)
    ex.trj.noiseDur = ex.trialdur(i);
    %Rightward
    ex.trj.startxy = ex.trj.rightStartxy; % -1 to 1 is left to right
    ex.trj.endxy = ex.trj.rightEndxy;
    [rightx(1:length(MakeTrajectory(ex.trj)),i), righty(1:length(MakeTrajectory(ex.trj)),i)] = MakeTrajectory(ex.trj);

    %Leftward
    ex.trj.startxy = ex.trj.leftStartxy; % 1 to -1 is right to left
    ex.trj.endxy = ex.trj.leftEndxy;
    [leftx(1:length(MakeTrajectory(ex.trj)),i),lefty(1:length(MakeTrajectory(ex.trj)),i)] = MakeTrajectory(ex.trj);
end


% ---- Final stimuli -----

%-- Set Auditory cue integrator params
ex.p.doppler = true;
ex.p.itd = true;
ex.p.ild = true;
ex.p.inverseSquareLaw = true;
ex.p.orbital = ex.trj.orbital;

%preallocate sound stimuli with a zeros matrix to avoit warning from
%interpolator:
ex.finalaudiostimulus = nan(ex.Fs*ex.dot_displacement_deg*2/5, ex.nrchannels, length(ex.driftingSpeeds), 4); %5 being the lowest possible speed, so longest possible trial duration
ex.p.RiseFallDur = 0.2;    % rise and fall time (sec)
for i =1:length(ex.driftingSpeeds)
    ex.finalaudiostimulus(1:length(rightx(~isnan(rightx(:,i)),i)),:,i,1) = RiseFall(ex.p, auditoryCueIntegrator(ex.p, ex.lowTone(~isnan(rightx(:,i)),i), rightx(~isnan(rightx(:,i)),i), righty(~isnan(righty(:,i)),i)));
    ex.finalaudiostimulus(1:length(rightx(~isnan(rightx(:,i)),i)),:,i,2) = RiseFall(ex.p, auditoryCueIntegrator(ex.p, ex.highTone(~isnan(rightx(:,i)),i), rightx(~isnan(rightx(:,i)),i), righty(~isnan(righty(:,i)),i)));
    ex.finalaudiostimulus(1:length(leftx(~isnan(leftx(:,i)),i)),:,i,3) = RiseFall(ex.p, auditoryCueIntegrator(ex.p, ex.lowTone(~isnan(leftx(:,i)),i), leftx(~isnan(leftx(:,i)),i), lefty(~isnan(lefty(:,i)),i)));
    ex.finalaudiostimulus(1:length(leftx(~isnan(leftx(:,i)),i)),:,i,4) = RiseFall(ex.p, auditoryCueIntegrator(ex.p, ex.highTone(~isnan(leftx(:,i)),i), leftx(~isnan(leftx(:,i)),i), lefty(~isnan(lefty(:,i)),i)));
end
% Target dot
% Black or White dot (randomize based on trial)
blackwhiteArray = [ones(1,length(ex.condShuffle)/2), zeros(1,length(ex.condShuffle)/2)];
ex.dotColor = blackwhiteArray(randperm(length(blackwhiteArray))) * 255;

% ------ OPEN PSYCHTOOLBOX DISPLAY SCREEN ----
HideCursor;
Priority(1); %Set PC to allocate max CPU and interrupt things that run in
%background if timing is critical
if debug
    %[window, windowRect]=Screen('OpenWindow',0, [128 128 128],[100 100 600 400],[],[],[],[],kPsychNeed32BPCFloat);
    [window, windowRect]=Screen('OpenWindow',screenNumber,ex.backgroundColor,[],[],[],[],[],kPsychNeed32BPCFloat);

else
    [window, windowRect]=Screen('OpenWindow',screenNumber,ex.backgroundColor,[],[],[],[],[],kPsychNeed32BPCFloat);
end
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
ex.resolution = [screenXpixels, screenYpixels];
Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% --- Timing ---
frameInt = Screen('GetFlipInterval',window);
slack = frameInt/2;
% frameRate =  1/frameInt;%Screen('NominalFrameRate',w);

% Target dot parameters
ex.ppd = pi*windowRect(3) / (atan(ex.screenWidth/ex.viewingDist/2)) / 360;
ex.dot_displacement = ex.dot_displacement_deg*ex.ppd;%angle2pix(ex, ex.dot_displacement_deg); % uses angle2pix to convert degrees to pixels
ex.dotRadius_deg = 0.25; % dot radius
ex.dotRadius = ex.dotRadius_deg*ex.ppd;

% Target dot drift positions
ex.flipsPerSec = 60;  % 60;         % number of phase changes we want from the visual stimulus, and thus the number of times we want to change visual stimulation on the screen
ex.flipWin = 1/ex.flipsPerSec;         % in seconds then actually in 1 sec the stimuli will change 12 times
ex.trialFlips = nan(ex.flipsPerSec*(ex.dot_displacement_deg*2/5+1), length(ex.trialdur)); %5 being the lowest possible speed, so longest possible trial duration add 1 s for fixation time
for i =1:length(ex.driftingSpeeds)
    ex.trialFlips(1:length((0:ex.flipWin:1+ex.trialdur(i))),i) = (0:ex.flipWin:1+ex.trialdur(i));
    % ex.trialFlips = ex.trialFlips(1:length((0:ex.flipWin:1+ex.trialdur(i)))-1,i);
end

frames = ex.flipsPerSec.*ex.trialdur; % frames
ex.xval = nan(ex.flipsPerSec*(ex.dot_displacement_deg*2/5), length(ex.trialdur));
for i =1:length(ex.driftingSpeeds)
    ex.xval(1:length(linspace(-1, 1, frames(i))),i) = linspace(-1, 1, frames(i)); %Normalized dot location change vector
    ex.xval(length(linspace(-1, 1, frames(i)))+1:end,i) = NaN;
end
ex.leftPos = screenXpixels / 2 - ex.dot_displacement; % displacement comes from visual angle conversion
ex.rightPos = screenXpixels / 2 + ex.dot_displacement;
ex.fixDur = 1;

ex.dotxPos = nan(ex.flipsPerSec*(ex.dot_displacement_deg*2/5+1),length(ex.trialdur), 2);
for i =1:length(ex.trialdur)
    ex.dotxPos(:,i,1) = [repmat(ex.leftPos + (ex.rightPos - ex.leftPos) * (ex.xval(1,i) + 1) / 2, ex.fixDur*ex.flipsPerSec,1); ex.leftPos + (ex.rightPos - ex.leftPos) * (ex.xval(:,i) + 1) / 2]; % still dot+ Rightward drift
    ex.dotxPos(:,i,2) = [repmat(ex.rightPos + (ex.leftPos - ex.rightPos) * (ex.xval(1,i) + 1) / 2, ex.fixDur*ex.flipsPerSec,1); ex.rightPos + (ex.leftPos - ex.rightPos) * (ex.xval(:,i) + 1) / 2]; % still dot+ Leftward drift
end

% --- EYELINK PARAMS ---

if ET
    dummymode = 0;
    EyelinkSetup(dummymode); %run calibration, etc
    
    % Put tracker in idle/offline mode before recording. Eyelink('SetOfflineMode') is recommended
    % however if Eyelink('Command', 'set_idle_mode') is used allow 50ms before recording as shown in the commented code:
    % Eyelink('Command', 'set_idle_mode');% Put tracker in idle/offline mode before recording
    % WaitSecs(0.05); % Allow some time for transition
    Eyelink('SetOfflineMode');% Put tracker in idle/offline mode before recording
    Eyelink('StartRecording'); % Start tracker recording
end
%Sham audio stimulus to avoid delay on first trial for sound activation
PsychPortAudio('FillBuffer', ex.pahandle, ex.finalaudiostimulus(:,:,1)');
PsychPortAudio('Start', ex.pahandle, 1, 0, 1);
% WaitSecs(2); % Allow some time to record a few samples before presenting first stimulus

% --- INITIALIZE EXPT ---
cnt = 0; %stim onset/ stim offset count
t = 0;

% --- INSTRUCTIONS ---
DrawFormattedText(window,'Track the motion signal and indicate the motion direction with the following keys: \n\n Press "F" for leftward motion, and "J" for rightward motion. \n\n For some tone trials, you will be prompted to track the motion with your gaze. \n\n When not prompted, you can keep your gaze still unless a visual cue is shown on the screen. \n\n Press Space to start','center', 'center',[0 0 0]);
Screen(window, 'Flip', 0);
KbTriggerWait(KbName('Space'), deviceNumber);
if t == 0
    ex.startRun = GetSecs();
    if ET == 1
        % % --- Generate TTL Pulse at task onset ---
        s = serialport('COM3', 115200);
        set(s, 'DataBits', 8, 'Parity', 'none', 'StopBits', 1, 'FlowControl', 'none'); %'BaudRate', 115200
        configureTerminator(s, 'CR/LF');

        % fopen(s);
        writeline(s,'RR');
        pause(1);
        a = 1;
        fprintf(s, '%02X', a);
        Eyelink('Message', 'TTL_Pulse_On');
    end
end
KbQueueCreate(deviceNumber,responseKeys);

% --- MAIN EXPERIMENT LOOP ---
for t = 1:length(ex.condShuffle) %size(ex.sorted_rep_emat, 1)
    thisCond = ex.condShuffle(t);
    condName = conditions(thisCond).name{:};
    if  contains(condName, 'Invol') && t == 1
        DrawFormattedText(window,'Part 1'... % :  '...
            ,'center', 'center',[0 0 0]);
        Screen(window, 'Flip', 0);
        WaitSecs(0.5);
    elseif contains(condName, 'Vol') && t == ex.repsPerRun*4+1
        DrawFormattedText(window,'Part 2: Track the moving tone direction with your gaze \n\n as accurately as you can'... % :  '...
            ,'center', 'center',[0 0 0]);
        Screen(window, 'Flip', 0);
        WaitSecs(0.5);
    elseif contains(condName, 'Vis') && t == ex.repsPerRun*8+1
        DrawFormattedText(window,'Part 3: Track the moving dot direction with your gaze \n\n as accurately as you can'... % :  '...
            ,'center', 'center',[0 0 0]);
        Screen(window, 'Flip', 0);
        WaitSecs(0.5);
    end
    KbQueueStart();
    % Generate audio_stim
    if contains(condName, 'Low') && contains(condName, 'Rightward') %ex.sorted_rep_emat(trial_number, 2) == 1 % lets define male tone to be low frequency
       finalstimulus = ex.finalaudiostimulus(:,:,t,1);
    elseif contains(condName, 'High') && contains(condName, 'Rightward') % lets do female tone
       finalstimulus = ex.finalaudiostimulus(:,:,t,2); % definitely more than the average for the female tone but ensures significant difference
    elseif contains(condName, 'Low') && contains(condName, 'Leftward') %ex.sorted_rep_emat(trial_number, 2) == 1 % lets define male tone to be low frequency
       finalstimulus = ex.finalaudiostimulus(:,:,t,3);
    elseif contains(condName, 'High') && contains(condName, 'Leftward') % lets do female tone
       finalstimulus = ex.finalaudiostimulus(:,:,t,4); % definitely more than the average for the female tone but ensures significant difference
    end

    cnt = cnt+1;
    

    % attempting to match dot direction to trial direction, so audio as
    % well?
    if contains(condName, 'Rightward') && contains(condName, 'Vis') %ex.sorted_rep_emat(trial_number, 3) == 1 % Left to Right
        dotXpos = ex.dotxPos(:,t,1);
    elseif contains(condName, 'Leftward') && contains(condName, 'Vis')% Right to Left
        dotXpos = ex.dotxPos(:,t,2);
    end

    for i = 1:length(ex.dotxPos(~isnan(ex.dotxPos(:,t,1)),t,1))-1 % Only run the flips of each trial
        %Play sound
        if (contains(condName, 'Low') || contains(condName, 'High')) && i == ex.fixDur*ex.flipsPerSec+1
            % Play the audio
            PsychPortAudio('FillBuffer', ex.pahandle, finalstimulus');
            PsychPortAudio('Start', ex.pahandle, 1, 0, 1);
        end
        % Calculate dot position in pixels based on audio trajectory (but
        % manually.. might have to change lin space if we adjust audio x, y
        if contains(condName, 'Vis')
            dotYpos = screenYpixels / 2;
            % Render dot
            Screen('FillOval', window, ex.dotColor(t), [dotXpos(i) - ex.dotRadius, dotYpos - ex.dotRadius, dotXpos(i) + ex.dotRadius, dotYpos + ex.dotRadius]);
        end
        if i == 1
            [VBLT, startTrial, FlipT, missed] = Screen(window, 'Flip', 0);%[VBLTimestamp StimulusOnsetTime FlipTimestamp Missed] = Screen('Flip', windowPtr [, when] [, dontclear]...
            flipTimes = startTrial;
            %%%%% send messages to Eyelink for event times
            if ET == 1
                Eyelink('Message', char(sprintf('Cond %s', condName)));
                Eyelink('Message', 'TRIALID %d', t);
                Eyelink('Message', 'STIM_ONSET');
            end
        else
            [VBLT,flipTime, FlipT, missed] = Screen(window, 'Flip',startTrial + ex.trialFlips(i) - slack); %,   %%% ex.flipTime(n,c)
            flipTimes = [flipTimes, flipTime];
            if ET == 1 && i == (ex.trialdur(t)+ex.fixDur)*ex.flipsPerSec+1
                Eyelink('Message', 'STIM_OFFSET');

            end

        end
    end
    if contains(condName, 'Low') || contains(condName, 'High')
        % Stop Audio
        PsychPortAudio('Stop', ex.pahandle);
    end
    % Clear screen after each trial
    Screen('Flip', window);
    cnt = cnt+1;

    key_input = '';
    % Check for key press
    while isempty(key_input)
        [pressed, firstPress] = KbQueueCheck();
        if (pressed == 1) && (firstPress(KbName('f')) > 0 ||(firstPress(KbName('j')) > 0))
            keyCode = find(firstPress);
            key_input = KbName(keyCode);
            pressed = 0;
            DrawFormattedText(window,'Next Trial'... % :  '...
                ,'center', 'center',[0 0 0]);
            Screen(window, 'Flip', 0);
            WaitSecs(0.3);
            break
        else
            key_input = '';
        end

    end
    KbQueueStop(); %stop recording before pressing space
     % Store response in response array
    response_array{t, 2} = key_input;

    %%%% refresh queue for next character
    KbQueueFlush();
end
% clean
KbQueueRelease();
PsychPortAudio('Close', ex.pahandle);
if ET == 1
    % % --- end TTL Pulse
    a = 0;
    fprintf(s, '%02X', a);
    fclose(s);
    delete(s)
    clear s
    Eyelink('Message', 'TTL_Pulse_Off');
    % ----
end
ex.runTime = GetSecs - ex.startRun;

%save data
savedir = fullfile(ex.root,'data',sprintf('%s_%s/',subject,ex.version));
if ~exist(savedir); mkdir(savedir); end
savename = fullfile(savedir, strcat(sprintf('/%s_audio_visual_motion_%s_date%s_fix',subject,ex.version,num2str(ex.date)), '.mat'));
save(savename,'ex','-v7.3')

%End psychtoolbox screen
Screen('CloseAll');
ShowCursor();

%% STEP 6: CLOSE EDF FILE. TRANSFER EDF COPY TO DISPLAY PC. CLOSE EYELINK CONNECTION. FINISH UP
if ET == 1
    % Put tracker in idle/offline mode before closing file. Eyelink('SetOfflineMode') is recommended.
    % However if Eyelink('Command', 'set_idle_mode') is used, allow 50ms before closing the file as shown in the commented code:
    % Eyelink('Command', 'set_idle_mode');% Put tracker in idle/offline mode
    % WaitSecs(0.05); % Allow some time for transition
    Eyelink('SetOfflineMode'); % Put tracker in idle/offline mode
    Eyelink('Command', 'clear_screen 0'); % Clear Host PC backdrop graphics at the end of the experiment
    WaitSecs(0.5); % Allow some time before closing and transferring file
    Eyelink('CloseFile'); % Close EDF file on Host PC
    % Transfer a copy of the EDF file to Display PC
    transferFile; % See transferFile function below
end
cleanup;

% Cleanup function used throughout the script above
    function cleanup
        try
            Screen('CloseAll'); % Close window if it is open
        end
        Eyelink('Shutdown'); % Close EyeLink connection
        ListenChar(0); % Restore keyboard output to Matlab
        ShowCursor; % Restore mouse cursor
        if ~IsOctave; commandwindow; end % Bring Command Window to front
    end

% Function for transferring copy of EDF file to the experiment folder on Display PC.
% Allows for optional destination path which is different from experiment folder
    function transferFile
        try
            if dummymode == 0 % If connected to EyeLink
                % Show 'Receiving data file...' text until file transfer is complete
                % Screen('FillRect', window, ex.backgroundColor); % Prepare background on backbuffer
                % Screen('DrawText', window, 'Receiving data file...', 5, height-35, 0); % Prepare text
                % Screen('Flip', window); % Present text
                fprintf('Receiving data file ''%s''\n', edfFile); % Print some text in Matlab's Command Window

                % Transfer EDF file to Host PC
                % [status =] Eyelink('ReceiveFile',['src'], ['dest'], ['dest_is_path'])
                status = Eyelink('ReceiveFile', [], savedir, 1);
                % Optionally uncomment below to change edf file name when a copy is transferred to the Display PC
                % % If <src> is omitted, tracker will send last opened data file.
                % % If <dest> is omitted, creates local file with source file name.
                % % Else, creates file using <dest> as name.  If <dest_is_path> is supplied and non-zero
                % % uses source file name but adds <dest> as directory path.
                % newName = ['Test_',char(datetime('now','TimeZone','local','Format','y_M_d_HH_mm')),'.edf'];
                % status = Eyelink('ReceiveFile', [], newName, 0);

                % Check if EDF file has been transferred successfully and print file size in Matlab's Command Window
                if status > 0
                    fprintf('EDF file size: %.1f KB\n', status/1024); % Divide file size by 1024 to convert bytes to KB
                end
                % Print transferred EDF file path in Matlab's Command Window
                fprintf('Data file ''%s'' can be found in ''%s''\n', edfFile, pwd);
            else
                fprintf('No EDF file saved in Dummy mode\n');
            end
        catch % Catch a file-transfer error and print some text in Matlab's Command Window
            fprintf('Problem receiving data file ''%s''\n', edfFile);
            psychrethrow(psychlasterror);
        end
    end
end