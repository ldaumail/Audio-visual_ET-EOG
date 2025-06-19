function auditory_visualsync_LD(studyName, subject, session, debug)
%Edited by Loic Daumail on 5/6/2025
% subject = 'sub-00',
% session = 1;
% debug = 1; 
% studyName = 'ADA';
global ex window screenNumber edfFile el 

ex.version = 'v01';
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

% Bring the Command Window to the front if it is already open
if ~IsOctave; commandwindow; end

% Psychtoolbox setup
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 0);


% --- AUDIO SETUP ---
InitializePsychSound(1);

% --- DISPLAY SETUP  ---
ex.viewingDist = 57; % Subject to screen distance, can adjust (cm)
ex.screenWidth = 47.5; % Screen width (cm)
screenNumber=max(Screen('Screens'));
ex.backgroundColor = [128 128 128];
if debug == 1
    ET = 0;
    ex.screenParams = SetResolution(max(Screen('Screens')),1920,1080,60);
    ex.gammaCorrection = 0;

else
    ET = 1;
    ex.screenParams = SetResolution(max(Screen('Screens')),1920,1080,60);
    ex.gammaCorrection = 1;
end

% --- KEYBOARD ----
[keyboardIndices, productNames, ~] = GetKeyboardIndices ;
deviceNumber = keyboardIndices(1);
keys_allowed = {'f', 'j'};  % making sure f and j are the only allowed keys
responseKeys = zeros(1,256);
responseKeys(KbName('f'))=1; % Key for leftward motion
responseKeys(KbName('j'))=1; % Key for rightward motion

% ---- STIMULI ---
ex.nrchannels = 2;
ex.Fs = 44100;
ex.pahandle = PsychPortAudio('Open', [], 1, 1, ex.Fs, ex.nrchannels);

% Setup MakeAuditoryNoise
ex.p.Fs = ex.Fs; % since Fs is already made
ex.p.noiseDur = 1; % Duration of noise in seconds -> might have to adjust down to 0.5
ex.p.filterWidth = 100; % Width of the filter -> need to adjust most likely
ex.p.noiseType = 'Gaussian'; % I choose gaussian even though example chooses notch
ex.Fcs = [150, 500];
ex.p.Fc = ex.Fcs(1,1);
ex.maleTone = MakeAuditoryNoise(ex.p);
ex.p.Fc = ex.Fcs(1,2);
ex.femaleTone = MakeAuditoryNoise(ex.p);

%Set up motion trajectory
ex.trj.rightStartxy = [-1, 0]; % -1 to 1 is left to right
ex.trj.rightEndxy = [1, 0];
ex.trj.leftStartxy = [1, 0]; % 1 to -1 is right to left
ex.trj.leftEndxy = [-1, 0];

ex.trj.Fs = ex.Fs;
ex.trj.noiseDur = 1;
ex.trj.doppler = true;
ex.trj.itd = true;
ex.trj.ild = true;
ex.trj.inverseSquareLaw = true;
ex.trj.orbital = true;
% MakeTrajectory
%Rightward
ex.trj.startxy = ex.trj.rightStartxy; % -1 to 1 is left to right
ex.trj.endxy = ex.trj.rightEndxy;
[rightx, righty] = MakeTrajectory(ex.trj);

%Leftward
ex.trj.startxy = ex.trj.leftStartxy; % 1 to -1 is right to left
ex.trj.endxy = ex.trj.leftEndxy;
[leftx,lefty] = MakeTrajectory(ex.trj);


%ex.ppd = pi* rect(3) / (atan(ex.screenWidth/ex.viewingDist/2)) / 360;



% --- EXPERIMENT DESIGN ---

ex.conds = {'LowLeftward','LowRightward','HighLeftward', 'HighRightward'}; %Low = low pitch tone, high = High pitch tone
ex.numConds = length(ex.conds);
ex.repsPerRun = 3;
ex.condShuffle = [];
for i =1:ex.repsPerRun
    ex.condShuffle = [ex.condShuffle, Shuffle([1:ex.numConds])];
end
for i =1:ex.numConds
    conditions(i).name = ex.conds(i);
    conditions(i).startTimes = [];
end

ex.flipsPerSec = 60;  % 60;         % number of phase changes we want from the visual stimulus, and thus the number of times we want to change visual stimulation on the screen
ex.flipWin = 1/ex.flipsPerSec;         % in seconds then actually in 1 sec the stimuli will change 12 times
ex.trialFlips = (0:ex.flipWin:1);
ex.trialFlips = ex.trialFlips(1:end-1);
frames = ex.flipsPerSec; % frames
ex.xval = linspace(-1, 1, frames); %Normalized dot location change vector

% ---- Final stimuli -----
ex.finalstimulus = nan(ex.Fs, ex.nrchannels, ex.numConds);
ex.finalstimulus(:,:,1) = auditoryCueIntegrator(ex.p, ex.maleTone, rightx, righty);
ex.finalstimulus(:,:,2) = auditoryCueIntegrator(ex.p, ex.femaleTone, rightx, righty);
ex.finalstimulus(:,:,3) = auditoryCueIntegrator(ex.p, ex.maleTone, leftx, lefty);
ex.finalstimulus(:,:,4) = auditoryCueIntegrator(ex.p, ex.femaleTone, leftx, lefty);
% Target dot
% Black or White dot (randomize based on trial)
blackwhiteArray = [ones(1,length(ex.condShuffle)/2), zeros(1,length(ex.condShuffle)/2)];
ex.dotColor = blackwhiteArray(randperm(length(blackwhiteArray))) * 255;

% ------ OPEN PSYCHTOOLBOX DISPLAY SCREEN ----
Priority(1); %Set PC to allocate max CPU and interrupt things that run in
%background if timing is critical
%[window, windowRect]=Screen('OpenWindow',0, [128 128 128],[100 100 600 400],[],[],[],[],kPsychNeed32BPCFloat);
[window, windowRect]=Screen('OpenWindow',screenNumber,ex.backgroundColor,[],[],[],[],[],kPsychNeed32BPCFloat);
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
ex.resolution = [screenXpixels, screenYpixels];
Screen('BlendFunction', window, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% --- Timing ---
frameInt = Screen('GetFlipInterval',window);
slack = frameInt/2;
frameRate =  1/frameInt;%Screen('NominalFrameRate',w);

% Target dot parameter

ex.visual_angle = 5; % degrees
ex.dot_displacement = angle2pix(ex, ex.visual_angle); % uses angle2pix to convert degrees to pixels
ex.dotRadius = 10; % dot radius, can adjust
% --- EYELINK PARAMS ---

if ET
    dummymode = 0;
    EyelinkSetup(dummymode);
    % Put tracker in idle/offline mode before recording. Eyelink('SetOfflineMode') is recommended
    % however if Eyelink('Command', 'set_idle_mode') is used allow 50ms before recording as shown in the commented code:
    % Eyelink('Command', 'set_idle_mode');% Put tracker in idle/offline mode before recording
    % WaitSecs(0.05); % Allow some time for transition
    Eyelink('SetOfflineMode');% Put tracker in idle/offline mode before recording
    Eyelink('StartRecording'); % Start tracker recording
    WaitSecs(0.1); % Allow some time to record a few samples before presenting first stimulus
end

% --- INITIALIZE EXPT ---
cnt = 0; %stim onset/ stim offset count
t = 0;
tLength = ex.trj.noiseDur;

% --- INSTRUCTIONS ---
DrawFormattedText(window,'Follow the drifting fixation dot and indicate the motion direction with the following keys: \n\n Press "F" for leftward motion, and "J" for rightward motion \n\n Press Space to start'... % :  '...
    ,'center', 'center',[0 0 0]);
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
    KbQueueStart();
    % Generate audio_stim
    if contains(condName, 'Low') && contains(condName, 'Rightward') %ex.sorted_rep_emat(trial_number, 2) == 1 % lets define male tone to be low frequency
       finalstimulus = ex.finalstimulus(:,:,1);
    elseif contains(condName, 'High') && contains(condName, 'Rightward') % lets do female tone
       finalstimulus = ex.finalstimulus(:,:,2); % definitely more than the average for the female tone but ensures significant difference
    elseif contains(condName, 'Low') && contains(condName, 'Leftward') %ex.sorted_rep_emat(trial_number, 2) == 1 % lets define male tone to be low frequency
       finalstimulus = ex.finalstimulus(:,:,3);
    elseif contains(condName, 'High') && contains(condName, 'Leftward') % lets do female tone
       finalstimulus = ex.finalstimulus(:,:,4); % definitely more than the average for the female tone but ensures significant difference
    end

    cnt = cnt+1;

    % % --- Generate TTL Pulse at trial onset ---
    % s = serialport('COM3', 115200);
    % set(s, 'DataBits', 8, 'Parity', 'none', 'StopBits', 1, 'FlowControl', 'none'); %'BaudRate', 115200
    % configureTerminator(s, 'CR/LF');
    % writeline(s,'RR');
    % pause(1);
    % a = 1;
    % fprintf(s, '%02X', a);
    % % --- end TTL Pulse
    % a = 0;
    % fprintf(s, '%02X', a);
    % fclose(s);
    % delete(s)
    % clear s
    % Eyelink('Message', 'TTL_Pulse');

    % Play the audio
    PsychPortAudio('FillBuffer', ex.pahandle, finalstimulus');
    PsychPortAudio('Start', ex.pahandle, 1, 0, 1);

    % attempting to match dot direction to trial direction, so audio as
    % well?
    if contains(condName, 'Rightward') %ex.sorted_rep_emat(trial_number, 3) == 1 % Left to Right
        startPos = screenXpixels / 2 - ex.dot_displacement; % displacement comes from visual angle conversion
        endPos = screenXpixels / 2 + ex.dot_displacement;
    elseif contains(condName, 'Leftward') % Right to Left
        startPos = screenXpixels / 2 + ex.dot_displacement;
        endPos = screenXpixels / 2 - ex.dot_displacement;
    end

    for i = 1:length(ex.xval)
        % Calculate dot position in pixels based on audio trajectory (but
        % manually.. might have to change lin space if we adjust audio x, y
        dotXpos = startPos + (endPos - startPos) * (ex.xval(i) + 1) / 2;
        dotYpos = screenYpixels / 2;
        % Render dot
        Screen('FillOval', window, ex.dotColor(t), [dotXpos - ex.dotRadius, dotYpos - ex.dotRadius, dotXpos + ex.dotRadius, dotYpos + ex.dotRadius]);
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
            if ET == 1 && i == tLength*ex.flipsPerSec+1
                Eyelink('Message', 'STIM_OFFSET');

            end

        end
    end
    % Stop Audio
    PsychPortAudio('Stop', ex.pahandle);
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
            % Store response in response array
            break
            pressed = 0;
        else
            key_input = '';
        end

    end
    KbQueueStop(); %stop recording before pressing space
    response_array{t, 2} = key_input;

    if (cnt/2 == 1) % && GetSecs-time >= 1)
        DrawFormattedText(window,'Pressed "Space" whenever you feel ready.'... %
            ,'center', 'center',[0 0 0]);
        Screen(window, 'Flip', 0);
        [~,~,~] =KbWait(deviceNumber,2);
        cnt = 0;
    end
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
                fprintf('Receiving data file ''%s.edf''\n', edfFile); % Print some text in Matlab's Command Window

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
                fprintf('Data file ''%s.edf'' can be found in ''%s''\n', edfFile, pwd);
            else
                fprintf('No EDF file saved in Dummy mode\n');
            end
        catch % Catch a file-transfer error and print some text in Matlab's Command Window
            fprintf('Problem receiving data file ''%s''\n', edfFile);
            psychrethrow(psychlasterror);
        end
    end
end