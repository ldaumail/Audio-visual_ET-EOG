function EyelinkSetup(dummymode)   

    global ex window screenNumber edfFile el 
    if isempty(screenNumber)
        screenNumber = max(Screen('Screens')); % Use default screen if none specified
    end
    %% STEP 1: INITIALIZE EYELINK CONNECTION; OPEN EDF FILE; GET EYELINK TRACKER VERSION
    
    % Initialize EyeLink connection (dummymode = 0) or run in "Dummy Mode" without an EyeLink connection (dummymode = 1);
    % dummymode = 0;
    
    % Optional: Set IP address of eyelink tracker computer to connect to.
    % Call this before initializing an EyeLink connection if you want to use a non-default IP address for the Host PC.
    %Eyelink('SetAddress', '10.10.10.240');
    
    EyelinkInit(dummymode); % Initialize EyeLink connection
    status = Eyelink('IsConnected');
    if status < 1 % If EyeLink not connected
        dummymode = 1; 
    end
       
    % % Open dialog box for EyeLink Data file name entry. File name up to 8 characters
    % prompt = {'Enter EDF file name (up to 8 characters)'};
    % dlg_title = 'Create EDF file';
    % def = {'demo'}; % Create a default edf file name
    % answer = inputdlg(prompt, dlg_title, 1, def); % Prompt for new EDF file name    
    % % Print some text in Matlab's Command Window if a file name has not been entered
    % if  isempty(answer)
    %     fprintf('Session cancelled by user\n')
    %     error('Session cancelled by user'); % Abort experiment (see cleanup function below)
    % end    
    % edfFile = answer{1}; % Save file name to a variable    
    % % Print some text in Matlab's Command Window if file name is longer than 8 characters
    % if length(edfFile) > 8
    %     fprintf('Filename needs to be no more than 8 characters long (letters, numbers and underscores only)\n');
    %     error('Filename needs to be no more than 8 characters long (letters, numbers and underscores only)');
    % end
    % 
    % % Open an EDF file and name it
    % failOpen = Eyelink('OpenFile', edfFile);
    % if failOpen ~= 0 % Abort if it fails to open
    %     fprintf('Cannot create EDF file %s', edfFile); % Print some text in Matlab's Command Window
    %     error('Cannot create EDF file %s', edfFile); % Print some text in Matlab's Command Window
    % end
    edfFile = sprintf('%s%d_%s.edf',ex.studyName,sscanf(ex.subject,'sub-%d'), ex.session);%sprintf('demo_fix.edf'); % overwrite the edf each time. or it would full the hard drive of eyelink computer
    Eyelink('Openfile', edfFile);

    % Get EyeLink tracker and software version
    % <ver> returns 0 if not connected
    % <versionstring> returns 'EYELINK I', 'EYELINK II x.xx', 'EYELINK CL x.xx' where 'x.xx' is the software version
    ELsoftwareVersion = 0; % Default EyeLink version in dummy mode
    [ver, versionstring] = Eyelink('GetTrackerVersion');
    if dummymode == 0 % If connected to EyeLink
        % Extract software version number. 
        [~, vnumcell] = regexp(versionstring,'.*?(\d)\.\d*?','Match','Tokens'); % Extract EL version before decimal point
        ELsoftwareVersion = str2double(vnumcell{1}{1}); % Returns 1 for EyeLink I, 2 for EyeLink II, 3/4 for EyeLink 1K, 5 for EyeLink 1KPlus, 6 for Portable Duo         
        % Print some text in Matlab's Command Window
        fprintf('Running experiment on %s version %d\n', versionstring, ver );
    end
    % Add a line of text in the EDF file to identify the current experimemt name and session. This is optional.
    % If your text starts with "RECORDED BY " it will be available in DataViewer's Inspector window by clicking
    % the EDF session node in the top panel and looking for the "Recorded By:" field in the bottom panel of the Inspector.
    preambleText = sprintf('RECORDED BY Psychtoolbox demo %s session name: %s', mfilename, edfFile);
    Eyelink('Command', 'add_file_preamble_text "%s"', preambleText);
    
    
    %% STEP 2: SELECT AVAILABLE SAMPLE/EVENT DATA
    % See EyeLinkProgrammers Guide manual > Useful EyeLink Commands > File Data Control & Link Data Control
    
    % Select which events are saved in the EDF file. Include everything just in case
    Eyelink('Command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
    % Select which events are available online for gaze-contingent experiments. Include everything just in case
    Eyelink('Command', 'link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT');
    % Select which sample data is saved in EDF file or available online. Include everything just in case
    if ELsoftwareVersion > 3  % Check tracker version and include 'HTARGET' to save head target sticker data for supported eye trackers
        Eyelink('Command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT');
        Eyelink('Command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT');
    else
        Eyelink('Command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT');
        Eyelink('Command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT');
    end
    
    %% STEP 3: OPEN GRAPHICS WINDOW
    
    % Open experiment graphics on the specified screen

    % window = Screen('OpenWindow', screenNumber, [128 128 128]); % Open graphics window
    Screen('Flip', window);
    % Return width and height of the graphics window/screen in pixels
    [width, height] = Screen('WindowSize', window);
    
    
    %% STEP 4: SET CALIBRATION SCREEN COLOURS/SOUNDS; PROVIDE WINDOW SIZE TO EYELINK HOST & DATAVIEWER; SET CALIBRATION PARAMETERS; CALIBRATE
    
    % Provide EyeLink with some defaults, which are returned in the structure "el".
    el = EyelinkInitDefaults(window);
    el.ppa_pahandle = ex.pahandle; 
    % set calibration/validation/drift-check(or drift-correct) size as well as background and target colors. 
    % It is important that this background colour is similar to that of the stimuli to prevent large luminance-based 
    % pupil size changes (which can cause a drift in the eye movement data)
    el.calibrationtargetsize = 3;% Outer target size as percentage of the screen
    el.calibrationtargetwidth = 0.7;% Inner target size as percentage of the screen
    el.backgroundcolour = [128 128 128];% RGB grey
    el.calibrationtargetcolour = [0 0 0];% RGB black
    % set "Camera Setup" instructions text colour so it is different from background colour
    el.msgfontcolour = [0 0 0];% RGB black
        
    % Use an image file instead of the default calibration bull's eye targets. 
    % Commenting out the following two lines will use default targets:
    el.calTargetType = 'image';
    el.calImageTargetFilename = [pwd '/' 'fixTarget.jpg'];
    
    % Set calibration beeps (0 = sound off, 1 = sound on)
    el.targetbeep = 1; %1;  % sound a beep when a target is presented
    el.feedbackbeep = 1; %1;  % sound a beep after calibration or drift check/correction
    
    % You must call this function to apply the changes made to the el structure above
    EyelinkUpdateDefaults(el);
    
    %Set a binocular calibration with BOTH. Else use LEFT or RIGHT
    Eyelink('command', 'active_eye = BOTH');
    
    % Set display coordinates for EyeLink data by entering left, top, right and bottom coordinates in screen pixels
    Eyelink('Command','screen_pixel_coords = %ld %ld %ld %ld', 0, 0, width-1, height-1);
    % Write DISPLAY_COORDS message to EDF file: sets display coordinates in DataViewer
    % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Pre-trial Message Commands
    Eyelink('Message', 'DISPLAY_COORDS %ld %ld %ld %ld', 0, 0, width-1, height-1);    
    % Set number of calibration/validation dots and spread: horizontal-only(H) or horizontal-vertical(HV) as H3, HV3, HV5, HV9 or HV13
    Eyelink('Command', 'calibration_type = HV9'); % horizontal-vertical 9-points
    % Allow a supported EyeLink Host PC button box to accept calibration or drift-check/correction targets via button 5
    Eyelink('Command', 'button_function 5 "accept_target_fixation"');
    % Hide mouse cursor
    HideCursor(screenNumber);
    % Start listening for keyboard input. Suppress keypresses to Matlab windows.
    ListenChar(-1);
    Eyelink('Command', 'clear_screen 0'); % Clear Host PC display from any previus drawing
    % Put EyeLink Host PC in Camera Setup mode for participant setup/calibration
    EyelinkDoTrackerSetup(el);
end