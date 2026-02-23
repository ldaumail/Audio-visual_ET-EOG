%This script was written by Loic Daumail
%Started on 10/27/2025


%% Load Eyelink EDF data, store blink free data in structure organized by participant, condition, trial, along with blink info
sample_rate = 2000;
version = '05';
names = {'sub-08'};

for i =1:length(names)
    name = names{i};

    %if contains(name, s_f_names) %check if subject had the slow condition first or the fast one
    folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
        sprintf('v%s/%s_v%s',version, name,version), '/');

    fileNames = dir(folderDir);
    fileNames = string({fileNames(contains({fileNames.name},'AUV')| contains({fileNames.name},'.mat')).name});

    edfName = char(strcat(folderDir,fileNames(contains(fileNames,'.edf'))));
    edfDat = Edf2Mat(edfName);
    exptName1 = strcat(folderDir,fileNames(contains(fileNames,'part1.mat')));
    exptDat1 = load(exptName1);
    exptName2 = strcat(folderDir,fileNames(contains(fileNames,'part2.mat')));
    exptDat2 = load(exptName2);
    exptName3 = strcat(folderDir,fileNames(contains(fileNames,'part3.mat')));
    exptDat3 = load(exptName3);
    
    %baseline correct data
    xbsl = exptDat1.ex.screenParams.width/2;
    ybsl = exptDat1.ex.screenParams.height/2;


    xLeft = edfDat.RawEdf.FSAMPLE.gx(1,:)-xbsl; %Left eye = 1 %center 0 relative to middle of the screen
    yLeft = edfDat.RawEdf.FSAMPLE.gy(1,:)-ybsl; %Left eye = 1 %center 0 relative to middle of the screen

    xRight = edfDat.RawEdf.FSAMPLE.gx(2,:)-xbsl; %Right eye = 2 %center 0 relative to middle of the screen
    yRight = edfDat.RawEdf.FSAMPLE.gy(2,:)-ybsl; %Right eye = 2 %center 0 relative to middle of the screen

    pixperdeg = exptDat1.ex.ppd;
    
    xL = xLeft/pixperdeg; %convert from pixels to degrees
    yL = yLeft/pixperdeg; %convert from pixels to degrees
        
    xR = xRight/pixperdeg; %convert from pixels to degrees
    yR = yRight/pixperdeg; %convert from pixels to degrees

    %remove blinks
    pupSize = edfDat.Samples.pupilSize;
    [eyePosL,blinksL] = removeBlinks(xL,yL, pupSize(:,1)); %detect blinks and replace them by NaNs
    xL = eyePosL(1,:);
    yL = eyePosL(2,:);
    [eyePosR,blinksR] = removeBlinks(xR,yR, pupSize(:,2)); %detect blinks and replace them by NaNs
    xR = eyePosR(1,:);
    yR = eyePosR(2,:);

    origX = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    origY = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    blinkTrs = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);

    clear conds elonsetIdx

    conds = [exptDat1.ex.conds,exptDat2.ex.conds,exptDat3.ex.conds];
    elonsetIdx = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET')).*2;%
    edfConds = edfDat.Events.Messages.info(contains(edfDat.Events.Messages.info, conds));

        for c =5:length(conds)
            clear onsetTcond offsetTcond
            %experimental conditions
            elCondIdx = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))).*2;
            longSpeedShuffle = [exptDat1.ex.longSpeedShuffle; exptDat2.ex.longSpeedShuffle; exptDat3.ex.longSpeedShuffle];
            driftingSpeedsCond = longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
            trialdur = [exptDat1.ex.trialdur;exptDat2.ex.trialdur;exptDat3.ex.trialdur];
            trialDurations = trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition
    
            onsetTcond = [];
            offsetTcond = [];
            for l =1:length(elCondIdx)
                onsetTcond = [onsetTcond elonsetIdx(find(elonsetIdx>=elCondIdx(l),1,'first'))];
                % offsetTcond = [offsetTcond offsetT(find(offsetT>=edfCondT(l),1,'first'))];
            end
    
            for n =1:length(onsetTcond)
    
                elSampleTimes = edfDat.RawEdf.FSAMPLE.time.*2;
                trstart = find(elSampleTimes == onsetTcond(n));
                trend = round(find(elSampleTimes == onsetTcond(n))+sample_rate*(exptDat1.ex.initialFixDur+ (2*exptDat1.ex.dot_displacement_deg)/driftingSpeedsCond(n)))-1;
                xtrL = xL(trstart(1):trend(1)); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
                xtrR = xR(trstart(1):trend(1)); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    
                ytrL = yL(trstart(1):trend(1)); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
                ytrR = yR(trstart(1):trend(1)); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    
                blinktrR = blinksR(trstart(1):trend(1)); %find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))
                blinktrL = blinksL(trstart(1):trend(1));
    
                origX(1:length(xtrL),n,c-4,1) = xtrL;
                origX(1:length(xtrR),n,c-4,2) = xtrR;
    
                origY(1:length(ytrL),n,c-4,1) = ytrL;
                origY(1:length(ytrR),n,c-4,2) = ytrR;
    
                blinkTrs(1:length(blinktrR),n,c-4,1) = blinktrR;
                blinkTrs(1:length(blinktrL),n,c-4,2) = blinktrL;
    
                % % %trial plot
                % figure('Position',[100 100 1200 600]); %
                % time = [repmat(0,1,sample_rate), 1:1/sample_rate:trialDurations(n)+1];
                % plot(time(1:end-1),origX(1:length(time(1:end-1)),n,c-4,1),'LineWidth', 2,'col','r')
                % hold on
                % %stim trajectory
                % xtraj = [repmat(20,1,sample_rate), (-1)*(-exptDat1.ex.dot_displacement_deg:driftingSpeedsCond(n)/sample_rate:exptDat1.ex.dot_displacement_deg)];
                % plot(time, xtraj ,'LineWidth', 2,'col','b')
                % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
                % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
                % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_electrooculogram_deg_%s.jpg',name,conds{c},n,version)));
                % 
    
            end
        end
             

    subj = sprintf('sub0%d',i);
    data.(subj).origx = origX; %origx and origy are all trials after blink removal, and before removing outliers
    data.(subj).origy = origY;

    data.(subj).blinkTrs = blinkTrs;
end


%% 2: Remove saccades from visual conditions data, add trials without saccades to structure, along with saccade onset t/offset t

%Important parameters and variables
fixations = 111;
pursuits = 222;
saccades = 333;

data.(subj).sacc_free_eye_posx = nan(size(data.(subj).origx));
data.(subj).sacc_free_eye_posy = nan(size(data.(subj).origy));
data.(subj).saccadesTrs = zeros(size(data.(subj).origx));
data.(subj).pursuitTrs = zeros(size(data.(subj).origx));
data.(subj).target_pos = nan(size(data.(subj).origx,1,2,3));
data.(subj).open_velocity_gain = nan(size(data.(subj).origx,2,3));
data.(subj).closed_velocity_gain = nan(size(data.(subj).origx,2,3));

for whicheye = 1:2 %1 = left, 2 = right
    % open_velo_gain = nan(length(longSpeedShuffle(contains(edfConds, conds(9)))), 5);
    % closed_velo_gain = nan(length(longSpeedShuffle(contains(edfConds, conds(9)))), 5);
    % target_velocity_trials = nan(size(data.(subj).origx,1), length(longSpeedShuffle(contains(edfConds, conds(9)))), 5);
    % filtered_velocity_trials = nan(size(data.(subj).origx,1), length(longSpeedShuffle(contains(edfConds, conds(9)))), 5);
    for c = 5:10
        driftingSpeedsCond = longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        trialDurations = trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition
        time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second

        for whichtrial = 1:length(driftingSpeedsCond)

            eye_position_trace(:,1) = data.(subj).origx(:, whichtrial, c-4, whicheye);
            eye_position_trace(:,2) = data.(subj).origy(:, whichtrial, c-4, whicheye);

            %% --- detect saccades
            [sac_index, classified, eye_velocity_trace] = detect_saccades(eye_position_trace, sample_rate);

            eye_position_removed = eye_position_trace;
            eye_velocity_removed = eye_velocity_trace;
            eye_position_removed(sac_index,:) = NaN;
            eye_velocity_removed(sac_index,:) = NaN;
            
            data.(subj).sacc_free_eye_posx(:, whichtrial, c-4, whicheye) = eye_position_removed(:,1);
            data.(subj).sacc_free_eye_posy(:, whichtrial, c-4, whicheye) = eye_position_removed(:,2);
            
            if ~isnan(sac_index)
                data.(subj).saccadesTrs(sac_index, whichtrial, c-4, whicheye) = 1;
            end

            if contains(conds{c}, 'Left')
                eye_velocity_removed = (-1)*eye_velocity_removed;
                target_position_trace = [repmat(exptDat1.ex.dot_displacement_deg,1,sample_rate*exptDat1.ex.initialFixDur-1), (-1)*(-exptDat1.ex.dot_displacement_deg:driftingSpeedsCond(whichtrial)/sample_rate:exptDat1.ex.dot_displacement_deg)];
                target_velocity_trace = -gradient(target_position_trace, time_unit);

            elseif contains(conds{c}, 'Right')
                target_position_trace = [(-1)*repmat(exptDat1.ex.dot_displacement_deg,1,sample_rate*exptDat1.ex.initialFixDur-1), (-exptDat1.ex.dot_displacement_deg:driftingSpeedsCond(whichtrial)/sample_rate:exptDat1.ex.dot_displacement_deg)];
                target_velocity_trace = gradient(target_position_trace, time_unit);
            end
            
            data.(subj).target_pos(1:length(target_position_trace), whichtrial, c-4) = target_position_trace;

            % target_velocity_trials(1:length(target_velocity_trace),whichtrial,c-8) = target_velocity_trace;
            % figure();
            % plot(eye_position_removed)
            % hold on
            % plot(target_position_trace)

            %% plot
            % time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second
            % eyedur = length(eye_position_trace(:,1));
            % plot_time = [0:eyedur-1]';
            % plot_time = (plot_time .* time_unit);
            % plot_time_removed = plot_time;
            % plot_time_removed(sac_index) = NaN;
            % 
            % saccade_vec = zeros(length(eye_position_trace),1);
            % saccade_vec(sac_index) = 1;
            % 
            % 
            % plot_time2 = [0:1/sample_rate:trialDurations(whichtrial)+1];
            % 
            % figure();
            % subplot(4,1,1)
            % plot(plot_time2,target_position_trace, 'LineWidth', 2);
            % hold on;
            % plot(plot_time2(1:end-1),eye_position_removed(1:length(plot_time2)-1,1),'LineWidth', 2);
            % ylabel('horizontal position (deg)')
            % xlim([0 length(target_position_trace)/sample_rate])
            % ylim([-25 25])
            % title('Gaze location, target location');
            % 
            % subplot(4,1,2)
            % plot(plot_time_removed,eye_velocity_removed(:,1));
            % xlim([0 length(target_position_trace)/sample_rate])
            % ylim([-40 40])
            % % plot(plot_time2,target_velocity_trace,'k-'); hold on;
            % 
            % ylabel('velocity (deg/sec)');
            % title('Saccade removed velocity');
            % 
            % subplot(4,1,3)
            % plot(plot_time_removed,data.(subj).blinkTrs(:, whichtrial, c, whicheye));
            % xlim([0 length(target_position_trace)/sample_rate])
            % ylim([0 2])
            % title('Blinks');
            % 
            % subplot(4,1,4)
            % plot(plot_time,saccade_vec);
            % xlim([0 length(target_position_trace)/sample_rate])
            % ylim([0 2])
            % xlabel('time (sec)');
            % title('Saccades');
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_eyelink_blink_saccade_velocity_%s.jpg',name,conds{c},whichtrial,version)));

            %% --- Detect smooth pursuit
            [pursuit_index, pursuit_onset, classified] = detect_pursuit(eye_position_trace, sample_rate, classified);

            if ~isnan(pursuit_index)
                data.(subj).pursuitTrs(pursuit_index, whichtrial, c-4, whicheye) = 1;
            end

            %% plot
            % figure();
            % time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second
            % eyedur = length(eye_position_trace(:,1));
            % plot_time = [0:eyedur-1]';
            % plot_time = (plot_time .* time_unit);
            % plot(plot_time(classified==fixations),eye_position_trace(classified==fixations,1),'b.');
            % hold on;
            % plot(plot_time(classified==pursuits),eye_position_trace(classified==pursuits,1),'r.');
            % hold on;
            % plot(plot_time(classified==saccades),eye_position_trace(classified==saccades,1),'g.');
            % hold on;
            % plot_time2 = [0:1/sample_rate:trialDurations(whichtrial)+1];
            % target_traj = [repmat(exptDat.ex.dot_displacement_deg,1,sample_rate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(whichtrial)/sample_rate:exptDat.ex.dot_displacement_deg)];
            % plot(plot_time2,target_position_trace, 'LineWidth', 2);
            % xlabel('time (sec)');
            % ylabel('x position (deg)');
            % legend('Fixation', 'Pursuit', 'Saccade')
            % xlim([0 length(target_position_trace)/sample_rate])
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_eyelink_saccade_pursuit_fixation_%s.jpg',name,conds{c},whichtrial,version)));
            %% --- interpolation (velocity)

            eye_velocity_interp = fillmissing(eye_velocity_removed(:,1),'linear','EndValues','extrap');

            %% plot
            % figure();
            % time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second
            % plot_time2 = [0:1/sample_rate:trialDurations(whichtrial)+exptDat.ex.initialFixDur];
            %
            % eyedur = length(eye_position_trace(:,1));
            % plot_time = [0:eyedur-1]';
            % plot_time = (plot_time .* time_unit);
            %
            %
            % subplot(2,1,1)
            % plot(plot_time,eye_velocity_removed(:,1));
            % hold on;
            % plot(plot_time2,target_velocity_trace,'k-');
            % xlim([0 length(target_traj)/sample_rate])
            % ylim([-100 100])
            % ylabel('velocity (deg/sec)');
            % title('Saccade removed velocity');
            % legend('Left eye velocity', 'Target velocity')
            %
            % subplot(2,1,2)
            % plot(plot_time_removed,eye_velocity_interp(:,1));
            % xlim([0 length(target_traj)/sample_rate])
            % ylim([-100 100])
            % xlabel('time (s)')
            % title('Interpolated velocity')
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_eyelink_interp_velocity_%s.jpg',name,conds{c},whichtrial,version)));
            %% --- Butterworth filter velocity

            fc = 40; % cut off frequency
            fs = sample_rate;
            n_order = 2; % 2nd order

            [b,a] = butter(n_order,fc/(fs/2)); %         freqz(b,a);

            filtered_velocity = filter(b,a,eye_velocity_interp);
            % filtered_velocity_trials(:,whichtrial,c-4) = filtered_velocity;

            % figure();
            % plot(plot_time_removed,eye_velocity_interp(:,1));
            % hold on
            % plot(plot_time_removed,filtered_velocity(:,1));
            % xlim([0 length(target_traj)/sample_rate])
            % ylim([-200 200])
            % xlabel('time (s)')
            % title('Interpolated vs filtered velocity')
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_eyelink_filtered_velocity_%s.jpg',name,conds{c},whichtrial,version)));

            [open_velo_gain(whichtrial,c-4), closed_velo_gain(whichtrial,c-4)] = pursuit_velocity_gain(classified, eye_position_removed, filtered_velocity, target_position_trace, target_velocity_trace, sample_rate, driftingSpeedsCond(whichtrial), whichtrial);
        end
    end
    data.(subj).open_velocity_gain = open_velo_gain;
    data.(subj).closed_velocity_gain = closed_velo_gain;
end

%% add eyelink cond names to the structure

data.eyelink_conds = conds(5:end);
data.biopac_conds = conds;
data.leftEye = 1;
data.rightEye = 2;

%% Add EOG data to the structure
EL_TTL_pulse_on = nan(3, length(names));
EL_TTL_pulse_off = nan(3, length(names));

acq_TTL_pulse_on = nan(3, length(names));
acq_TTL_pulse_off = nan(3, length(names));

for i =1:length(names)
    
    name = names{i};
    folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
        sprintf('v%s/%s_v%s',version, name,version), '/');
    fileNames = dir(folderDir);
    acqName = strcat(folderDir,fileNames(contains({fileNames.name},'eog.mat')).name);
    acq_data = load(acqName);
    
    %lowpass filter EOG data
    nyq = sample_rate/2;
    % fc = 35;                 % cutoff Hz (adjust if you choose)
    % order = 4;
    % % design
    % [b,a] = butter(order, fc/nyq, 'low');
    % filt_eog_L = filtfilt(b,a,acq_data.data(:,7));
    % acq_data.data(:,7) = filt_eog_L;
    % filt_eog_R = filtfilt(b,a,acq_data.data(:,8));
    % acq_data.data(:,8) = filt_eog_R;

    ds_factor = 5;
    new_fs = sample_rate / ds_factor;
    nyq_ds = new_fs / 2;
    
    fc_hp = 0.1;       % Butterworth High-pass
    fc_lp = 35;        % Butterworth Low-pass
    order = 4;         % Filter order

    % Downsample to improve numerical conditioning
    raw_ds_L = resample(acq_data.data(:,7), 1, ds_factor);
    raw_ds_R = resample(acq_data.data(:,8), 1, ds_factor);

    %SOS (second order sections, to filter sequentially and reduce filter load)
    [sos_hp, g_hp] = butter(order, [fc_hp  fc_lp]/nyq_ds, 'bandpass');
    
    %Apply filter
    filt_eog_L = filtfilt(sos_hp, g_hp, raw_ds_L);
    filt_eog_R = filtfilt(sos_hp, g_hp, raw_ds_R);

    %Upsample
    us_factor = 5;
    upsampled_filt_eog_L = resample(filt_eog_L , us_factor, 1);
    upsampled_filt_eog_R = resample(filt_eog_R , us_factor, 1);
    
    EL_TTL_pulse_on(:,i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_On'));
    EL_TTL_pulse_off(:,i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_Off'));

    TTL_pulse = find(diff(acq_data.data(:,9)).^2 > 0)+1;

    if length(TTL_pulse) == 6
        acq_TTL_pulse_on(:,i) = TTL_pulse([1,3,5]);
        acq_TTL_pulse_off(:,i) = TTL_pulse([2,4,6]);
    elseif length(TTL_pulse) == 7
        acq_TTL_pulse_on(:,i) = TTL_pulse([2,4,6]);
        acq_TTL_pulse_off(:,i) = TTL_pulse([3,5,7]);
    end

    % figure();
    % subplot(2,1,1)
    % plot(acq_data.data(:,1))
    % % xlim([11460000; 11550000])
    % subplot(2,1,2)
    % plot(acq_data.data(:,7), col = 'b')
    % hold on
    % plot(filt_eog_L,col = 'r')
    % ylabel('Left X')
    % xlim([11460000; 11550000])
    % xline(11821100)
    % xlim([2039250; 2041450])
    % subplot(3,1,2)
    % plot(acq_data.data(:,8))
    % hold on
    % plot(filt_eog_R)
    % ylabel('Left Y')
    % % xlim([2039250; 2041450])
    % subplot(3,1,3)
    % plot(acq_data.data(:,9))
    % ylabel('TTL Pulse')
    % % xlim([2039250; 2041450])
    % hold on
    % % xline(TTL_pulse, 'LineWidth', 2)
    % ylim([0; 6])


    eyeLink_acq_on_diff = EL_TTL_pulse_on(:,i)*2 - acq_TTL_pulse_on(:,i); % 
    eyeLink_acq_off_diff = EL_TTL_pulse_off(:,i)*2 - acq_TTL_pulse_off(:,i);


    clear origX origY
    origX = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat1.ex.conds)+length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    origY = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat1.ex.conds)+length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    filtX = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat1.ex.conds)+length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    filtY = nan((1+max(exptDat1.ex.trialdur))*sample_rate,exptDat1.ex.repsPerRun, length(exptDat1.ex.conds)+length(exptDat2.ex.conds)+length(exptDat3.ex.conds),2);
    acqonsetIdx = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET')).*2  - eyeLink_acq_on_diff(3);%

    %-------- detect any offset between Eyelink and adjusted EOG data ----
    %Correlate Eyelink Left Eye position digital data of first trial of visually guided condition with EOG digital data
    %This step is for detecting any kind of additional offset
    %between the EOG adjusted event markers and Eyelink event markers
    driftingSpeedsCond = longSpeedShuffle(contains(edfConds, conds(9))); %all drifting speeds of all trials for a given condition (speed varies across trials)
    acqCondIdx = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(9))).*2 - eyeLink_acq_on_diff(3); %times of onset of specific condition
    acqonsetTcond = [];
    for l =1:length(acqCondIdx)
        acqonsetTcond = [acqonsetTcond acqonsetIdx(find(acqonsetIdx>=acqCondIdx(l),1,'first'))];
    end
    
    %1: prepare Eyelink digital trial eye position data
    eyelink_trL = data.(subj).origx(1:sum(~isnan(data.(subj).origx(:,1,5,1))),1,5,1);
    nan_vals = isnan(eyelink_trL); %Eyelink has nans due to blinks, so we need to exclude these to calculate pearson's R
    eyelink_trL_noNan = eyelink_trL(~nan_vals);

    %2: prepare biopac Eyelink "analog" trial data
    acqSampleIdx = edfDat.RawEdf.FSAMPLE.time.*2 - eyeLink_acq_on_diff(3);
    trstart = acqSampleIdx(acqSampleIdx == acqonsetTcond(1));
    trend = round(acqSampleIdx(acqSampleIdx == acqonsetTcond(1))+sample_rate*(exptDat1.ex.initialFixDur+ (2*exptDat1.ex.dot_displacement_deg)/driftingSpeedsCond(1)))-1;
    acq_trL = acq_data.data(trstart(1)-600:trend(1)+600,1);
    
    for e =1:1200
        acq_dat = acq_trL(e:e+length(eyelink_trL)-1);
        corrMat = corrcoef(eyelink_trL_noNan, acq_dat(~nan_vals));
        pearsonR(e) = corrMat(1,2);
    end
    [M, maxCorrIdx] = max(pearsonR);
    offsetIdx = maxCorrIdx-600;

    % figure();
    % subplot(2,1,1)
    % plot(eyelink_trL)
    % subplot(2,1,2)
    % plot(acq_data.data(trstart(1):trend(1),1))
    % 
    driftingSpeeds = nan(length(driftingSpeedsCond), length(conds));
    % ----- Get EOG data ------ %
    for c =1:length(conds)
        clear acqCondIdx acqonsetTcond elonsetTcond trstart trend acqtrL
        %experimental conditions onset times
        acqCondIdx = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))).*2 - eyeLink_acq_on_diff(3); %times of onset of specific condition
        elCondIdx = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))).*2;
        %informative variables
        longSpeedShuffle = [exptDat1.ex.longSpeedShuffle; exptDat2.ex.longSpeedShuffle; exptDat3.ex.longSpeedShuffle];
        driftingSpeedsCond = longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        driftingSpeeds(:,c) = driftingSpeedsCond;
        
        trialdur = [exptDat1.ex.trialdur;exptDat2.ex.trialdur;exptDat3.ex.trialdur];
        trialDurations = trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition

        acqonsetTcond = [];
        % elonsetTcond = [];
        for l =1:length(acqCondIdx)
            acqonsetTcond = [acqonsetTcond acqonsetIdx(find(acqonsetIdx>=acqCondIdx(l),1,'first'))+offsetIdx];
            % elonsetTcond = [elonsetTcond elonsetIdx(find(elonsetIdx>=elCondIdx(l),1,'first'))];
        end

        for n = 1:length(acqonsetTcond)

            %Extract analog Eyelink trial data
            acqSampleIdx = edfDat.RawEdf.FSAMPLE.time.*2 - eyeLink_acq_on_diff(3)+ offsetIdx;
            trstart = acqSampleIdx(acqSampleIdx == acqonsetTcond(n));
            trend = round(acqSampleIdx(acqSampleIdx == acqonsetTcond(n))+sample_rate*(exptDat1.ex.initialFixDur+ (2*exptDat1.ex.dot_displacement_deg)/driftingSpeedsCond(n)))-1;                                                             

            xtrL = acq_data.data(trstart(1):trend(1),7); 
            xtrR = acq_data.data(trstart(1):trend(1),8); 
           
            
            % 
            % figure()
            % subplot(6,1,1)
            % plot(xtrL)
            % subplot(6,1,2)
            % plot(xtrR)
            % subplot(6,1,3)

            origX(1:length(xtrL),n,c,1) = xtrL;
            origX(1:length(xtrR),n,c,2) = xtrR;

            filt_xtrL = upsampled_filt_eog_L(trstart(1):trend(1)); 
            filt_xtrR = upsampled_filt_eog_R(trstart(1):trend(1));

            filtX(1:length(xtrL),n,c,1) = filt_xtrL;
            filtX(1:length(xtrR),n,c,2) = filt_xtrR;

            % figure();
            % subplot(4,1,1)
            % plot(xtrL)
            % subplot(4,1,2)
            % plot(xtrR)
            % subplot(4,1,3)
            % plot(filt_xtrL)
            % subplot(4,1,4)
            % plot(filt_xtrR)

            % % %trial plot (would still need to rescale the EOG signal
            % between -20 and 20 deg)
            % 
            % figure('Position',[100 100 1200 600]); %
            % n = 75;
            % % c =8;
            % time = [repmat(0,1,ACQsamplerate), 1:1/ACQsamplerate:trialDurations(n)+1];
            % plot(time(1:end-1),origX(1:length(time(1:end-1)),n,c,2),'LineWidth', 2,'col','r')
            % hold on
            % %stim trajectory
            % xtraj = [(-1)*repmat(-20,1,ACQsamplerate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(n)/ACQsamplerate:exptDat.ex.dot_displacement_deg)];
            % plot(time, xtraj ,'LineWidth', 2,'col','b')
            % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%s_biopac_electrooculogram_deg_%s.jpg',name,conds{c},n,version)));


        end
        %plot x/y data
        % xax = (1:size(origX,1))./ACQsamplerate;
        % figure('Position',[100 100 1200 600]); %
        % subplot(2,1,1)
        % plot(xax,origX(:,:,c,1),'LineWidth', 1)
        % xlabel("Time (s)")
        % ylabel("Horizontal Displacement (deg)")
        % title("Left eye")
        % subplot(2,1,2)
        % plot(xax,origX(:,:,c,2),'LineWidth', 1)
        % xlabel("Time (s)")
        % ylabel("Horizontal Displacement (deg)")
        % title("Right eye")
        % 
        % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
        % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
        % saveas(gcf,strcat(plotdir, sprintf('%s_%s_left_right_eye_electrooculogram_deg_%s.jpg',name,conds{c},version)));
        
    end
    data.(subj).eog_xpos = origX;
    data.(subj).eog_filt_xpos = filtX;
    data.(subj).driftingSpeeds = driftingSpeeds;

    %% Save data

    save(strcat(folderDir, sprintf('%s_model_data_filt_audio_visual_motion_v%s', name, version)), 'data')

end

