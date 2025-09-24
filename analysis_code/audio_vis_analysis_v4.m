%Loic Daumail
%Summer 2025
%Analysis of eye movement data combined with EOG

ELsamplerate = 1000;
version = '04';
names = {'sub-02'};

for i =1:length(names)
    name = names{i};

    %if contains(name, s_f_names) %check if subject had the slow condition first or the fast one
    folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
        sprintf('v%s/%s_v%s',version, name,version), '/');

    fileNames = dir(folderDir);
    fileNames = string({fileNames(contains({fileNames.name},'AUV')| contains({fileNames.name},'.mat')).name});

    edfName = char(strcat(folderDir,fileNames(contains(fileNames,'.edf'))));
    edfDat = Edf2Mat(edfName);
    exptName = strcat(folderDir,fileNames(contains(fileNames,'fix.mat')));
    exptDat = load(exptName);
    
    %baseline correct data
    xbsl = exptDat.ex.screenParams.width/2;
    ybsl = exptDat.ex.screenParams.height/2;


    xLeft = edfDat.RawEdf.FSAMPLE.gx(1,:)-xbsl; %Left eye = 1 %center 0 relative to middle of the screen
    yLeft = edfDat.RawEdf.FSAMPLE.gy(1,:)-ybsl; %Left eye = 1 %center 0 relative to middle of the screen

    xRight = edfDat.RawEdf.FSAMPLE.gx(2,:)-xbsl; %Right eye = 2 %center 0 relative to middle of the screen
    yRight = edfDat.RawEdf.FSAMPLE.gy(2,:)-ybsl; %Right eye = 2 %center 0 relative to middle of the screen

    pixperdeg = exptDat.ex.ppd;
    
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

    % xR(xR > 1000000) = 0;
    % yR(yR > 1000000) = 0;


    %% get trials data triggered to trial onset times and trial offset times
    
    clear speed s xpTr ypTr 

    origX = nan((1+max(exptDat.ex.trialdur))*ELsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    origY = nan((1+max(exptDat.ex.trialdur))*ELsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    blinkTrs = nan((1+max(exptDat.ex.trialdur))*ELsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);

    clear conds acqonsetT offsetT
    conds = exptDat.ex.conds;

    acqonsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET'));%
    %offsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_OFFSET'));%

    edfConds = edfDat.Events.Messages.info(contains(edfDat.Events.Messages.info, conds));
    for c =1:length(conds)
        clear onsetTcond offsetTcond
        %experimental conditions
        edfCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))); %times of onset of specific condition
        driftingSpeedsCond = exptDat.ex.longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        trialDurations = exptDat.ex.trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition
        % xtraj = exptDat.ex.xval(:,contains(edfConds, conds(c))); %all trajectories of all trials for a given condition (speed varies across trials)

        onsetTcond = [];
        offsetTcond = [];
        for l =1:length(edfCondT)
            onsetTcond = [onsetTcond acqonsetT(find(acqonsetT>=edfCondT(l),1,'first'))];
            % offsetTcond = [offsetTcond offsetT(find(offsetT>=edfCondT(l),1,'first'))];
        end

        for n =1:length(onsetTcond)
            xtrL = xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            xtrR = xR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))

            ytrL = yL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            ytrR = yR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))

            blinktrR = blinksR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)));
            blinktrL = blinksL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+ELsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)));

            origX(1:length(xtrL),n,c,1) = xtrL;
            origX(1:length(xtrR),n,c,2) = xtrR;

            origY(1:length(ytrL),n,c,1) = ytrL;
            origY(1:length(ytrR),n,c,2) = ytrR;

            blinkTrs(1:length(blinktrR),n,c,1) = blinktrR;
            blinkTrs(1:length(blinktrL),n,c,2) = blinktrL;

            % %trial plot
            % figure('Position',[100 100 1200 600]); %
            % time = [repmat(0,1,samplerate), 1:1/samplerate:trialDurations(n)+1];
            % plot(time(1:end-1),origX(1:length(time(1:end-1)),n,c),'LineWidth', 2,'col','r')
            % hold on
            % %stim trajectory
            % xtraj = [repmat(20,1,samplerate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(n)/samplerate:exptDat.ex.dot_displacement_deg)];
            % plot(time, xtraj ,'LineWidth', 2,'col','b')
            % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
            % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%s_electrooculogram_deg_%s.jpg',name,conds{c},n,version)));


        end
        % %plot x/y data
        % xax = (1:size(origX,1))./1000;
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



    subj = sprintf('sub0%d',i);
    data.(subj).origx = origX; %origx and origy are all trials after blink removal, and before removing outliers
    data.(subj).origy = origY;

    % data.(subj).meanadjx = xpTr; %xpTr had outlier trials removed
    % data.(subj).meanadjy = ypTr;
    data.(subj).blinkTrs = blinkTrs;


    % [data.(char(speed)).(subj).eyeMovDat,cOrigX,cOrigY] = eyeMovDetectTr(folderDir,char(subj),xpTr(1000+length(oscillRedDotPts):17000,:,:), ypTr(1000+length(oscillRedDotPts):17000,:,:), blinkTrs(1000+length(oscillRedDotPts):17000,:,:), samplerate, conds);
    % data.(char(speed)).(subj).saccFreex = cOrigX; % cOrigX had saccades with amplitude >2 deg removed, 200ms from saccade onset time
    % data.(char(speed)).(subj).saccFreey = cOrigY;



end



%% Sort data based on speed

n_reps_per_speed = 5; %roughly the amount of trials with the same speed (on the higher end)
n_speeds = 16; %number of speeds for a given condition
for i =1:length(names)
    name = names{i};
    conds = exptDat.ex.conds;
    edfConds = edfDat.Events.Messages.info(contains(edfDat.Events.Messages.info, conds));

    xeyeDat = data.(subj).origx;
    yeyeDat = data.(subj).origy;

    origX = nan((1+max(exptDat.ex.trialdur))*ELsamplerate, n_reps_per_speed, n_speeds, length(exptDat.ex.conds),2);
    origY = nan((1+max(exptDat.ex.trialdur))*ELsamplerate, n_reps_per_speed, n_speeds, length(exptDat.ex.conds),2);

    for c =1:length(conds)
        driftingSpeedsCond = exptDat.ex.longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        speeds = unique(driftingSpeedsCond);

        for s =1:length(speeds)

            xeyeTrs = xeyeDat(:,driftingSpeedsCond == speeds(s),c,:);
            yeyeTrs = yeyeDat(:,driftingSpeedsCond == speeds(s),c,:);

            origX(:,1:size(xeyeTrs,2),s,c,:) = xeyeTrs;
            origY(:,1:size(yeyeTrs,2),s,c,:) = yeyeTrs;

        end

    end
    subj = sprintf('sub0%d',i);
    sorted_data.(subj).origx = origX; %origx and origy are all trials after blink removal, and before removing outliers
    sorted_data.(subj).origy = origY;

end


%% Eye position error relative to the target

%horizontal 
% oscillRedDotPts =[0:1/1000:timeOn/60];
% flipTimes = 0:1/samplerate:exptDat.ex.blockLength-exptDat.ex.trialFixation;
% flipTimes = flipTimes(1:length(flipTimes)-1);

trRelPos = nan(max(exptDat.ex.trialdur)*ELsamplerate, size(origX,2), length(speeds), length(exptDat.ex.conds),2, length(names));

for i =1:length(names)
    subj = sprintf('sub0%d',i);
    name = names{i};

    for c = 1:length(exptDat.ex.conds) %condition number
        for s =1:length(speeds)
            for t =1:size(sorted_data.(subj).origx,2) %trial number

                if contains(exptDat.ex.conds(c), "Left" )%If cond is visual leftward
                    stimPos = (-1)*(-exptDat.ex.dot_displacement_deg:speeds(s)/ELsamplerate:exptDat.ex.dot_displacement_deg);

                elseif contains(exptDat.ex.conds(c), "Right" )%If cond is visual rightward
                    stimPos = (1)*(-exptDat.ex.dot_displacement_deg:speeds(s)/ELsamplerate:exptDat.ex.dot_displacement_deg);

                end

                if contains(exptDat.ex.conds(c), "Vis" )
                    xEye = sorted_data.(subj).origx(ELsamplerate+1:ELsamplerate+length(stimPos)-1,t,s,c,:); %
                else
                    xEye = sorted_data.(subj).origx(1:length(stimPos)-1,t,s,c,:); %
                end

                % figure();
                % plot(stimPos)
                % hold on
                % plot(squeeze(xEye(:,1,1,1,2)))
                % 


                relPos = (squeeze(xEye(1:end,1,1,1,:))'- stimPos(1:end-1)).^2;
                trRelPos(1:length(relPos),t,s,c,:) = relPos';
            end

        end
    end
    combtrRelPos(:,:,:,1,:,i) = [trRelPos(:,:,:,1,:,i), trRelPos(:,:,:,2,:,i)];
    combtrRelPos(:,:,:,2,:,i) = [trRelPos(:,:,:,3,:,i), trRelPos(:,:,:,4,:,i)];
    combtrRelPos(:,:,:,3,:,i) = [trRelPos(:,:,:,5,:,i), trRelPos(:,:,:,6,:,i)];
    combtrRelPos(:,:,:,4,:,i) = [trRelPos(:,:,:,7,:,i), trRelPos(:,:,:,8,:,i)];
    combtrRelPos(:,:,:,5,:,i) = [trRelPos(:,:,:,9,:,i), trRelPos(:,:,:,10,:,i)];
end

%% AVG eye position error
AVGmatRelPos = nan(n_speeds,length(exptDat.ex.conds)/2,2, length(names)); %

for i =1:length(names)
    for c = 1:length(exptDat.ex.conds)/2 %condition number
        for s =1:length(speeds)
            AVGmatRelPos(s,c,:,i) = squeeze(mean(sqrt(sum(combtrRelPos(:,:,s,c,:,i),2,'omitnan'))./sum(~isnan(combtrRelPos(:,:,s,c,:,i)),2,'omitnan'), 'omitnan'));
        end 
    end
end

%% Eye position error without square root
% AVGmatRelPos = nan(n_speeds,length(exptDat.ex.conds)/2,2, length(names)); %
% 
% for i =1:length(names)
%     for c = 1:length(exptDat.ex.conds)/2 %condition number
%         for s =1:length(speeds)
%             AVGmatRelPos(s,c,:,i) = squeeze(mean(sum(combtrRelPos(:,:,s,c,:,i),2,'omitnan')./sum(~isnan(combtrRelPos(:,:,s,c,:,i)),2,'omitnan'), 'omitnan'));
%         end 
%     end
% end



%% Plot

conditions = {'Low Frequency Involuntary','High Frequency Involuntary','Low Frequency Voluntary','High Frequency Voluntary', 'Visually Guided'}; 
eye = {'Left', 'Right'};
yvar = AVGmatRelPos;
ylab = {'Horizontal eye position error relative to target (deg)'};
ylims = [0 20];
EPEBarPlotSTD(yvar,conditions, ylab, ylims, eye, speeds)

plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
saveas(gcf,strcat(plotDir, sprintf('sub-02_position_error_horizontal.jpg')));




%% Load .acq data from BIOPAC
ELsamplerate = 1000;
ACQsamplerate = 2000;
for i =1:length(names)

    acq_data = load(strcat(folderDir, sprintf('%s_audio_visual_motion_v%s_date20250916_run1.mat', name, version)));

    EL_TTL_pulse_on(i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_On'));
    EL_TTL_pulse_off(i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_Off'));

    TTL_pulse = find(diff(acq_data.data(:,9)).^2 > 0)+1;
    acq_TTL_pulse_on(i) = TTL_pulse(2);
    acq_TTL_pulse_off(i) = TTL_pulse(end);
    figure();
    % subplot(3,1,1)
    % plot(acq_data.data(:,1))
    % xlim([11821000; 11830000])
    % xline(11821100)
    % xlim([2039250; 2041450])
    % subplot(3,1,2)
    % plot(acq_data.data(:,2))
    % xlim([2039250; 2041450])
    % subplot(3,1,3)
    % plot(acq_data.data(:,9))
    % xlim([2039250; 2041450])
    % hold on
    % xline(TTL_pulse, 'LineWidth', 2)

    eyeLink_acq_on_diff = EL_TTL_pulse_on(i)*ACQsamplerate/ELsamplerate - acq_TTL_pulse_on(i); % 
    eyeLink_acq_off_diff = EL_TTL_pulse_off(i)*ACQsamplerate/ELsamplerate - acq_TTL_pulse_off(i);

    % blinksIdxL = find(blinksL).*2 + edfDat.Samples.time(1)*2 - eyeLink_acq_on_diff;
    % blinksIdxR = find(blinksR).*2 + edfDat.Samples.time(1)*2 - eyeLink_acq_on_diff;

    blinksIdxL = find(blinksL) + edfDat.Samples.time(1) - eyeLink_acq_on_diff;
    blinksIdxR = find(blinksR) + edfDat.Samples.time(1) - eyeLink_acq_on_diff;


    blinksMat = nan(length(acq_data.data),2);
    blinksMat(blinksIdxL,1) = 1;
    blinksMat(~blinksIdxL,1) = 0;
    blinksMat(blinksIdxR,2) = 1;
    blinksMat(~blinksIdxR,2) = 0;

    origX = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    origY = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    blinkTrs = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);

    acqonsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET')).*2  - eyeLink_acq_on_diff;%
    elonsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET')).*2;%
    %eloffsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_OFFSET')).*2;%
    for c =1:length(conds)
        clear onsetTcond offsetTcond
        %experimental conditions onset times
        acqCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))).*2 - eyeLink_acq_on_diff; %times of onset of specific condition
        elCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))).*2;
        %informative variables
        driftingSpeedsCond = exptDat.ex.longSpeedShuffle(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        trialDurations = exptDat.ex.trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition

        acqonsetTcond = [];
        elonsetTcond = [];
        %eloffsetTcond = [];
        for l =1:length(acqCondT)
            acqonsetTcond = [acqonsetTcond acqonsetT(find(acqonsetT>=acqCondT(l),1,'first'))];
            elonsetTcond = [elonsetTcond elonsetT(find(elonsetT>=elCondT(l),1,'first'))];
            %eloffsetTcond = [eloffsetTcond eloffsetT(find(offsetT>=edfCondT(l),1,'first'))];
        end

        for n = 1:length(acqonsetTcond)
            acqSampleTimes = edfDat.RawEdf.FSAMPLE.time.*2 - eyeLink_acq_on_diff;
            trstart = acqSampleTimes(acqSampleTimes == acqonsetTcond(n));
            trend = round(acqSampleTimes(acqSampleTimes == acqonsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n));

            % trstart = find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n));
            % trend = round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n));

            xtrL = acq_data.data(trstart(1):trend(1),1); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            xtrR = acq_data.data(trstart(1):trend(1),4); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))

            ytrL = acq_data.data(trstart(1):trend(1),2); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            ytrR = acq_data.data(trstart(1):trend(1),5); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            
            elSampleTimes = edfDat.RawEdf.FSAMPLE.time.*2;
            eltrstart = elSampleTimes(elSampleTimes == elonsetTcond(n));
            eltrend = round(elSampleTimes(elSampleTimes == elonsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n));
            blinktrL = blinksMat(eltrstart(1):eltrend(1),1);
            blinktrR = blinksMat(eltrstart(1):eltrend(1),2);
            

% figure()
% subplot(6,1,1)
% plot(xtrL)
% subplot(6,1,2)
% plot(xtrR)
% subplot(6,1,3)
% plot(ytrL)
% subplot(6,1,4)
% plot(ytrR)
% subplot(6,1,5)
% plot(blinktrL)
% subplot(6,1,6)
% plot(blinktrR)

            % xtrL(blinktrL == 1) = NaN;
            % xtrR(blinktrR == 1) = NaN;
            % 
            % ytrL(blinktrL == 1) = NaN;
            % ytrR(blinktrR == 1) = NaN;

            origX(1:length(xtrL),n,c,1) = xtrL;
            origX(1:length(xtrR),n,c,2) = xtrR;

            origY(1:length(ytrL),n,c,1) = ytrL;
            origY(1:length(ytrR),n,c,2) = ytrR;

            blinkTrs(1:length(blinktrR),n,c,1) = blinktrR;
            blinkTrs(1:length(blinktrL),n,c,2) = blinktrL;

            % % %trial plot (would still need to rescale the EOG signal
            % between -20 and 20 deg)

            % figure('Position',[100 100 1200 600]); %
            % % n = 5;
            % % c =8;
            % time = [repmat(0,1,ACQsamplerate), 1:1/ACQsamplerate:trialDurations(n)+1];
            % plot(time(1:end-1),origX(1:length(time(1:end-1)),n,c,2),'LineWidth', 2,'col','r')
            % hold on
            % %stim trajectory
            % xtraj = [(-1)*repmat(-20,1,ACQsamplerate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(n)/ACQsamplerate:exptDat.ex.dot_displacement_deg)];
            % plot(time, xtraj ,'LineWidth', 2,'col','b')
            % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
            % % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
            % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%s_electrooculogram_deg_%s.jpg',name,conds{c},n,version)));


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

end