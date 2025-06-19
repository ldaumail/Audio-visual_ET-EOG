%Loic Daumail
%Summer 2025
%Analysis of eye movement data combined with EOG

samplerate = 1000;
version = '04';
names = {'sub-01'};

for i =1:length(names)
    name = names{i};

    %if contains(name, s_f_names) %check if subject had the slow condition first or the fast one
    folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
        sprintf('v%s/%s_v%s',version, name,version), '/');

    fileNames = dir(folderDir);
    fileNames = string({fileNames(contains({fileNames.name},'AUV')| contains({fileNames.name},'.mat')).name});

    edfName = char(strcat(folderDir,fileNames(contains(fileNames,'.edf'))));
    edfDat = Edf2Mat(edfName);
    exptName = strcat(folderDir,fileNames(contains(fileNames,'.mat')));
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

    xpTr = nan((1+max(exptDat.ex.trialdur))*samplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    ypTr = nan((1+max(exptDat.ex.trialdur))*samplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    origX = nan((1+max(exptDat.ex.trialdur))*samplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    origY = nan((1+max(exptDat.ex.trialdur))*samplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    blinkTrs = nan((1+max(exptDat.ex.trialdur))*samplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);

    clear conds onsetT offsetT
    conds = exptDat.ex.conds;

    onsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET'));%
    %offsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_OFFSET'));%
    TTL_pulse_on = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_On'));
    
    edfConds = edfDat.Events.Messages.info(contains(edfDat.Events.Messages.info, conds));
    for c =1:length(conds)
        clear onsetTcond offsetTcond
        %experimental conditions
        edfCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))); %times of onset of specific condition
        driftingSpeedsCond = exptDat.ex.driftingSpeeds(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
        trialDurations = exptDat.ex.trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition
        % xtraj = exptDat.ex.xval(:,contains(edfConds, conds(c))); %all trajectories of all trials for a given condition (speed varies across trials)

        onsetTcond = [];
        offsetTcond = [];
        for l =1:length(edfCondT)
            onsetTcond = [onsetTcond onsetT(find(onsetT>=edfCondT(l),1,'first'))];
            % offsetTcond = [offsetTcond offsetT(find(offsetT>=edfCondT(l),1,'first'))];
        end

        for n =1:length(onsetTcond)
            xtrL = xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            xtrR = xR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))

            ytrL = yL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
            ytrR = yR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n))); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))

            blinktrR = blinksR(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)));
            blinktrL = blinksL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(n))+samplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)));

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
        %plot x/y data
        xax = (1:size(origX,1))./1000;
        figure('Position',[100 100 1200 600]); %
        subplot(2,1,1)
        plot(xax,origX(:,:,c,1),'LineWidth', 1)
        xlabel("Time (s)")
        ylabel("Horizontal Displacement (deg)")
        title("Left eye")
        subplot(2,1,2)
        plot(xax,origX(:,:,c,2),'LineWidth', 1)
        xlabel("Time (s)")
        ylabel("Horizontal Displacement (deg)")
        title("Right eye")

        sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
        plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
        saveas(gcf,strcat(plotdir, sprintf('%s_%s_left_right_eye_electrooculogram_deg_%s.jpg',name,conds{c},version)));
        
    end



    % subj = sprintf('sub0%d',i);
    % data.(char(speed)).(subj).origx = origX; %origx and origy are all trials after blink removal, and before removing outliers
    % data.(char(speed)).(subj).origy = origY;
    % 
    % data.(char(speed)).(subj).meanadjx = xpTr; %xpTr had outlier trials removed
    % data.(char(speed)).(subj).meanadjy = ypTr;
    % data.(char(speed)).(subj).blinkTrs = blinkTrs;


    % [data.(char(speed)).(subj).eyeMovDat,cOrigX,cOrigY] = eyeMovDetectTr(folderDir,char(subj),xpTr(1000+length(oscillRedDotPts):17000,:,:), ypTr(1000+length(oscillRedDotPts):17000,:,:), blinkTrs(1000+length(oscillRedDotPts):17000,:,:), samplerate, conds);
    % data.(char(speed)).(subj).saccFreex = cOrigX; % cOrigX had saccades with amplitude >2 deg removed, 200ms from saccade onset time
    % data.(char(speed)).(subj).saccFreey = cOrigY;



end