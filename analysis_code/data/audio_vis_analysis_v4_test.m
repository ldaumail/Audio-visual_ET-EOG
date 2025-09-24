%Loic Daumail
%Summer 2025
%Analysis of eye movement data combined with EOG

%% Load .acq data from BIOPAC
ELsamplerate = 2000;
ACQsamplerate = 2000;
version = '04';
names = {'sub-01'};
for i =1:length(names)
    name = names{i};
    folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
        sprintf('v%s_test/%s_v%s',version, name,version), '/');
    %%% Biopac Data
    acq_data = load(strcat(folderDir, sprintf('%s_audio_visual_motion_v%s_date20250627T153057_run2.mat', name, version)));

    %%% Eyelink Data

    fileNames = dir(folderDir);
    fileNames = string({fileNames(contains({fileNames.name},'TES1_02')| contains({fileNames.name},'.mat')).name});

    edfName = char(strcat(folderDir,fileNames(contains(fileNames,'.edf'))));
    edfDat = Edf2Mat(edfName);
    exptName = strcat(folderDir,fileNames(contains(fileNames,'153057_fix.mat')));
    exptDat = load(exptName);
    xbsl = exptDat.ex.screenParams.width/2;
    ybsl = exptDat.ex.screenParams.height/2;


    xLeft = edfDat.RawEdf.FSAMPLE.gx(1,:)-xbsl; %Left eye = 1 %center 0 relative to middle of the screen
    yLeft = edfDat.RawEdf.FSAMPLE.gy(1,:)-ybsl; %Left eye = 1 %center 0 relative to middle of the screen

    pixperdeg = exptDat.ex.ppd;

    xL = xLeft/pixperdeg; %convert from pixels to degrees
    yL = yLeft/pixperdeg; %convert from pixels to degrees

    %remove blinks
    pupSize = edfDat.Samples.pupilSize;
    [eyePosL,blinksL] = removeBlinks(xL,yL, pupSize(:,1)); %detect blinks and replace them by NaNs
    xL = eyePosL(1,:);
    xL(1:300) = 0;
    yL = eyePosL(2,:);



    onsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_ONSET'));%
    offsetT = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'STIM_OFFSET'));%
    conds = exptDat.ex.conds;
    edfConds = edfDat.Events.Messages.info(contains(edfDat.Events.Messages.info, conds));
    
    for c =1:length(conds)
        clear onsetTcond offsetTcond
        edfCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))); %times of onset of specific condition
        trialDurations = exptDat.ex.trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition

        onsetTcond = [];
        offsetTcond = [];
        for l =1:length(edfCondT)
            onsetTcond = [onsetTcond onsetT(find(onsetT>=edfCondT(l),1,'first'))];
            offsetTcond = [offsetTcond offsetT(find(offsetT>=edfCondT(l),1,'first'))];
        end
    end

    EL_TTL_pulse_on_time(i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_On'));%-edfDat.Samples.time(1);
    EL_TTL_pulse_off_time(i) = edfDat.Events.Messages.time(strcmp(edfDat.Events.Messages.info, 'TTL_Pulse_Off'));%-edfDat.Samples.time(1);
    (EL_TTL_pulse_on_time-onsetT)
    (EL_TTL_pulse_off_time-offsetT)
    % hold on
    TTL_pulse = find(diff(acq_data.data(:,9)).^2 > 0)+1;
    acq_TTL_pulse_on(i) = TTL_pulse(1);
    acq_TTL_pulse_off(i) = TTL_pulse(2);

    eyelink_TTL_pulse_on(i) = find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_on_time(1),1,'first');
    eyelink_TTL_pulse_off(i) = find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_off_time(1),1,'first');

    eyeLink_acq_on_diff =   eyelink_TTL_pulse_on(i) - acq_TTL_pulse_on(i); % 
    eyeLink_acq_off_diff =  eyelink_TTL_pulse_off(i) - acq_TTL_pulse_off(i);
    
    eyeLinkDur = find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_off_time(1),1,'first') -find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_on_time(1),1,'first');
    acqknowlDur = acq_TTL_pulse_off(1) - acq_TTL_pulse_on(1);

    (eyeLinkDur-acqknowlDur)/2
    (eyeLink_acq_off_diff-eyeLink_acq_on_diff)/2

    % figure('Position',[100 100 1200 600])
    % subplot(3,1,1)
    % plot((1:length(acq_data.data(:,1)))./ACQsamplerate,acq_data.data(:,1))
    % xline(acq_TTL_pulse_on./ACQsamplerate, color = 'r', LineWidth= 2)
    % hold on
    % xline(acq_TTL_pulse_off./ACQsamplerate, color = 'r', LineWidth= 2)
    % hold on
    % xline((find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-eyeLink_acq_off_diff)./ELsamplerate, LineWidth= 2)
    % hold on
    % xline((find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-eyeLink_acq_on_diff)./ELsamplerate, LineWidth= 2, color = 'b')
    % 
    % title('Acqknowledge recording with Acqknowledge TTL Pulse event markers(red)+stim offset marker from eyelink (black)')
    % subplot(3,1,2)
    % plot((1:length(acq_data.data(:,1)))./ACQsamplerate,acq_data.data(:,9), color = 'r')
    % xline((find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_on(1),1,'first')-eyeLink_acq_on_diff)./ELsamplerate, LineWidth= 2)
    % hold on
    % xline((find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_off(1),1,'first')-eyeLink_acq_on_diff)./ELsamplerate, LineWidth= 2)
    % title('Acqknowledge TTL Pulse recording with Eyelink TTL Pulse event marker')
    % subplot(3,1,3)
    % plot((1:length(xL))./ELsamplerate, xL)
    % xline(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first')./ELsamplerate, LineWidth= 2)
    % hold on
    % xline(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')./ELsamplerate, LineWidth= 2)
    % hold on
    % xline(find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_on(1),1,'first')./ELsamplerate, LineWidth= 2)
    % hold on
    % xline(find(edfDat.RawEdf.FSAMPLE.time == EL_TTL_pulse_off(1),1,'first')./ELsamplerate, LineWidth= 2)
    % title('Eyelink recording with Eyelink event markers')
    % 
    % 
    % plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    % saveas(gcf,strcat(plotDir, sprintf('eyelink_vs_acqknowledge_test5.jpg')));



    % Find time series data most correlated between Acqknowledge and
    % Eyelink Channels
    xLeft_acq = acq_data.data(:,1);
    
    % normXLeft_acq = (xLeft_acq - min(xLeft_acq))./(max(xLeft_acq)-min(xLeft_acq));
    % 
    % normXLeft_eyelink = (xL - min(xL))./(max(xL)-min(xL));

    pearsonR = nan(length(xLeft_acq)-(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first')),1);
    for n = 1:length(xLeft_acq)-(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'))
            r = corrcoef(xLeft_acq(n:n+(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'))),xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'):find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')));
            pearsonR(n,1) = r(2,1);
    end
    maxIDX = find(pearsonR == max(pearsonR,[], 'all'),1,'first');



    % 
    % figure();
    % subplot(3,1,1)
    % plot(normXLeft_acq)
    % subplot(3,1,2)
    % plot(normXLeft_eyelink(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'):find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')))
    % subplot(3,1,3)
    % plot(pearsonR)

    figure('Position',[100 100 1200 600]);
    subplot(3,1,1)
    time = 0:1/ACQsamplerate:(length(xLeft_acq)-1)/ACQsamplerate;
    plot(time,xLeft_acq)
    xlabel('Time (sec)')
    ylabel('Horizontal eye position (mA)')
    title('Left eye horizontal data from Acqknowledge recording (2 kHz)')
    subplot(3,1,2)
    %time = [repmat(0,1,ELsamplerate), 1:1/ELsamplerate:exptDat.ex.trialdur+1];
    time = 0:1/ELsamplerate:(length(xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'):find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')))-1)/ELsamplerate;
    plot(time, xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'):find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')))
    xlabel('Time (sec)')
    ylabel('Horizontal eye position (deg)')
    title('Left eye horizontal trial data from Eyelink recording (2 kHz)')
    subplot(3,1,3)
    time = 0:1/ACQsamplerate:(length(pearsonR)-1)/ACQsamplerate;
    plot(time, pearsonR)
    xlim([0 80])
    xlabel('Time (sec)')
    ylabel('Correlation')
    title('Pearson correlation coefficient over time')
        plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    saveas(gcf,strcat(plotDir, sprintf('eyelink_vs_acqknowledge_correlation_test1.jpg')));

    figure();
    plot(xLeft_acq(maxIDX:maxIDX +(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'))))
    hold on
    plot(xL(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first'):find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')))
    legend('Acqknowledge', 'Eyelink')
    plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    saveas(gcf,strcat(plotDir, sprintf('eyelink_vs_acqknowledge_trial_data_test1.jpg')));



    % Based on the time point of max correlation, look at TTL pulse event
    % markers of Eyelink, compared to event markers of Acqknowledge 

    %Trial start
    figure('Position',[100 100 1200 600])
    subplot(1,1,1)
    plot((0:1/ACQsamplerate:(length(acq_data.data(:,1))-1)/ACQsamplerate)*1000,acq_data.data(:,1))
    xlim([60500 60700])
    xline(1000*acq_TTL_pulse_on./ACQsamplerate, color = 'r', LineWidth= 2)
    hold on

    xline(1000*(eyelink_TTL_pulse_on-eyeLink_acq_on_diff)./ACQsamplerate, color = 'k', LineWidth= 2)
    hold on
    xline(1000*(find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first')-eyeLink_acq_on_diff)./ELsamplerate,color = 'b', LineWidth= 2)
    hold on
    xline(1000*maxIDX./ACQsamplerate, color = 'g', LineWidth= 2)

    legend('Electrocculogram', 'AcqKnowledge TTL Pulse On', 'Eyelink TTL Pulse On','Eyelink Stim Onset', 'Max correlation start time', 'FontSize', 20)
    xlabel('Time (ms)')
    plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    saveas(gcf,strcat(plotDir, sprintf('eyelink_vs_acqknowledge_trial_start_data_test1_2.jpg')));

    

    % Trial end
    figure('Position',[100 100 1200 600])
    plot((0:1/ACQsamplerate:(length(acq_data.data(:,1))-1)/ACQsamplerate)*1000,acq_data.data(:,1))
    xlim([69000 72000])
    xline(1000*acq_TTL_pulse_off./ACQsamplerate, color = 'r', LineWidth= 2)
    hold on
    xline(1000*(eyelink_TTL_pulse_off-eyeLink_acq_on_diff)./ACQsamplerate, color = 'k', LineWidth= 2)
    hold on
    xline(1000*(find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-eyeLink_acq_on_diff)./ELsamplerate,color = 'b', LineWidth= 2)
    hold on
    xline(1000*((find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(1),1,'first')-find(edfDat.RawEdf.FSAMPLE.time == onsetTcond(1),1,'first')+ maxIDX)./ACQsamplerate),color = 'g', LineWidth= 2)
    %xline(1000*(trialDurations(1)+ maxIDX./ACQsamplerate), color = 'k', LineWidth= 2)

    legend('Electrocculogram', 'AcqKnowledge TTL Pulse Off', 'Eyelink TTL Pulse Off','Eyelink Stim Offset', 'Max correlation end time', 'FontSize', 20)
    xlabel('Time (ms)')
    plotDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    saveas(gcf,strcat(plotDir, sprintf('eyelink_vs_acqknowledge_trial_end_data_test1.jpg')));





    % blinksIdxL = find(blinksL) - eyeLink_acq_on_diff;
    % blinksIdxR = find(blinksR) - eyeLink_acq_on_diff;
    % 

    % blinksMat = nan(length(acq_data.data),2);
    % blinksMat(blinksIdxL,1) = 1;
    % blinksMat(~blinksIdxL,1) = 0;
    % blinksMat(blinksIdxR,2) = 1;
    % blinksMat(~blinksIdxR,2) = 0;
    % 
    % origX = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    % origY = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    % blinkTrs = nan((1+max(exptDat.ex.trialdur))*ACQsamplerate,exptDat.ex.repsPerRun, length(exptDat.ex.conds),2);
    % 
    % for c =1:length(conds)
    %     clear onsetTcond offsetTcond
    %     %experimental conditions
    %     acqCondT = edfDat.Events.Messages.time(contains(edfDat.Events.Messages.info, conds(c))) - eyeLink_acq_on_diff; %times of onset of specific condition
    %     driftingSpeedsCond = exptDat.ex.driftingSpeeds(contains(edfConds, conds(c))); %all drifting speeds of all trials for a given condition (speed varies across trials)
    %     trialDurations = exptDat.ex.trialdur(contains(edfConds, conds(c)));% durations of trials for specific condition
    % 
    %     onsetTcond = [];
    %     offsetTcond = [];
    %     for l =1:length(acqCondT)
    %         onsetTcond = [onsetTcond onsetT(find(onsetT>=acqCondT(l),1,'first'))];
    %         offsetTcond = [offsetTcond offsetT(find(offsetT>=acqCondT(l),1,'first'))];
    %     end
    % 
    %     for n = 1:length(onsetTcond)
    %         xtrL = acq_data.data(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),1); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    %         xtrR = acq_data.data(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),4); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    % 
    %         ytrL = acq_data.data(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),2); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    %         ytrR = acq_data.data(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),5); %find(edfDat.RawEdf.FSAMPLE.time == offsetTcond(n))
    % 
    %         blinktrL = blinksMat(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),1);
    %         blinktrR = blinksMat(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n)):round(find(edfDat.RawEdf.FSAMPLE.time - eyeLink_acq_on_diff == onsetTcond(n))+ACQsamplerate*(2*exptDat.ex.dot_displacement_deg)/driftingSpeedsCond(n)),2);
    % 
    % 
    %         xtrL(blinktrL == 1) = NaN;
    %         xtrR(blinktrR == 1) = NaN;
    % 
    %         ytrL(blinktrL == 1) = NaN;
    %         ytrR(blinktrR == 1) = NaN;
    % 
    %         origX(1:length(xtrL),n,c,1) = xtrL;
    %         origX(1:length(xtrR),n,c,2) = xtrR;
    % 
    %         origY(1:length(ytrL),n,c,1) = ytrL;
    %         origY(1:length(ytrR),n,c,2) = ytrR;
    % 
    %         blinkTrs(1:length(blinktrR),n,c,1) = blinktrR;
    %         blinkTrs(1:length(blinktrL),n,c,2) = blinktrL;
    % 
    %         % % %trial plot
    %         figure('Position',[100 100 1200 600]); %
    %         subplot(2,1,1)
    %         time = [repmat(0,1,ACQsamplerate), 1:1/ACQsamplerate:trialDurations(n)+1];
    %         plot(time(1:end-1),origX(1:length(time(1:end-1)),n,c,2),'LineWidth', 2,'col','r')
    %         hold on
    %         %stim trajectory
    %         xtraj = [repmat(20,1,ACQsamplerate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(n)/ACQsamplerate:exptDat.ex.dot_displacement_deg)];
    %         plot(time, -xtraj ,'LineWidth', 2,'col','b')
    %         subplot(2,1,2)
    %         time = [repmat(0,1,ELsamplerate), 1:1/ELsamplerate:trialDurations(n)+1];
    %         plot(time(1:end-1),data.sub01.origx(1:length(time(1:end-1)),n,c),'LineWidth', 2,'col','r')
    %         hold on
    %         %stim trajectory
    %         xtraj = [repmat(20,1,ELsamplerate), (-1)*(-exptDat.ex.dot_displacement_deg:driftingSpeedsCond(n)/ELsamplerate:exptDat.ex.dot_displacement_deg)];
    %         plot(time, -xtraj ,'LineWidth', 2,'col','b')
    % 
    %         sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
    %         % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    %         % saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%s_electrooculogram_deg_%s.jpg',name,conds{c},n,version)));
    % 
    % 
    %     end
    %     %plot x/y data
    %     % xax = (1:size(origX,1))./ACQsamplerate;
    %     % figure('Position',[100 100 1200 600]); %
    %     % subplot(2,1,1)
    %     % plot(xax,origX(:,:,c,1),'LineWidth', 1)
    %     % xlabel("Time (s)")
    %     % ylabel("Horizontal Displacement (deg)")
    %     % title("Left eye")
    %     % subplot(2,1,2)
    %     % plot(xax,origX(:,:,c,2),'LineWidth', 1)
    %     % xlabel("Time (s)")
    %     % ylabel("Horizontal Displacement (deg)")
    %     % title("Right eye")
    %     % 
    %     % sgtitle(strcat(conds{c}),'fontweight','bold','Interpreter', 'none')
    %     % plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
    %     % saveas(gcf,strcat(plotdir, sprintf('%s_%s_left_right_eye_electrooculogram_deg_%s.jpg',name,conds{c},version)));
    % 
    % end   

end