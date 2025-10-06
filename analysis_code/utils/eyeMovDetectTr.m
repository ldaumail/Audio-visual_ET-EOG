function [eyeMovDat, cleanxDat, cleanyDat] = eyeMovDetectTr(folderDir,session,xDat, yDat, all_blinks, samplerate, conditions) 
%developped by Loic Daumail on 10/28/2022
% folderDir = folderDir;
% xDat = xpTr(1000+length(oscillRedDotPts):17000,:,:);%origX;
% yDat = ypTr(1000+length(oscillRedDotPts):17000,:,:);%origY;
% session = char(names{i});
% all_blinks = blinkTrs(1000+length(oscillRedDotPts):17000,:,:);
% samplerate = samplerate;
% conditions = exptDat.ex.conds;
% folderDir = char(folderDir);
% xDat = subjDat.(folders{i}).origx;
% yDat = subjDat.(folders{i}).origy;
% session = char(folders{i});
% xDat(all(isnan(xDat)),:,:) = 0;
% yDat(all(isnan(yDat)),:,:) = 0;
xcluster = strcat('x', session);

ampl = [];
veloc = [];
durations = [];
cleanxDat = nan(size(xDat));
cleanyDat = nan(size(yDat));
for c =1:size(xDat,3) %loop through conditions
    for tr =1:size(xDat,2) %loop through trials
        saccOnsets = nan(20,1); %to save microsaccade onset times
        saccOffsets = nan(20,1); %to save microsaccade onset times
        trialAmps = nan(20,1); %save microsaccade amplitudes
  
        samples = [];
        samples(:,1) = 1:size(xDat,1);
        %     size of the trial time series data
        if ~isempty(find(xDat(:,tr,c))) && ~all(isnan(all_blinks(:,tr,c))) && ~all(isnan(xDat(:,tr,c)))
            xT = xDat(:,tr,c); 
            yT = yDat(:,tr,c);
 
            samples(:,2) = nan();%+xBaseline; %horizontal position of the left eye in degrees baseline corrected
            samples(:,3) = nan(); %vertical position of the left eye in degrees
            samples(:,4) = xT; %horiz pos right eye
            samples(:,5) = yT; %vert pos right eye
            blinks = all_blinks(:,tr,c);
            
            recording = ClusterDetection.EyeMovRecording.Create(folderDir, session, samples, blinks, samplerate);
            
            % Runs the saccade detection
            try          
            [saccades, stats] = recording.FindSaccades();
            enum = ClusterDetection.SaccadeDetector.GetEnum;

            durations = [durations; saccades(:,3)];
           
%             figure();
%             plot(samples(:,4))
%             hold on
%             for s =1:length(saccades(:,1))
%                 xline(saccades(s,1))
%                 hold on
%             end
            for s =1:length(saccades(:,enum.startIndex)) %loop through all microssaccades/ saccades found in one trial
                if  ~isempty(find(saccades(s,enum.startIndex),1)) %&& ( saccades(s,enum.startIndex)>times(codes==23)-times(1)-250 &&  saccades(s,enum.startIndex) < times(codes == 24)- times(1)) %if there is at least 1 microsaccade and it occurs between trial onset and stim offset
                    
                    %msaccTrials(tr) = msaccTrials(tr)+1; %trials to exclude since they have microsaccades between stim  onset and stim offset
                    saccOnsets(s) = saccades(s,enum.startIndex);%- 
                    saccOffsets(s) = saccades(s,enum.endIndex);%- 
                   
                    trialAmps(s) = saccades(s,enum.amplitude); %save microsaccade amplitude
                    
                end
            end

            % Plots a main sequence
            ampl = [ampl; saccades(:,enum.amplitude)];
            veloc = [veloc; saccades(:,enum.peakVelocity)];
            trial = sprintf('tr%d',tr);
            eyeMovDat.(xcluster).(conditions{c}).(trial).stats = stats;
            eyeMovDat.(xcluster).(conditions{c}).(trial).saccades = saccades;
            eyeMovDat.(xcluster).(conditions{c}).(trial).enum = enum;
            eyeMovDat.(xcluster).(conditions{c}).(trial).samples = samples;
            eyeMovDat.(xcluster).(conditions{c}).(trial).msaccOn = saccOnsets;
            eyeMovDat.(xcluster).(conditions{c}).(trial).msaccAmp = trialAmps;
            if ~isempty(find(trialAmps > 0.1)) %clean up data from microsaccades
                clear s
                sidx = find(trialAmps > 0.1);
                cleanxDat(1:length(xT),tr,c) = xT;
                cleanyDat(1:length(yT),tr,c) = yT;
                for s = 1:length(sidx)
                    idx = sidx(s);
                    sacOn = saccOnsets(idx);
                    sacOff = saccOffsets(idx);
%                     cleanxDat(sacOn:sacOff+min([200 length(cleanxDat(sacOff+1:end,tr,c))]),tr,c) = nan;
%                     cleanyDat(sacOn:sacOff+min([200 length(cleanxDat(sacOff+1:end,tr,c))]),tr,c) = nan;
                    cleanxDat(sacOn:sacOff,tr,c) = nan;
                    cleanyDat(sacOn:sacOff,tr,c) = nan;
%                     cleanxDat(:,tr,c) = fillmissing(cleanxDat(:,tr,c),'previous');
%                     cleanyDat(:,tr,c) = fillmissing(cleanyDat(:,tr,c),'previous');
                end 
            else
                cleanxDat(1:length(xT),tr,c) = xT;
                cleanyDat(1:length(yT),tr,c) = yT;
            end
            catch
                fprintf('trial num: %d, condition: %d', tr, c);
            end
        end
  
    end
end
%figure(); plot(samples(:,4)); hold on; plot(cleanxDat(:,tr,c)-1)
% fprintf('%d', quantile(durations,.95))