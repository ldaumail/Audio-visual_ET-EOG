function [sac_index, classified, eye_velocity_trace] = detect_saccades(eye_position_trace, sample_rate)

%find saccade indices
%Loic Daumail
%10/06/2025
saccades = 333;
time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second

%eyedur: length of eye trace in # of samples
eyedur = length(eye_position_trace(:,1));


% Compute velocity
eye_velocity_trace(:,1) = gradient(eye_position_trace(:,1),time_unit);
eye_velocity_trace(:,2) = gradient(eye_position_trace(:,2),time_unit);
%eye_velocity_trace: gaze velocity in deg/s

% saccade and fixation thresholds
vel_threshold = 30; % deg 22-30 (majority: 30)
amp_threshold = 0.5; % deg
length_threshold = 16 / 1000; % sec (>200 ms: fixations < 200 saccades, >100 ms: fixations -- complex stimuli)
remove_frames = 2;

%Variables of interest
cumamp_vis = NaN;
classified = ones(eyedur,1);

tmp_velo_index = find(abs(eye_velocity_trace(:, 1))>vel_threshold);

% find saccade onsets and offsets
if ~isempty(tmp_velo_index)
    check_seq = diff(tmp_velo_index) == 1;
    id_sac_offset = find(check_seq == 0);
    id_sac_onset = id_sac_offset + 1;
    sac_onset = [tmp_velo_index(1); tmp_velo_index(id_sac_onset)];
    sac_offset = [tmp_velo_index(id_sac_offset); tmp_velo_index(end)];
else
    sac_onset = NaN;
    sac_offset = NaN;
end

vel_index = [];
temp_amp = [];
temp_length = [];
final_sac_onset = [];
final_sac_offset = [];

if ~isnan(sac_onset)
    nsac = 0;
    nsac_visible = 0;
    cumamp_vis = 0;
    for id_sac = 1:length(sac_onset)

        % saccade amplitude
        sac_amplitude_x(id_sac) = max(eye_position_trace(sac_onset(id_sac):sac_offset(id_sac),1)) - ...
            min(eye_position_trace(sac_onset(id_sac):sac_offset(id_sac),1));
        sac_amplitude_y(id_sac) = max(eye_position_trace(sac_onset(id_sac):sac_offset(id_sac),2)) - ...
            min(eye_position_trace(sac_onset(id_sac):sac_offset(id_sac),2));

        sac_amplitude(id_sac) = sqrt(sac_amplitude_x(id_sac).^2 + sac_amplitude_y(id_sac).^2)';
        amp_index(id_sac) = sac_amplitude(id_sac) > amp_threshold;

        % saccade length
        sac_length(id_sac) = (sac_offset(id_sac) - sac_onset(id_sac) + 1) .*time_unit;
        length_index(id_sac) = sac_length(id_sac) > length_threshold;

        if amp_index(id_sac) + length_index(id_sac) == 2 %if saccade amplitude is higher than threshold and if length index is higher than threshold
            if sac_onset(1)-remove_frames>=1 && sac_offset(end)+remove_frames<=eyedur
                vel_index = [vel_index sac_onset(id_sac)-remove_frames:sac_offset(id_sac)+remove_frames];
                final_sac_onset = [final_sac_onset sac_onset(id_sac)-remove_frames];
                final_sac_offset = [final_sac_offset sac_offset(id_sac)+remove_frames];
            else
                vel_index = [vel_index sac_onset(id_sac):sac_offset(id_sac)];
                final_sac_onset = [final_sac_onset sac_onset(id_sac)];
                final_sac_offset = [final_sac_offset sac_offset(id_sac)];
            end
            nsac = nsac + 1;
            temp_amp = [temp_amp sac_amplitude(id_sac)];
            temp_length = [temp_length sac_length(id_sac)];
        end

    end

    meanamp = mean(temp_amp);
    meanlength = mean(temp_length);
    cumamp = sum(temp_amp);

end

% mark saccades
sac_index = vel_index;
classified(sac_index) = saccades;

% count saccades
% tmp_sac = diff(sac_index);
% void_sac_id = sac_index(tmp_sac==2);
% nsac = nsac - sum(tmp_sac==2);
% cumamp_inv = cumamp - cumamp_vis;

% find the last saccade onset and offset
% if ~isempty(final_sac_onset)
%     last_sac_onset = final_sac_onset(end);
%     if ~isempty(void_sac_id)
%         if inv_sac_onset(end) == void_sac_id
%             last_sac_onset = final_sac_onset(end-1);
%         end
%     end
%     last_sac_offset = final_sac_offset(end);
%     first_sac_onset = final_sac_onset(1)*time_unit;
%     first_sac_offset = final_sac_offset(1)*time_unit;
%     last_sac_onset = last_sac_onset*time_unit;
%     last_sac_offset = last_sac_offset*time_unit;
% end


end