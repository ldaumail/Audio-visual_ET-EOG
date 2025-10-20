function [pursuit_index,pursuit_onset, classified] = detect_pursuit(eye_position_trace, sample_rate, classified)  

% eye movement classification
fixations = 111;
pursuits = 222;
saccades = 333;

eyedur = length(eye_position_trace(:,1));
time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second

dispersion_threshold = 1; % deg

% fixation period
dt_window_time = 100; % ms
dt_window = dt_window_time / (1000/sample_rate); %in num samples 

%% --- detect pursuit

dt_window_begin = 1*sample_rate; %1s initial fixation
dt_window_end = min(dt_window_begin + dt_window, eyedur);

while dt_window_begin < eyedur

    dt_dispersion = get_dispersion(dt_window_begin,dt_window_end,eye_position_trace);

    if dt_dispersion < dispersion_threshold
        while dt_dispersion < dispersion_threshold && dt_window_end < eyedur %
            dt_window_end = dt_window_end + 1;
            dt_dispersion = get_dispersion(dt_window_begin,dt_window_end,eye_position_trace);
        end
        for i = dt_window_begin:dt_window_end
            if classified(i) ~= saccades
                classified(i) = fixations;
            end % 
        end % 
        dt_window_begin = dt_window_end + 1;
        dt_window_end = min(dt_window_begin + dt_window, eyedur);
    else
        if classified(dt_window_begin) ~= saccades
            classified(dt_window_begin) = pursuits;
        end
        if dt_window_begin > 1
            if classified(dt_window_begin-1) ~= saccades
                classified(dt_window_begin-1) = pursuits;
            end
        end
        dt_window_begin = dt_window_begin + 1;
    end 
end

pursuit_index = find(classified == pursuits);
check_seq_pursuit = diff(pursuit_index);
pursuit_offset = pursuit_index(check_seq_pursuit ~= 1);
next_pursuit_onset = check_seq_pursuit(check_seq_pursuit ~= 1) + pursuit_offset - 1;
pursuit_offset = pursuit_offset + 1;
total_pursuit = sum(classified == pursuits);

for id_pursuit = 1:length(pursuit_offset)
    if sum(classified(pursuit_offset(id_pursuit):next_pursuit_onset(id_pursuit))) == fixations*(next_pursuit_onset(id_pursuit)-pursuit_offset(id_pursuit)+1);
        classified(pursuit_offset(id_pursuit):next_pursuit_onset(id_pursuit)) = pursuits;
    end
end % for

if isempty(pursuit_index)
    pursuit_index = NaN;
end
pursuit_onset = pursuit_index(1)*time_unit;

%     if makeplots
%         subplot(nrow,ncol,5+(whicheye-1)*ncol); 
%         plot(plot_time(classified==fixations),eye_position_trace(classified==fixations,1),'b.'); hold on;
% %         plot(plot_time(classified==pursuits),eye_position_trace(classified==pursuits,1),'r.'); hold on;
%         plot(plot_time(classified==saccades),eye_position_trace(classified==saccades,1),'g.'); hold on;
%         plot(plot_time,target_position_trace(:,1),'k-'); hold on;
%         plot(target_time,target_position,'go'); hold on; % target location
%         plot(occluder_time,occluder_position,'co');hold on;
%         xlabel('time (sec)');
%         ylabel('x position (deg)');
%         title(['saccade removed position ' eyetitle{whicheye}]);
%     end
end