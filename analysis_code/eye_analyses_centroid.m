% this code does not work alone!! (but is the main part)
% should always be embedded in other functions (e.g., plot_velocity)
% to load data and set whether to do a sliding window analysis or not
        
%% --- settings

centroid = 0.2;

% eye movement classification
fixations = 111;
pursuits = 222;
saccades = 333;

% basic settings
sample_rate = 120; 
scale_factor = 2; 
time_unit = (1000/sample_rate)/1000;
dist_unit = scale_factor/60;

targetY = 360;
targetY_deg = 360*dist_unit;

% saccade and fixation thresholds
vel_threshold = 30; % deg 22-30 (majority: 30)
length_threshold = 16 / 1000; % sec (>200 ms: fixations < 200 saccaades, >100 ms: fixations -- complex stimuli)
amp_threshold = 0.5; % deg
dispersion_threshold = 1; % deg
vel_crit_col = 1;
remove_frames = 2; 

% fixation period
dt_window_time = 100; % ms
dt_window = dt_window_time / (1000/sample_rate);

% pursuit periods
open_window_time = 100; % ms
open_window = open_window_time / (1000/sample_rate); 
closed_window_begin = 200 / (1000/sample_rate); % index
closed_window_end = 400 / (1000/sample_rate); % index

% plot settings 
nrow = 2;
ncol = 6;
if makeplots
    h = figure(whichtrial);
    set(h, 'Position', [0 0 2500 1000]);
end


% variables of interest
subid_cell = {subid};
begin_fixation_location = [NaN NaN];
end_fixation_location = [NaN NaN];
breakpoint = NaN;
pursuit_onset = NaN;
PQnS_visible = NaN;
PQnS_invisible = NaN;
slip_visible = NaN;
slip_invisible = NaN;
open_gain = NaN;
closed_gain = NaN;
invisible_gain = NaN;
nsac = NaN;
nsac_visible = NaN;
nsac_invisible = NaN;
peak_velo = NaN;
peak_acc = NaN;
rmse_position = NaN;
rmse_position_invisible = NaN;
total_pursuit = NaN;
meanamp = NaN;
meanlength = NaN;
cumamp = NaN;
cumamp_vis = NaN;
cumamp_inv = NaN;
closed_vis = NaN;
closed_vis_length = NaN;
slip_open = NaN;
slip_closed = NaN;
last_sac_onset = NaN;
last_sac_offset = NaN;
first_sac_onset = NaN;
first_sac_offset = NaN;
SAD = NaN;
SED = NaN;
SEDtotal = NaN;
PQnST_invisible = NaN;
total_pursuit_visible = NaN;
total_pursuit_invisible = NaN;
rmse_position_visible = NaN;
rmse_position_open = NaN;
rmse_position_closed = NaN;
final_sac_onset = NaN;
final_sac_offset = NaN;
rel_slip_open = NaN;
rel_slip_closed = NaN;
rel_slip_visible = NaN;
rel_slip_invisible = NaN;


% arrange data
%%% [ 1trial, 2dist, 3velo, 4target_dist, 5occluder_start, 6hit_or_not, 7block

tracking_dist = data_matrix(whichtrial,2); % deg (excluding visible distance)
curr_speed = data_matrix(whichtrial,3); % deg/sec
occluder_position = data_matrix(whichtrial,5); % deg (visible distance)
dist_estimate = data_matrix(whichtrial,4) + centroid; % deg (excluding visible distance)
initial_position = -30 * dist_unit + centroid; % deg (distance prior to fixation)

target_position = tracking_dist + occluder_position + initial_position; % deg
target_time = target_position / curr_speed; % sec

est_position = dist_estimate + occluder_position; % deg

initial_time = initial_position / curr_speed; % sec

stamped_initial_position = eye_data(whichtrial).targetX(1) * dist_unit + centroid;
stamped_initial_time = stamped_initial_position / curr_speed;

eyedur = length(eye_data(whichtrial).time_stamp(eye_data(whichtrial).time_stamp~=0));

plot_time = [0:eyedur-1]';
plot_time = (plot_time .* time_unit) + stamped_initial_time;

targetX = eye_data(whichtrial).targetX(1:eyedur) .* dist_unit + centroid; 
% occluder_index = max(find(targetX < occluder_position));
% occluder_time = plot_time(occluder_index);

occluder_time = occluder_position / curr_speed;
occluder_index = max(find(plot_time<occluder_time));

pursuit_time = plot_time(1:occluder_index);

fit_range = 0.3; % sec (for piecewise linear fit) 
fit_index = find(plot_time>0 & plot_time<fit_range);

x_data = plot_time(fit_index);

% behavior data for saving 

visible_distance = occluder_position;
invisible_distance = tracking_dist;

visible_duration = occluder_time;
invisible_duration = invisible_distance./curr_speed;

est_distance = data_matrix(whichtrial,4);

error_distance = est_distance - invisible_distance; 
error_time = error_distance./curr_speed;

colormat = {'b-', 'r-'};
eyetitle = {'(L)', '(R)'};

%% --- target information

% target position 
target_position_trace(:,1) = eye_data(whichtrial).targetX(1:eyedur,1) + centroid; % x position
target_position_trace(:,2) = repmat(targetY,eyedur,1); % y position
target_position_trace = target_position_trace .* dist_unit;
target_position_trace(:,2) = target_position_trace(:,2) - targetY_deg;

% target velocity
target_velocity_trace(:,1) = gradient(target_position_trace(:,1),time_unit); % x velocity
target_velocity_trace(:,2) = gradient(target_position_trace(:,2),time_unit); % y velocity
target_velocity_trace(:,3) = sqrt(target_velocity_trace(:,1).^2 + target_velocity_trace(:,2).^2); % x and y 

if size(target_position_trace,1) > 2 && numel(x_data)~=0

    if makeplots % all target plots 

        % target position 
        subplot(nrow,ncol,1);
        plot(target_position_trace(:,1),target_position_trace(:,2),'k-'); hold on; % target position
        plot(target_position,targetY_deg-targetY_deg,'go'); hold on; % target location
        plot(occluder_position,targetY_deg-targetY_deg,'co');hold on;
        xlabel('x position (deg)');
        ylabel('y position (deg)');
        title('position x position');
        legend('target','Location','Best');

        subplot(nrow,ncol,2);
        plot(plot_time,target_position_trace(:,1),'k-'); hold on; % target position
        plot(target_time,target_position,'go'); hold on; % target location
        plot(occluder_time,occluder_position,'co'); hold on;
        xlabel('time (sec)');
        ylabel('x position (deg)');
        title('position over time');
        legend('target','target loc','occluder loc','Location','Best');

    end

    %% --- plot eye position
    eye_position_trace(:,1) = eye_data(whichtrial).eyeX(1:eyedur,whicheye);
    eye_position_trace(:,2) = eye_data(whichtrial).eyeY(1:eyedur,whicheye);

    eye_position_trace = eye_position_trace .* dist_unit;
    eye_position_trace(:,2) = eye_position_trace(:,2) - targetY_deg;

    if makeplots
        subplot(nrow,ncol,1);
        plot(eye_position_trace(:,1),eye_position_trace(:,2),colormat{whicheye}); hold on; % eye position
        subplot(nrow,ncol,2);
        plot(plot_time,eye_position_trace(:,1),colormat{whicheye}); hold on; % eye position
    end

    %% --- plot eye velocity

    eye_velocity_trace(:,1) = gradient(eye_position_trace(:,1),time_unit); 
    eye_velocity_trace(:,2) = gradient(eye_position_trace(:,2),time_unit);
    theta_trace = atan(eye_velocity_trace(:,2)./eye_velocity_trace(:,1));
    eye_velocity_trace(:,3) = eye_velocity_trace(:,1)./cos(theta_trace);
%     eye_velocity_trace(:,3) = sqrt(eye_velocity_trace(:,1).^2 + eye_velocity_trace(:,2).^2);

    if makeplots
        subplot(nrow,ncol,3+(whicheye-1)*ncol);
        plot(plot_time,eye_velocity_trace(:,vel_crit_col),colormat{whicheye}); hold on; 
        plot(plot_time,target_velocity_trace(:,vel_crit_col),'k-'); hold on; 
        xlabel('time (sec)');
        ylabel('velocity (deg/sec)');
        title(['velocity ' eyetitle{whicheye}]);
    end

    %% --- detect saccade

    classified = ones(eyedur,1);

    tmp_velo_index = find(abs(eye_velocity_trace(:,vel_crit_col))>vel_threshold);

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

            if amp_index(id_sac) + length_index(id_sac) == 2
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
                if sac_onset(id_sac) < occluder_index
                    nsac_visible = nsac_visible + 1;
                    cumamp_vis = cumamp_vis + sac_amplitude(id_sac);
                end
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
    tmp_sac = diff(sac_index);
    void_sac_id = sac_index(tmp_sac==2);
    nsac = nsac - sum(tmp_sac==2);
    nsac_visible = nsac_visible - sum(void_sac_id < occluder_index);
    nsac_invisible = nsac - nsac_visible;
    cumamp_inv = cumamp - cumamp_vis;

    % find the last saccade onset and offset
    inv_sac_onset = final_sac_onset(final_sac_onset >= occluder_index);
    inv_sac_offset = final_sac_offset(final_sac_offset >= occluder_index);
    
    if ~isempty(inv_sac_onset)
        last_sac_onset = inv_sac_onset(end);
        if ~isempty(void_sac_id) 
            if inv_sac_onset(end) == void_sac_id
                last_sac_onset = inv_sac_onset(end-1);
            end
        end
        last_sac_offset = inv_sac_offset(end);
        first_sac_onset = inv_sac_onset(1)*time_unit;
        first_sac_offset = inv_sac_offset(1)*time_unit;
    end
    last_sac_onset = last_sac_onset*time_unit;
    last_sac_offset = last_sac_offset*time_unit;
    

    % plot
    plot_time_removed = plot_time;
    eye_position_removed = eye_position_trace;
    eye_velocity_removed = eye_velocity_trace;
    plot_time_removed(sac_index) = NaN;

    eye_position_removed(sac_index,:) = NaN;
    eye_velocity_removed(sac_index,:) = NaN;

    if makeplots
        subplot(nrow,ncol,4+(whicheye-1)*ncol);
        plot(plot_time_removed,eye_velocity_removed(:,vel_crit_col),colormat{whicheye}); hold on;
        plot(plot_time,target_velocity_trace(:,vel_crit_col),'k-'); hold on;
        xlabel('time (sec)');
        ylabel('velocity (deg/sec)');
        title(['saccade removed velocity ' eyetitle{whicheye}]);
        ylim([-40 40]);
%         axis([min(plot_time) max(plot_time) -40 40]);
    end

    %% --- detect pursuit

    dt_window_begin = 1;
    dt_window_end = min(dt_window_begin + dt_window, eyedur);

    while dt_window_begin < eyedur

        dt_dispersion = get_dispersion(dt_window_begin,dt_window_end,eye_position_trace);

        if dt_dispersion < dispersion_threshold
            while dt_dispersion < dispersion_threshold && dt_window_end < eyedur
                dt_window_end = dt_window_end + 1;
                dt_dispersion = get_dispersion(dt_window_begin,dt_window_end,eye_position_trace);
            end
            for i = dt_window_begin:dt_window_end
               if classified(i) ~= saccades
                   classified(i) = fixations;
               end % if
            end % for
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
        end % if
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

    if makeplots
        subplot(nrow,ncol,5+(whicheye-1)*ncol); 
        plot(plot_time(classified==fixations),eye_position_trace(classified==fixations,1),'b.'); hold on;
%         plot(plot_time(classified==pursuits),eye_position_trace(classified==pursuits,1),'r.'); hold on;
        plot(plot_time(classified==saccades),eye_position_trace(classified==saccades,1),'g.'); hold on;
        plot(plot_time,target_position_trace(:,1),'k-'); hold on;
        plot(target_time,target_position,'go'); hold on; % target location
        plot(occluder_time,occluder_position,'co');hold on;
        xlabel('time (sec)');
        ylabel('x position (deg)');
        title(['saccade removed position ' eyetitle{whicheye}]);
    end

    %% --- interpolation (velocity) 

    eye_velocity_interp = fillmissing(eye_velocity_removed(:,vel_crit_col),'linear','EndValues','extrap');
%     eye_velocity_interp = inpaint_nans(eye_velocity_removed(:,vel_crit_col));
%     eye_velocity_interp = fillmiss(eye_velocity_removed(:,vel_crit_col));


    %% --- Butterworth filter

    fc = 40; % cut off frequency
    fs = sample_rate;
    n_order = 2; % 2nd order 

    [b,a] = butter(n_order,fc/(fs/2)); %         freqz(b,a);

    filtered_velocity = filter(b,a,eye_velocity_interp);

    %% -- piecewise linear regression

    fit_plot_x = linspace(x_data(1),x_data(end),100);

    params0 = [0 0 2 -5 0.09];
    options = optimset('MaxFunEvals',100000,'MaxIter',100000);
    LB = [-1000 -1000 0 -1000 -1000];

    fit_range_id = fit_range*sample_rate;

    slide_start_id = min(find(plot_time>0));
    slide_end_id = occluder_index-fit_range_id;

    if slideornot == 1
        slide_range = slide_start_id:slide_end_id;
    else
        slide_range = slide_start_id;
    end

    pp = 1;
    breakpointmat = nan(length(slide_range),1);
    slope_firstmat = nan(length(slide_range),1);
    slope_secondmat = nan(length(slide_range),1);
    ic_firstmat = nan(length(slide_range),1);
    ic_secondmat = nan(length(slide_range),1);

    for whichslide = slide_range
        
        if whichslide+(fit_range_id)-1 <= length(plot_time)
            fit_index = whichslide:whichslide+(fit_range_id)-1;
            t_x_data = plot_time(fit_index);
            t_y_data = filtered_velocity(fit_index);

            [params_hat, sse] = fminsearch(@(params) costfun(params,t_x_data,t_y_data),params0,options);
            if slideornot == 0 
                if (params_hat(3) - params_hat(1)) > 0.25*curr_speed && params_hat(5) > 0.05 && params_hat(5) < .6
                    breakpointmat(pp) = params_hat(5);
                    slope_firstmat(pp) = params_hat(1);
                    ic_firstmat(pp) = params_hat(2);
                    slope_secondmat(pp) = params_hat(3);
                    ic_secondmat(pp) = params_hat(4);
                end
            elseif slideornot == 1
                if (params_hat(3) - params_hat(1)) > 0.25*curr_speed && params_hat(5) > 0.05 && params_hat(5) < occluder_time
                    breakpointmat(pp) = params_hat(5);
                    slope_firstmat(pp) = params_hat(1);
                    ic_firstmat(pp) = params_hat(2);
                    slope_secondmat(pp) = params_hat(3);
                    ic_secondmat(pp) = params_hat(4);
                end
            end
            
        end
        pp = pp + 1;

    end

    breakpoint = nanmean(breakpointmat);
    slope_first = nanmean(slope_firstmat);
    ic_first = nanmean(ic_firstmat);
    slope_second = nanmean(slope_secondmat);
    ic_second = nanmean(ic_secondmat);

    meanparams = [slope_first ic_first slope_second ic_second breakpoint];

    fit_plot_y = func_piecewise(meanparams,fit_plot_x');
    
    classified2 = classified; 
    classified2(plot_time >= breakpoint & plot_time <= pursuit_onset) = pursuits;
    
    if makeplots

        subplot(nrow,ncol,6+(whicheye-1)*ncol);
        plot(pursuit_time,target_velocity_trace(1:occluder_index),'k-'); hold on;
        plot(pursuit_time,filtered_velocity(1:occluder_index),'b-'); hold on; 
        if ~isempty(fit_plot_y)
            plot(fit_plot_x,fit_plot_y,'r-'); hold on;
        end
        plot(repmat(breakpoint,1,81),-40:40,'c--'); hold on;
        plot(repmat(pursuit_onset,1,81),-40:40,'g--'); hold on;
        title(['smooth pursuit period ' eyetitle{whicheye}]);
        xlabel('time (sec)');
        ylabel('velocity (deg/s)');
        axis([min(pursuit_time) max(pursuit_time) -40 40]);
        
        subplot(nrow,ncol,5+(whicheye-1)*ncol); 
        plot(plot_time(classified2==pursuits),eye_position_trace(classified2==pursuits,1),'r.'); hold on;
        
        subplot(nrow,ncol,4+(whicheye-1)*ncol);
        plot(repmat(breakpoint,1,81),-40:40,'c-'); hold on;
        plot(repmat(target_time,1,81),-40:40,'g--'); hold on; % target location
        plot(repmat(occluder_time,1,81),-40:40,'c--'); hold on;

    end

    %% --- pursuit score calculation 

    pursuit_start_visible = find(target_position_trace(:,1)>0);
    pursuit_start_visible_index = pursuit_start_visible(1);

    visible_range = pursuit_start_visible_index:occluder_index;
    invisible_range = occluder_index+1:eyedur;

    total_stim_pursuit_visible = occluder_index - pursuit_start_visible_index + 1;
    total_stim_pursuit_invisible = (eyedur - (occluder_index+1)) + 1;

    temp_pursuit = classified == pursuits;  
    total_pursuit_visible = sum(temp_pursuit(visible_range));
    total_pursuit_invisible = sum(temp_pursuit(invisible_range));

    PQnS_visible = 100 * (total_pursuit_visible/total_stim_pursuit_visible);
    PQnS_invisible = 100 * (total_pursuit_invisible/total_stim_pursuit_invisible);
    
    PQnST_invisible = 100 * (total_pursuit_invisible*time_unit/invisible_duration);

    %% --- fixation location calculation 

    temp_fixations = classified == fixations;

    begin_fixation_offset = find(temp_fixations==0,1) - 1;
    begin_fixation_onset = find(temp_fixations==1,1);
    end_fixation_onset = find(temp_fixations==0,1,'last') + 1;
    end_fixation_offset = find(temp_fixations==1,1,'last');

    if begin_fixation_onset - 1 < dt_window 
        begin_range = begin_fixation_onset:begin_fixation_offset;
        begin_fixation_location = [mean(eye_position_trace(begin_range,1)), mean(eye_position_trace(begin_range,2))];
    end
    if eyedur - end_fixation_offset < dt_window
        end_range = end_fixation_onset:end_fixation_offset;
        end_fixation_location = [mean(eye_position_trace(end_range,1)), mean(eye_position_trace(end_range,2))];
    end

    if makeplots
        subplot(nrow,ncol,1);
        plot(begin_fixation_location(1),begin_fixation_location(2),'bs'); hold on;
        plot(end_fixation_location(1),end_fixation_location(2),'bs'); hold on;
    end

    %% --- retinal slip calculation 
    % difference in velocity between the object and eye 

    slip_visible = mean(target_velocity_trace(visible_range,3) - filtered_velocity(visible_range));
    slip_invisible = mean(target_velocity_trace(invisible_range,3) - filtered_velocity(invisible_range));
    
    rel_slip_visible = mean((target_velocity_trace(visible_range,3) - filtered_velocity(visible_range))./filtered_velocity(visible_range));
    rel_slip_invisible = mean((target_velocity_trace(invisible_range,3) - filtered_velocity(invisible_range))./filtered_velocity(invisible_range));

    %% --- calculate pursuit gain
    % ratio of the average pursuit velocity to target 

    if ~sum(classified==pursuits) == 0 
        if ~isnan(breakpoint)    
            breakpoint_index = find(plot_time>breakpoint, 1);
            closed_range = closed_window_begin+breakpoint_index:closed_window_end+breakpoint_index;
            if closed_window_begin+breakpoint_index < occluder_index-1
                closed_vis_range = closed_window_begin+breakpoint_index:occluder_index-1;
                closed_vis_length = length(closed_vis_range);
                closed_vis = mean(filtered_velocity(closed_vis_range))/target_velocity_trace(1,3);
                slip_closed = mean(filtered_velocity(closed_vis_range)) - target_velocity_trace(1,3);
                rel_slip_closed = mean((filtered_velocity(closed_vis_range) - target_velocity_trace(1,3))./filtered_velocity(closed_vis_range));
            end
            if length(filtered_velocity) > breakpoint_index+open_window
                open_range = breakpoint_index:breakpoint_index+open_window;
                open_gain = mean(filtered_velocity(open_range))/target_velocity_trace(1,3);
                slip_open = mean(filtered_velocity(open_range))-target_velocity_trace(1,3);
                rel_slip_open = mean((filtered_velocity(open_range))-target_velocity_trace(1,3)./filtered_velocity(open_range));
                if length(filtered_velocity) < max(closed_range)
                    closed_range = closed_window_begin:length(filtered_velocity);
                end
                closed_gain = mean(filtered_velocity(closed_range))/target_velocity_trace(1,3); 
                peak_velo = max(filtered_velocity(open_range));
                peak_acc = max(diff(filtered_velocity(open_range)));
                rmse_position_open = rmse(eye_position_trace(open_range,:),target_position_trace(open_range,:));
                rmse_position_closed = rmse(eye_position_trace(closed_range,:),target_position_trace(closed_range,:));
            end
            invisible_gain = mean(filtered_velocity(invisible_range))/target_velocity_trace(1,3);
        end

    end

    %% --- calculate position rmse

    rmse_position = rmse(eye_position_trace,target_position_trace);
    rmse_position_visible = rmse(eye_position_trace(visible_range,:),target_position_trace(visible_range,:));
    rmse_position_invisible = rmse(eye_position_trace(invisible_range,:),target_position_trace(invisible_range,:));
    
    
    
    %% --- calculate SAD & SED
    
    if length(invisible_range) > 1
        eyedisp = sqrt( ( max(eye_position_trace(invisible_range,1))-min(eye_position_trace(invisible_range,1)) )^2 ...
            + ( max(eye_position_trace(invisible_range,2))-min(eye_position_trace(invisible_range,2)) )^2 );

        SAD = cumamp_inv/invisible_distance;
        SEDtotal = eyedisp - cumamp_inv;
        SED = SEDtotal/invisible_distance;
    end

    %% --- save filtered velocity traces
    
    if whicheye == 1
        VeloTraceMat1(whichtrial+(whichblock-1)*100,1:eyedur) = filtered_velocity;
    elseif whicheye == 2
        VeloTraceMat2(whichtrial+(whichblock-1)*100,1:eyedur) = filtered_velocity;
    end

end

%% --- save into a table

EyeTable = table(subid_cell,whichtrial,begin_fixation_location,end_fixation_location,...
    breakpoint,pursuit_onset,PQnS_visible,PQnS_invisible, ...
    slip_visible,slip_invisible,open_gain,closed_gain,invisible_gain,nsac,nsac_visible,nsac_invisible, peak_velo, peak_acc, ...
    rmse_position, rmse_position_invisible, total_pursuit, meanamp, meanlength, ...
    cumamp, cumamp_vis, cumamp_inv, closed_vis, closed_vis_length, slip_open, slip_closed, last_sac_onset, last_sac_offset, first_sac_onset, first_sac_offset, SAD, SED, SEDtotal, PQnST_invisible, total_pursuit_visible, total_pursuit_invisible, ...
    rmse_position_visible, rmse_position_open, rmse_position_closed, ...
    rel_slip_open, rel_slip_closed, rel_slip_visible, rel_slip_invisible);
if whicheye == 1
    if p == 1
        T_left = EyeTable;
    else
        T_left = [T_left; EyeTable];
    end
    saccades_left{whichtrial,1} = final_sac_onset;
    saccades_left{whichtrial,2} = final_sac_offset;
    behavior = [behavior; curr_speed visible_distance visible_duration invisible_distance invisible_duration error_distance error_time];
elseif whicheye == 2
    if p == 1
        T_right = EyeTable;
    else
        T_right = [T_right; EyeTable];
    end
    saccades_right{whichtrial,1} = final_sac_onset;
    saccades_right{whichtrial,2} = final_sac_offset;
end


%% --- clear data for next trial

clearvars -except ASD TD whichgroup whichsubject whichblock subid p T_left T_right ... 
    data_matrix eye_data lefteye righteye makeplots behavior VeloTraceMat1 VeloTraceMat2 ...
    whicheye slideornot whichtrial saccades_left saccades_right;




        

    
    




