function [open_gain, closed_gain] = pursuit_velocity_gain(classified, eye_position_trace, filtered_velocity, target_position_trace, target_velocity_trace, sample_rate, curr_speed, tnum)
%Loic Daumail 10/07/2025
%curr_speed = target speed? in deg/sec
%curr_speed = driftingSpeedsCond(whichtrial);


pursuits = 222;
open_gain = NaN; 
closed_gain = NaN;
breakpoint = NaN;
eyedur = length(eye_position_trace(:,1));
time_unit = (1000/sample_rate)/1000; %interval between each sampled data point in second
initialFixDur = 1;
plot_time = [0:eyedur-1]';
plot_time = (plot_time .* time_unit);

%% Step 1 -- piecewise linear regression
%This step is to detect pursuit onset time
%The break point is the point in time for which the first and second
%regression cross

fit_range = 0.3; % sec 
fit_index = find(plot_time>initialFixDur & plot_time<initialFixDur+fit_range);

x_data = plot_time(fit_index);

params0 = [0 0 2 -5 initialFixDur+0.09]; %slope 1, ic1, slope 2, ic2, breakpoint =initial params
options = optimset('MaxFunEvals',100000,'MaxIter',100000);

% slideornot      = 1;
% fit_range_id = fit_range*sample_rate;
% 
% slide_start_id = min(find(plot_time>0))+initialFixDur*sample_rate;
% slide_end_id = max(find(plot_time>0 & plot_time<=fit_range))+initialFixDur*sample_rate;

% if slideornot == 1
% slide_range = slide_start_id:slide_end_id;
% else
%  slide_range = slide_start_id;
% end
% 
% pp = 1;
% breakpointmat = nan(length(slide_range),1);
% slope_firstmat = nan(length(slide_range),1);
% slope_secondmat = nan(length(slide_range),1);
% ic_firstmat = nan(length(slide_range),1);
% ic_secondmat = nan(length(slide_range),1);
% 
% for whichslide = slide_range
% 
%     if whichslide+(fit_range_id)-1 <= length(plot_time)
%         fit_index = whichslide:whichslide+(fit_range_id)-1;
%         t_x_data = plot_time(fit_index);
%         t_y_data = filtered_velocity(fit_index);
% 
%         [params_hat, sse] = fminsearch(@(params) costfun(params,t_x_data,t_y_data),params0,options);
%         if slideornot == 0
%             if (params_hat(3) - params_hat(1)) > 0.25*curr_speed && params_hat(5) > 0.05 && params_hat(5) < .6
%                 breakpointmat(pp) = params_hat(5);
%                 slope_firstmat(pp) = params_hat(1);
%                 ic_firstmat(pp) = params_hat(2);
%                 slope_secondmat(pp) = params_hat(3);
%                 ic_secondmat(pp) = params_hat(4);
%             end
%         elseif slideornot == 1
%             if (params_hat(3) - params_hat(1)) > 0.25*curr_speed && params_hat(5) > 0.05 && params_hat(5) < x_data(end)
%                 breakpointmat(pp) = params_hat(5);
%                 slope_firstmat(pp) = params_hat(1);
%                 ic_firstmat(pp) = params_hat(2);
%                 slope_secondmat(pp) = params_hat(3);
%                 ic_secondmat(pp) = params_hat(4);
%             end
%         end
% 
%     end
%     pp = pp + 1;
% 
% end
% 
% breakpoint = nanmean(breakpointmat);
% slope_first = nanmean(slope_firstmat);
% ic_first = nanmean(ic_firstmat);
% slope_second = nanmean(slope_secondmat);
% ic_second = nanmean(ic_secondmat);
% meanparams = [slope_first ic_first slope_second ic_second breakpoint];


t_x_data = plot_time(fit_index);
t_y_data = filtered_velocity(fit_index);

[params_hat, sse] = fminsearch(@(params) costfun(params,t_x_data,t_y_data),params0,options);

if (params_hat(3) - params_hat(1)) > 0.25*curr_speed && params_hat(5) >= initialFixDur+0.05 && params_hat(5) < plot_time(end) %params_hat(5) > 0.05
    breakpoint = params_hat(5);
    slope_first = params_hat(1);
    ic_first = params_hat(2);
    slope_second = params_hat(3);
    ic_second = params_hat(4);
end


% classified2 = classified;
% classified2(plot_time >= breakpoint & plot_time <= pursuit_onset) = pursuits;
% 
% 
% figure('Position', [0 0 2500 1000]);
% subplot(3,1,1)
% plot(plot_time(1:length(target_velocity_trace)),target_velocity_trace,'k-');
% hold on;
% plot(plot_time,filtered_velocity,'b-');
% hold on;
% fit_plot_x = linspace(x_data(1),x_data(end),100);
% fit_plot_y = func_piecewise(params_hat,fit_plot_x');
% if ~isempty(fit_plot_y)
%     plot(fit_plot_x,fit_plot_y,'r-');
%     hold on;
% end
% plot(repmat(breakpoint,1,81),-40:40,'c--'); hold on;
% % plot(repmat(pursuit_onset,1,81),-40:40,'g--'); hold on;
% if ~isnan(breakpoint)
%     title(['smooth pursuit period, breakpoint = Success']);
% else
%     title(['smooth pursuit period, breakpoint = Fail ']);
% end
% xlabel('time (sec)');
% ylabel('velocity (deg/s)');
% axis([min(plot_time(1:length(target_velocity_trace))) max(plot_time(1:length(target_velocity_trace))) -40 40]);
% legend('target velocity', 'filtered left eye velocity', 'regression fit', 'breakpoint')
% 
% subplot(3,1,2);
% % plot(plot_time(classified2==pursuits),eye_position_trace(classified2==pursuits,1),'r.'); hold on;
% plot(plot_time,eye_position_trace(:,1),'r.'); hold on;
% axis([min(plot_time(1:length(target_velocity_trace))) max(plot_time(1:length(target_velocity_trace))) -40 40]);
% 
% legend('smooth pursuit')
% subplot(3,1,3);
% plot(repmat(params_hat(5),1,81),-40:40,'c-'); hold on;
% legend('breakpoint')
% plotdir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/analysis_plots/');
% saveas(gcf,strcat(plotdir, sprintf('%s_trial%d_eyelink_regression_%s.jpg','sub-02',tnum,'v04')));

% saveas(gcf,strcat(plotdir, sprintf('%s_%s_trial%d_eyelink_regression_%s.jpg',name,conds{c},whichtrial,version)));



%%% --- calculate pursuit gain
% ratio of the average pursuit velocity to target
open_window_time = 100; % ms %Open loop for the first 100 to 200 ms, when the eye is not perfectly tracking the target yet
open_window = open_window_time / (1000/sample_rate); 
closed_window_begin = 200 / (1000/sample_rate); % index %
closed_window_end = length(target_position_trace); %400 / (1000/sample_rate); % index

if ~sum(classified==pursuits) == 0
    if ~isnan(breakpoint)
        breakpoint_index = find(plot_time>breakpoint, 1);
        closed_range = closed_window_begin+breakpoint_index:closed_window_end;
        if closed_window_begin+breakpoint_index < length(plot_time)-1
            closed_vis_range = closed_window_begin+breakpoint_index:length(plot_time)-1;
            closed_vis_length = length(closed_vis_range);
            closed_vis = mean(filtered_velocity(closed_vis_range))/target_velocity_trace(1,initialFixDur*sample_rate+1);
            slip_closed = mean(filtered_velocity(closed_vis_range)) - target_velocity_trace(1,initialFixDur*sample_rate+1);
            rel_slip_closed = mean((filtered_velocity(closed_vis_range) - target_velocity_trace(1,initialFixDur*sample_rate+1))./filtered_velocity(closed_vis_range));
        end
        if length(filtered_velocity) > breakpoint_index+open_window
            open_range = breakpoint_index:breakpoint_index+open_window;
            open_gain = mean(filtered_velocity(open_range))/target_velocity_trace(1,initialFixDur*sample_rate+2);
            slip_open = mean(filtered_velocity(open_range))-target_velocity_trace(1,initialFixDur*sample_rate+2);
            rel_slip_open = mean((filtered_velocity(open_range))-target_velocity_trace(1,initialFixDur*sample_rate+2)./filtered_velocity(open_range));
            if length(filtered_velocity) < max(closed_range)
                closed_range = closed_window_begin:length(filtered_velocity);
            end
            closed_gain = mean(filtered_velocity(closed_range))/target_velocity_trace(1,initialFixDur*sample_rate+2);
            peak_velo = max(filtered_velocity(open_range));
            peak_acc = max(diff(filtered_velocity(open_range)));
            rmse_position_open = rmse(eye_position_trace(open_range,:),[target_position_trace(open_range)', repmat(0,length(open_range), 1)]); %X and Y target position: [target_position_trace(open_range)', repmat(0,length(open_range), 1)]
            rmse_position_closed = rmse(eye_position_trace(closed_range,:),[target_position_trace(closed_range)', repmat(0,length(closed_range), 1)]); %
        end
    end

end

end