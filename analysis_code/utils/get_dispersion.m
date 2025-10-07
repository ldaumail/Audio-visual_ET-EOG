function dt_dispersion = get_dispersion(dt_window_begin, dt_window_end,eye_position_trace)

dt_max_x = max(eye_position_trace(dt_window_begin:dt_window_end,1));
dt_max_y = max(eye_position_trace(dt_window_begin:dt_window_end,2));
dt_min_x = min(eye_position_trace(dt_window_begin:dt_window_end,1));
dt_min_y = min(eye_position_trace(dt_window_begin:dt_window_end,2));

dt_dispersion = abs(dt_max_x-dt_min_x) + abs(dt_max_y-dt_min_y);

end