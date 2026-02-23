channels = [7 8];
labels = {'Left eye', 'Right eye'};
fc = 5;
figure
for i = 1:2
    raw = acq_data.data(:, channels(i));

    [b,a] = butter(order, fc/nyq, 'low');
    filt = filtfilt(b,a, raw);

    N = length(raw);
    freq = (0:N-1)*(fs/N);
    half = 1:floor(N/2);

    P_raw  = abs(fft(raw)).^2 / N;
    P_filt = abs(fft(filt)).^2 / N;

    subplot(1,2,i)
    hold on
    plot(freq(half), 10*log10(P_raw(half)),  'k')
    plot(freq(half), 10*log10(P_filt(half)), 'r', 'LineWidth', 1.5)
    xline(fc, '--b')

    title(labels{i})
    xlabel('Frequency (Hz)')
    ylabel('Power (dB)')
    xlim([0 100])
    grid on
    legend('Raw', 'Filtered')
end

sgtitle('EOG Power Spectrum Before vs After Low-Pass Filtering')




% sample_rate = 2000;
% nyq = sample_rate / 2;
% 
% fc_low  = 0.1;
% fc_high = 20;
% order_bp = 4; % Note: a 4th order bandpass results in 8th order total
% 
% 
% % -------------------------
% % Check for NaNs in Raw Data
% % -------------------------
% % If your raw data has even one NaN, filtfilt will turn the whole signal to NaN.
% raw = acq_data.data(:,7); 
% if any(isnan(raw))
%     fprintf('Warning: Raw data contains NaNs. Interpolating...\n');
%     raw = fillmissing(raw, 'linear');
% end
% 
% % Downsample from 2000Hz to 200Hz
% ds_factor = 10; 
% raw_ds = resample(raw, 1, ds_factor); 
% new_fs = sample_rate / ds_factor;
% 
% % Now filter at the lower sample rate
% nyq_ds = new_fs / 2;

%-------
% Data
%-------
names = {'sub-10'};
version = '05';
name = names{1};
folderDir = strcat('/Users/ldaumail3/Documents/research/eyetracking_EOG/Audio_Visual_EOG-main/data/',...
    sprintf('v%s/%s_v%s',version, name,version), '/');
fileNames = dir(folderDir);
acqName = strcat(folderDir,fileNames(contains({fileNames.name},'eog.mat')).name);
acq_data = load(acqName);

% -------------------------
% Parameters
% -------------------------
sample_rate = 2000;
ds_factor = 5;
new_fs = sample_rate / ds_factor;
nyq_ds = new_fs / 2;

fc_hp = 0.1;       % Butterworth High-pass
fc_lp = 20;        % Butterworth Low-pass
order = 4;         % Filter order
% stop_atten = 60;   % Attenuation in stopband (dB)

% -------------------------
% 1. Data Cleaning & Downsampling
% -------------------------
raw = acq_data.data(:,7);

% Crucial: Fix NaNs before resampling or filtering
if any(isnan(raw))
    raw = fillmissing(raw, 'linear');
end

% Downsample to improve numerical conditioning
raw_ds = resample(raw, 1, ds_factor);

% -------------------------
% 2. Design the Filters (SOS format)
% -------------------------
% High-pass: Butterworth (Maximally flat in the passband)
[sos_hp, g_hp] = butter(order, [fc_hp  fc_lp]/nyq_ds, 'bandpass');
%[sos_hp, g_hp] = butter(order, fc_hp/nyq_ds, 'high');

% Low-pass: Chebyshev Type II (Guaranteed 60dB drop above 35Hz)
% [sos_lp, g_lp] = butter(order, fc_lp/nyq_ds, 'low');
% [sos_lp, g_lp] = cheby2(order, stop_atten, fc_lp/nyq_ds, 'low');

% -------------------------
% 3. Apply Sequentially
% -------------------------
% First, remove the slow drift
hp_data = filtfilt(sos_hp, g_hp, raw_ds);

% Second, remove the high-frequency noise
% final_data = filtfilt(sos_lp, g_lp, hp_data);

% -------------------------
% 4. Verification (Visualization)
% -------------------------
figure;
hold on;
[pxx_raw, f_raw] = pwelch(raw_ds, [], [], [], new_fs);
[pxx_filt, f_filt] = pwelch(hp_data, [], [], [], new_fs);

plot(f_raw, 10*log10(pxx_raw), 'Color', [0.7 0.7 0.7], 'DisplayName', 'Downsampled Raw');
plot(f_filt, 10*log10(pxx_filt), 'b', 'LineWidth', 1.5, 'DisplayName', 'Hybrid Filtered');

xline(fc_lp, '--r', '20Hz Cutoff');
xline(fc_hp, '--r', '0.1Hz Cutoff');
grid on;
xlim([0 100])
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
title('Power Spectral Density Comparison');
legend('Raw', 'Band-pass (0.1–20 Hz)') 

% % [sos, g] = butter(order_bp, [fc_low  fc_high]/nyq_ds, 'bandpass'); % Generate Second order sections (SOS) and G (Gain)
% % fvtool(sos); % Opens the Filter Visualization Tool
% % -------------------------
% % 3. Apply the filter
% % -------------------------
% % Note: We pass 'sos' and 'g' into filtfilt
% filt_bp = filtfilt(sos, g, raw);
% 
% N = length(raw_ds);
% freq = (0:N-1)*(fs/N);
% half = 1:floor(N/2);
% 
% P_raw = abs(fft(raw)).^2 / N;
% % P_hp  = abs(fft(filt_hp)).^2 / N;
% P_bp  = abs(fft(filt_bp)).^2 / N;
% 
% figure; hold on
% plot(freq(half), 10*log10(P_raw(half)), 'k')
% hold on
% % plot(freq(half), 10*log10(P_hp(half)),  'b')
% plot(freq(half), 10*log10(P_bp(half)),  'r', 'LineWidth', 1.5)
% 
% xline(fc_low, '--b', '0.1 Hz HP')
% xline(fc_high,  '--r', '20 Hz LP')
% 
% xlabel('Frequency (Hz)')
% ylabel('Power (dB)')
% title('EOG Power Spectrum: Raw vs High-pass vs Band-pass')
% legend('Raw', 'Band-pass (0.1–20 Hz)') %'High-pass (0.1 Hz)',
% xlim([0 100])
% grid on
% 
