
import numpy as np
from torch.utils.data import DataLoader
import torch
import scipy.io as sio

# def get_network_performance(model, test_pats, test_targets):
#     # this presents the test patterns and targets to see how well the trained network generalizes
#     # to the test patterns (which it did not see during training)
#     # this just gives you overall performance
#     eval_loss, eval_perf = model.evaluate(test_pats, test_targets)
#     # returns just the overall performance
#     return eval_perf

###--------------------------------------------------------------------
# Data loading and preprocessing
###--------------------------------------------------------------------
#-----------------------------------------------
# Load data and generate the Pytorch dataset
#-----------------------------------------------
def load_mat_data(filepath):

    data = sio.loadmat(filepath)

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][::16,:,8:10,:] #Downsample 16 times with ::4
    blink = data['data'][0, 0]['sub01'][0, 0]['blinkTrs'][::16,:,4:6,:]
    saccade = data['data'][0, 0]['sub01'][0, 0]['saccadesTrs'][::16,:,4:6,:]
    pursuit = data['data'][0, 0]['sub01'][0, 0]['pursuitTrs'][::16,:,4:6,:]
    
    eyelink_x = data['data'][0, 0]['sub01'][0, 0]['origx'][::16,:,4:6,:] #blinkFreeX

    conditions = data['data'][0,0]['eyelink_conds'][0][4:6]
    speed = data['data'][0, 0]['sub01'][0, 0]['driftingSpeeds'][:,8:10]

    # -------------------------
    # Clean direction labels
    # -------------------------
    dir_labels = []
    for item in conditions:
        string_val = str(item[0])
        if 'Leftward' in string_val:
            dir_labels.append(0)
        elif 'Rightward' in string_val:
            dir_labels.append(1)

    dir_labels = np.array(dir_labels)  # (n_conditions,)

    # -------------------------
    # Get dimensions
    # -------------------------
    T, n_trials, n_cond, n_ch = eog.shape

    # -------------------------
    # Reshape into trials
    # -------------------------
    X_list = []
    x_pos = []
    x_dir = []
    x_speed = []

    for c in range(n_cond):
        for t in range(n_trials):
            t_speed = speed[t,c]
            fix_dur = 1
            t_dur = 40/t_speed #40/t_speed #40 = target trajectory distance from -20 to 20 deg = 40 deg
            sample_dur = 2 #sample duration we want to keep for the training, as shortest trial = 2 seconds 
            fs = 2000/16 #downsampled 16 times from 2000 Hz to 125 Hz
            t_samples = np.floor(t_dur*fs)
            samples = sample_dur*fs
            # EOG → (channels, time)
            sampleStart = int(fix_dur*fs+np.floor((t_samples)/2)-(samples)/2) #take 2 secs at the middle of the trial for each trial
            sampleEnd = int(fix_dur*fs+np.floor((t_samples)/2)+(samples)/2)
            eog_trial = eog[sampleStart:sampleEnd, t, c, :].T 

            blink_trial = blink[sampleStart:sampleEnd, t, c, :].T
            sacc_trial = saccade[sampleStart:sampleEnd, t, c, :].T
            pursuit_trial = pursuit[sampleStart:sampleEnd, t, c, :].T

            # Concatenate channels
            trial = np.concatenate([
                eog_trial,
                blink_trial,
                sacc_trial,
                pursuit_trial
            ], axis=0)  # (channels_total (n_variables*n_eyes), time)

            X_list.append(trial)
            x_pos.append(eyelink_x[sampleStart:sampleEnd,t, c,:].T)
            x_dir.append(dir_labels[c])
            x_speed.append(speed[t, c])

    X = np.array(X_list)          # (N_trials, channels, time)
    # X = np.nan_to_num(X, nan=0.0)
    x_pos = np.array(x_pos)
    x_dir = np.array(x_dir)
    x_speed = np.array(x_speed)

    return X, x_pos, x_dir, x_speed


#-----------------------------------------------
# Load data 1 eye at a time
#-----------------------------------------------
def load_mat_data_1eye(filepath):
    #filepath = op.join(data_dir, f"{participant}_v05", f"{participant}_model_data_filt_audio_visual_motion_v05_1.mat")
    data = sio.loadmat(filepath)

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][::16,:,8:10,:] #Downsample 16 times with ::4
    blink = data['data'][0, 0]['sub01'][0, 0]['blinkTrs'][::16,:,4:6,:]
    saccade = data['data'][0, 0]['sub01'][0, 0]['saccadesTrs'][::16,:,4:6,:]
    pursuit = data['data'][0, 0]['sub01'][0, 0]['pursuitTrs'][::16,:,4:6,:]
    
    eyelink_x = data['data'][0, 0]['sub01'][0, 0]['origx'][::16,:,4:6,:] #blinkFreeX

    conditions = data['data'][0,0]['eyelink_conds'][0][4:6]
    speed = data['data'][0, 0]['sub01'][0, 0]['driftingSpeeds'][:,8:10]

    # -------------------------
    # Clean direction labels
    # -------------------------
    dir_labels = []
    for item in conditions:
        string_val = str(item[0])
        if 'Leftward' in string_val:
            dir_labels.append(0)
        elif 'Rightward' in string_val:
            dir_labels.append(1)

    dir_labels = np.array(dir_labels)  # (n_conditions,)

    # -------------------------
    # Get dimensions
    # -------------------------
    T, n_trials, n_cond, n_ch = eog.shape

    # -------------------------
    # Reshape into trials
    # -------------------------
    X_list = []
    x_pos = []
    x_dir = []
    x_speed = []

    for c in range(n_cond): #motion direction (leftward/rightward)
        for t in range(n_trials):
            for e in range(n_ch):
                t_speed = speed[t,c]
                fix_dur = 1
                t_dur = 40/t_speed #40/t_speed #40 = target trajectory distance from -20 to 20 deg = 40 deg
                sample_dur = 2 #sample duration we want to keep for the training, as shortest trial = 2 seconds 
                fs = 2000/16 #downsampled 16 times from 2000 Hz to 125 Hz
                t_samples = np.floor(t_dur*fs)
                samples = sample_dur*fs
                # EOG → (channels, time)
                sampleStart = int(fix_dur*fs+np.floor((t_samples)/2)-(samples)/2) #take 2 secs at the middle of the trial for each trial
                sampleEnd = int(fix_dur*fs+np.floor((t_samples)/2)+(samples)/2)
                eog_trial = eog[sampleStart:sampleEnd, t, c, e]

                blink_trial = blink[sampleStart:sampleEnd, t, c, e]
                sacc_trial = saccade[sampleStart:sampleEnd, t, c, e]
                pursuit_trial = pursuit[sampleStart:sampleEnd, t, c, e]

                # Concatenate channels
                trial = np.stack([
                    eog_trial,
                    blink_trial,
                    sacc_trial,
                    pursuit_trial
                ], axis=0)  # (channels_total (n_variables*n_eyes), time)

                X_list.append(trial)
                x_pos.append(eyelink_x[sampleStart:sampleEnd,t, c, e])
                x_dir.append(dir_labels[c])
                x_speed.append(speed[t, c])

    X = np.array(X_list)          # (N_trials, channels, time)
    # X = np.nan_to_num(X, nan=0.0)
    x_pos = np.array(x_pos)
    x_dir = np.array(x_dir)
    x_speed = np.array(x_speed)

    return X, x_pos, x_dir, x_speed

#-----------------------------------------------
# Interpolate data when there are blinks or eye position is too far away from the screen
#-----------------------------------------------
#Usage:
        # X_train_eog = ch.interpolate_gaps(X_train[:, :2, :],train_mask)
        # x_pos_train = ch.interpolate_gaps(x_pos_train, train_mask)
import numpy as np

def interpolate_gaps(data, mask):
    #data = x_pos_train.copy()
    data_interp = data.copy()

    n_trials, n_channels, n_timepoints = data.shape

    for trial in range(n_trials):
        for ch in range(n_channels):

            m = mask[trial, ch]

            # contiguous regions
            diff = np.diff(m.astype(int))

            starts = np.where(diff == 1)[0] + 1
            ends   = np.where(diff == -1)[0]

            if m[0]:
                starts = np.r_[0, starts]

            if m[-1]:
                ends = np.r_[ends, n_timepoints - 1]

            for start, end in zip(starts, ends):

                # ------------------------------------------------
                # Find nearest valid LEFT anchor
                # ------------------------------------------------
                left_idx = start - 1

                while left_idx >= 0 and np.isnan(
                    data_interp[trial, ch, left_idx]
                ):
                    left_idx -= 1

                # ------------------------------------------------
                # Find nearest valid RIGHT anchor
                # ------------------------------------------------
                right_idx = end + 1

                while (
                    right_idx < n_timepoints
                    and np.isnan(
                        data_interp[trial, ch, right_idx]
                    )
                ):
                    right_idx += 1

                # skip if no valid anchors
                if left_idx < 0 or right_idx >= n_timepoints:
                    continue

                left_val = data_interp[trial, ch, left_idx]
                right_val = data_interp[trial, ch, right_idx]

                # still skip if anchors invalid
                if np.isnan(left_val) or np.isnan(right_val):
                    continue

                # interpolate ONLY inside gap
                gap_len = end - start + 1

                interp_vals = np.linspace(
                    left_val,
                    right_val,
                    gap_len + 2
                )[1:-1]

                data_interp[
                    trial,
                    ch,
                    start:end+1
                ] = interp_vals

    return data_interp

def get_network_output(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    
    # all_dir = []
    # all_speed = []
    # all_targets_dir = []
    
    all_pos = []
    all_targets_pos = []
    all_targets_speed = []

    with torch.no_grad():
        for X, x_pos_norm, x_pos, x_speed in loader:
            pred_pos = model(X)
            # pred_dir, pred_speed = model(X)
            
            all_pos.append(pred_pos.cpu())
            all_targets_pos.append(x_pos_norm.cpu())
            # all_dir.append(pred_dir.cpu())
            # all_speed.append(pred_speed.cpu())
            # all_targets_dir.append(x_dir.cpu())
            all_targets_speed.append(x_speed.cpu())

    # pred_dir = torch.cat(all_dir).numpy()
    # pred_speed = torch.cat(all_speed).numpy()
    pred_pos = torch.cat(all_pos).numpy()
    # y_dir = torch.cat(all_targets_dir).numpy()
    
    x_pos_norm = torch.cat(all_targets_pos).numpy()
    x_speed = torch.cat(all_targets_speed).numpy()

    return pred_pos, x_pos_norm, x_speed

def get_direction_performance(pred_dir, y_dir):
    pred_labels = np.argmax(pred_dir, axis=1)
    acc = np.mean(pred_labels == y_dir)
    return acc

# def get_pos_performance(pred_pos, x_pos):
#     rmse = np.sqrt(np.mean((pred_pos.squeeze() - x_pos)**2))
#     return rmse

def get_pos_performance(pred_pos, x_pos):
    pred_pos = pred_pos.squeeze()

    valid = (~np.isnan(pred_pos)) & (~np.isnan(x_pos))

    rmse = np.sqrt(np.mean((pred_pos[valid] - x_pos[valid])**2))
    return rmse

# def get_pos_performance(pred_pos, x_pos):
#     pred_pos = pred_pos.squeeze()

#     n_trials, n_eyes, n_timepts = pred_pos.shape()
#     rmse = np.full(n_eyes, np.nan)

#     for eye in range(n_eyes):

#         valid = (
#             ~np.isnan(pred_pos[:, eye, :]) &
#             ~np.isnan(x_pos[:, eye, :])
#         )

#         rmse[eye] = np.sqrt(
#             np.mean(
#                 (pred_pos[:, eye, :][valid] -
#                  x_pos[:, eye, :][valid])**2
#             )
#         )
#     return rmse

# def get_pos_performance_per_speed(pred_pos, x_pos, x_speed):
#     pred_pos = pred_pos.squeeze()
#     speed_list= np.unique(x_speed)
#     rmse_per_speed = np.full(len(speed_list), np.nan)

#     for i, speed in enumerate(speed_list):
#         idx = x_speed == speed

#         if np.sum(idx) > 0:
#             rmse_per_speed[i] = np.sqrt(
#                 np.mean((pred_pos[idx] - x_pos[idx])**2)
#             )

#     return rmse_per_speed


def get_pos_performance_per_speed(pred_pos, x_pos, x_speed):
    pred_pos = pred_pos.squeeze()

    speed_list = np.unique(x_speed)
    rmse_per_speed = np.full(len(speed_list), np.nan)

    for i, speed in enumerate(speed_list):

        idx = (
        (x_speed[:, None, None] == speed) &
        (~np.isnan(pred_pos)) &
        (~np.isnan(x_pos))
        )

        if np.any(idx):
            rmse_per_speed[i] = np.sqrt(
                np.mean((pred_pos[idx] - x_pos[idx])**2)
            )

    return rmse_per_speed

def get_network_cat_performance(pred_out, targets, n_classes):
    pred_labels = np.argmax(pred_out, axis=1)

    cat_perf = np.zeros(n_classes)

    for i in range(n_classes):
        idx = targets == i

        if np.sum(idx) == 0:
            continue

        correct = np.sum(pred_labels[idx] == i)
        cat_perf[i] = correct / np.sum(idx)

    return cat_perf


### Figures
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os


def plot_traces(
    X,
    y_dir,
    y_speed,
    subjects,
    save_path,
    save_name,
    ylim,
    channel_idx=0,
    time=None,
    alpha_trials=0.05,
    alpha_subject=0.8,
):
    """
    Plot EOG traces across trials and subjects.

    Parameters
    ----------
    X : array (N_trials, channels, time)
    y_dir : array (N_trials,) 0=Left, 1=Right
    y_speed : array (N_trials,)
    subjects : array (N_trials,)
    channel_idx : int (which channel to plot)
    time : array (time axis)
    """

    directions = {0: "Leftward", 1: "Rightward"}
    unique_speeds = np.unique(y_speed)

    if time is None:
        time = np.arange(X.shape[-1])

    fig, axes = plt.subplots(
        len(unique_speeds),
        2,
        figsize=(12, 4 * len(unique_speeds)),
        sharex=True,
        sharey=True
    )

    if len(unique_speeds) == 1:
        axes = np.array([axes])

    for i, speed in enumerate(unique_speeds):
        for j, d in enumerate([0, 1]):

            ax = axes[i, j]

            # select trials
            mask = (y_speed == speed) & (y_dir == d)
            X_sel = X[mask]
            subj_sel = subjects[mask]

            if len(X_sel) == 0:
                continue

            # -------------------------
            # Plot all trials (light)
            # -------------------------
            for trial in X_sel:
                ax.plot(
                    time,
                    trial[channel_idx],
                    color="gray",
                    alpha=alpha_trials,
                    linewidth=0.5
                )

            # -------------------------
            # Plot subject averages
            # -------------------------
            unique_subjects = np.unique(subj_sel)

            for sub in unique_subjects:
                sub_mask = subj_sel == sub
                sub_mean = X_sel[sub_mask].mean(axis=0)

                ax.plot(
                    time,
                    sub_mean[channel_idx],
                    linewidth=2,
                    alpha=alpha_subject,
                    label=sub
                )

            # -------------------------
            # Titles
            # -------------------------
            ax.set_title(f"{directions[d]} | Speed={speed}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"Channel {channel_idx}")
            ax.set_ylim(ylim)

    # cleaner legend (only once)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Subjects")

    plt.tight_layout()
    saveDir = op.join(save_path)
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(saveDir, save_name), dpi=300, bbox_inches='tight')
    plt.show()

#Plot trials data
import matplotlib.pyplot as plt
import numpy as np

def plot_trial(
    data,
    trial_idx=0,
    original_data=None,
    channel_names=("Left eye", "Right eye")
):
    """
    Plot EOG traces for one trial.

    Parameters
    ----------
    data : ndarray
        Shape (n_trials, n_channels, n_timepoints)

    trial_idx : int
        Trial to plot.

    original_data : ndarray or None
        Optional original data before interpolation.

    channel_names : tuple
        Names for channels.
    """

    n_trials, n_channels, n_timepoints = data.shape

    time = np.arange(n_timepoints)

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(12, 4 * n_channels),
        sharex=True
    )

    # if only one channel
    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):

        # interpolated / cleaned data
        axes[ch].plot(
            time,
            data[trial_idx, ch],
            linewidth=2,
            label="Interpolated"
        )

        # optional original data
        if original_data is not None:
            axes[ch].plot(
                time,
                original_data[trial_idx, ch],
                alpha=0.5,
                linestyle="--",
                label="Original"
            )

        axes[ch].set_title(
            f"Trial {trial_idx} — {channel_names[ch]}"
        )

        axes[ch].set_ylabel("Amplitude")
        axes[ch].legend()

    axes[-1].set_xlabel("Timepoints")

    plt.tight_layout()
    plt.show()
#========================================================================================
#
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op

def plot_pred_vs_true(
    x_pos,
    x_pred,
    subjects=None,
    save_path=".",
    save_name="figure",
    channel_idx=0,
    time=None,
    max_trials=20,          # avoid plotting hundreds
    ylim =None,
):
    """
    Plot predicted vs true eye position traces per trial.

    Parameters
    ----------
    x_pos : array (N, T) or (N, C, T)
    x_pred : array (N, T) or (N, C, T)
    subjects : array (N,) optional
    channel_idx : which channel to plot if C > 1
    time : array (T,)
    """

    # -------------------------
    # Handle shapes
    # -------------------------
    if x_pos.ndim == 3:
        x_pos = x_pos[:, channel_idx, :]
        x_pred = x_pred[:, channel_idx, :]

    num_trials = min(x_pos.shape[0], max_trials)

    if time is None:
        time = np.arange(x_pos.shape[-1])

    # -------------------------
    # Plot
    # -------------------------
    fig, axes = plt.subplots(
        num_trials, 1,
        figsize=(10, 2.5 * num_trials),
        sharex=True
    )

    if num_trials == 1:
        axes = [axes]

    for i in range(num_trials):

        ax = axes[i]

        ax.plot(time, x_pos[i], label="True", linewidth=2)
        ax.plot(time, x_pred[i], label="Pred", linestyle="--")

        if subjects is not None:
            ax.set_title(f"Trial {i} | Subject: {subjects[i]}")
        else:
            ax.set_title(f"Trial {i}")

        ax.set_ylabel("Position")

        if i == 0:
            ax.legend()
        ax.set_ylim(ylim)

    axes[-1].set_xlabel("Time")

    plt.tight_layout()

    # -------------------------
    # Save
    # -------------------------
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(op.join(save_path, save_name), dpi=300)
    plt.show()