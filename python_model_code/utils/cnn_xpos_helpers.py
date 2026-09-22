
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

#===================================================
### Figures
#===================================================
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

# ==================
# Plot RMSE per speed across cross-validations
# ================== 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op

def plot_speed_rmse_duration_comparison(results,
                                   participant_colors,
                                   save_path=None,
                                   filename="speed_specific_prediction_rmse_by_duration.png"):

    durations = sorted(results.keys())
    n_durations = len(durations)

    fig, axes = plt.subplots(
        1,
        n_durations,
        figsize=(5*n_durations, 6),
        sharey=True
    )

    if n_durations == 1:
        axes = [axes]

    for ax, duration in zip(axes, durations):

        crossvalit_perfs = results[duration]["speed_rmse"]

        # (participants, speeds, CV)
        n_participants, n_speeds, n_cv = crossvalit_perfs.shape

        # Average across CV folds
        mean_perfs = np.nanmean(crossvalit_perfs, axis=2)

        participants = np.repeat(np.arange(n_participants), n_speeds)
        speeds = np.tile(np.arange(n_speeds)+5, n_participants)

        df = pd.DataFrame({
            "participant": participants,
            "speed": speeds,
            "RMSE": mean_perfs.reshape(-1)
        })

        order = np.sort(df["speed"].unique())
        speed_to_idx = {s: i for i, s in enumerate(order)}

        # Violin
        sns.violinplot(
            x="speed",
            y="RMSE",
            data=df,
            order=order,
            ax=ax,
            inner=None,
            color="lightgray"
        )

        # Individual participants
        sns.stripplot(
            x="speed",
            y="RMSE",
            data=df,
            order=order,
            jitter=True,
            size=5,
            ax=ax
        )

        # Connect participant points
        for p in range(n_participants):

            sub = df[df["participant"] == p]

            x_vals = [speed_to_idx[s] for s in sub["speed"]]

            ax.plot(
                x_vals,
                sub["RMSE"],
                color=participant_colors[p],
                alpha=0.7,
                linewidth=1.5
            )

        # Mean ± SEM
        means = df.groupby("speed")["RMSE"].mean()
        sems = df.groupby("speed")["RMSE"].sem()

        ax.errorbar(
            np.arange(len(order)),
            means.loc[order],
            yerr=sems.loc[order],
            fmt='-o',
            color='black',
            linewidth=2,
            capsize=4,
            zorder=10
        )

        ax.set_title(f"{duration:.1f} s", fontsize=18, fontweight="bold")
        ax.set_xlabel("Speed (deg/s)", fontsize=14)
        ax.set_ylim(0, 1.8)

    axes[0].set_ylabel("RMSE", fontsize=14)

    fig.suptitle(
        "Prediction error across stimulus speeds and durations",
        fontsize=20,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0,0,1,0.95])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            op.join(save_path, filename),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

#========================================================================================
#Plot predicted vs true traces

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
    Plot predicted vs true (eyelink) gaze position traces per trial.

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

# Same data but plotted differently with true = x,  pred = y 

def plot_pred_vs_true_position_clouds(
    results,
    speeds=None,
    durations=None,
    subjects=None,
    max_trials=None,
    channel_idx=0,
    save_path=None,
    filename="predicted_vs_true_position_clouds",
    alpha=0.15,
    point_size=5,
    xlim=None,
    ylim=None,
):
    """

    Plot predicted eye position vs true eyelink gaze position as point clouds,
    organized by stimulus speed and sample duration.

    Creates multiple 4x4 figures rather than one large figure.
    Each figure contains 16 speeds, with durations represented
    by separate figures.

    Each trial is plotted in a different color.

    Parameters
    ----------
    results : dict
        Dictionary containing the true and predicted position arrays.

        Expected structure:

            results[duration][speed]["x_pos"]
            results[duration][speed]["x_pred"]

        Arrays can have shape:
            (trials, time)
        or
            (trials, channels, time)

    speeds : list, optional
        Speeds to plot. If None, uses all speeds found in results.

    durations : list, optional
        Durations to plot. If None, uses all durations.

    subjects : array-like, optional
        Subject ID for each trial.

    max_trials : int, optional
        Maximum number of trials per condition. None = all trials.

    channel_idx : int
        Channel to use if arrays have shape (trials, channels, time).

    alpha : float
        Transparency of individual points.

    point_size : float
        Size of individual points.

    xlim, ylim : tuple, optional
        Axis limits.


    """

    # ------------------------------------------------------------
    # Determine durations
    # ------------------------------------------------------------
    if durations is None:
        durations = sorted(results.keys())

    # ------------------------------------------------------------
    # Determine speeds
    # ------------------------------------------------------------
    if speeds is None:
        n_speeds = results[durations[0]]["speed_rmse"].shape[1]
        speeds = np.arange(n_speeds) + 5

    speeds = list(speeds)

    # ------------------------------------------------------------
    # Split speeds into groups of 16
    # ------------------------------------------------------------
    speed_groups = [
        speeds[i:i + 16]
        for i in range(0, len(speeds), 16)
    ]

    # ------------------------------------------------------------
    # One figure per duration × speed group
    # ------------------------------------------------------------
    for duration in durations:

        for group_idx, speed_group in enumerate(speed_groups):

            n_speeds_group = len(speed_group)

            # ----------------------------------------------------
            # Create 4x4 figure
            # ----------------------------------------------------
            fig, axes = plt.subplots(
                4,
                4,
                figsize=(16, 16),
                squeeze=False
            )

            axes = axes.flatten()

            # ----------------------------------------------------
            # Plot each speed
            # ----------------------------------------------------
            for s, speed in enumerate(speed_group):

                ax = axes[s]

                # ------------------------------------------------
                # Select trials with this speed
                # ------------------------------------------------
                speed_mask = (
                    results[duration]["true_speed"] == speed
                )

                x_pos = results[duration]["true_pos"][speed_mask]
                x_pred = results[duration]["pred_pos"][speed_mask]

                # Handle (trials, channels, time)
                if x_pos.ndim == 3:
                    x_pos = x_pos[:, channel_idx, :]
                    x_pred = x_pred[:, channel_idx, :]

                n_trials = x_pos.shape[0]

                if max_trials is not None:
                    n_trials = min(n_trials, max_trials)

                x_pos = x_pos[:n_trials]
                x_pred = x_pred[:n_trials]

                # ------------------------------------------------
                # Colors: one color per trial
                # ------------------------------------------------
                trial_colors = plt.cm.turbo(
                    np.linspace(0, 1, n_trials)
                )

                all_true = []
                all_pred = []

                # ------------------------------------------------
                # Plot each trial
                # ------------------------------------------------
                for trial in range(n_trials):

                    true_values = x_pos[trial].reshape(-1)
                    pred_values = x_pred[trial].reshape(-1)

                    valid = (
                        np.isfinite(true_values)
                        & np.isfinite(pred_values)
                    )

                    true_values = true_values[valid]
                    pred_values = pred_values[valid]

                    if len(true_values) == 0:
                        continue

                    all_true.append(true_values)
                    all_pred.append(pred_values)

                    ax.scatter(
                        true_values,
                        pred_values,
                        s=point_size,
                        alpha=alpha,
                        color=trial_colors[trial],
                        rasterized=True
                    )

                # ------------------------------------------------
                # Overall statistics
                # ------------------------------------------------
                if len(all_true) > 0:

                    true_all = np.concatenate(all_true)
                    pred_all = np.concatenate(all_pred)

                    if xlim is not None:
                        line_min = xlim[0]
                        line_max = xlim[1]
                    else:
                        line_min = min(
                            np.min(true_all),
                            np.min(pred_all)
                        )
                        line_max = max(
                            np.max(true_all),
                            np.max(pred_all)
                        )

                    # Identity line
                    ax.plot(
                        [line_min, line_max],
                        [line_min, line_max],
                        "--",
                        color="black",
                        linewidth=1.5
                    )

                    # r
                    if len(true_all) > 1:

                        r = np.corrcoef(
                            true_all,
                            pred_all
                        )[0, 1]

                        rmse = np.sqrt(
                            np.mean(
                                (pred_all - true_all) ** 2
                            )
                        )

                        ax.text(
                            0.05,
                            0.95,
                            f"r = {r:.2f}\nRMSE = {rmse:.2f}",
                            transform=ax.transAxes,
                            va="top",
                            fontsize=10
                        )

                # ------------------------------------------------
                # Formatting
                # ------------------------------------------------
                ax.set_title(
                    f"{speed}°/s",
                    fontsize=14,
                    fontweight="bold"
                )

                ax.set_xlabel(
                    "True x position",
                    fontsize=10
                )

                ax.set_ylabel(
                    "Predicted x position",
                    fontsize=10
                )

                if xlim is not None:
                    ax.set_xlim(xlim)

                if ylim is not None:
                    ax.set_ylim(ylim)

                ax.set_aspect("equal", adjustable="box")

            # ----------------------------------------------------
            # Hide unused panels
            # ----------------------------------------------------
            for s in range(n_speeds_group, 16):
                axes[s].axis("off")

            # ----------------------------------------------------
            # Figure title
            # ----------------------------------------------------
            first_speed = speed_group[0]
            last_speed = speed_group[-1]

            fig.suptitle(
                f"Duration = {duration:.1f} s | "
                f"Speeds {first_speed}–{last_speed}°/s",
                fontsize=20,
                fontweight="bold"
            )

            plt.tight_layout(
                rect=[0, 0, 1, 0.96]
            )

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------
            if save_path is not None:

                os.makedirs(save_path, exist_ok=True)

                filename_group = (
                    f"{filename}_duration_{duration:.1f}"
                    f"_speeds_{first_speed}_{last_speed}.png"
                )

                plt.savefig(
                    op.join(save_path, filename_group),
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.show()


import os
import matplotlib.pyplot as plt
def plot_invalid_values_per_trial(
    training_data,
    speeds=None,
    durations=None,
    channel_idx=0,
    max_trials=None,
    percentage=False,
    save_path=None,
    filename="invalid_values_per_trial",
):
    """
    Plot the number of NaN/Inf values for each trial,
    organized by duration and stimulus speed.

    One figure is created for each duration, with up to 16 speed
    conditions arranged in a 4 x 4 grid.

    Parameters
    ----------
    training_data : dict
        Dictionary containing:

            results[duration]["true_pos"]
            results[duration]["pred_pos"]
            results[duration]["true_speed"]

        Arrays can have shape:
            (trials, time)
        or:
            (trials, channels, time)

    speeds : list, optional
        Speeds to plot. If None, uses 5, 6, ... based on speed_rmse.

    durations : list, optional
        Durations to plot. If None, uses all durations.

    channel_idx : int
        Channel to use if arrays have shape (trials, channels, time).

    max_trials : int, optional
        Maximum number of trials to display per speed.

    percentage : bool
        If True, plot percentage of invalid values rather than
        number of invalid values.

    save_path : str, optional
        Directory in which to save figures.

    filename : str
        Base filename for saved figures.
    """

    # ------------------------------------------------------------
    # Determine durations
    # ------------------------------------------------------------
    if durations is None:
        durations = sorted(training_data.keys())

    # ------------------------------------------------------------
    # Determine speeds
    # ------------------------------------------------------------
    if speeds is None:
        n_speeds = len(np.unique(training_data[durations[0]]["training_speed"]))
        speeds = np.arange(n_speeds) + 5

    speeds = list(speeds)

    # ------------------------------------------------------------
    # Split speeds into groups of 16
    # ------------------------------------------------------------
    speed_groups = [
        speeds[i:i + 16]
        for i in range(0, len(speeds), 16)
    ]

    # ------------------------------------------------------------
    # Loop over durations
    # ------------------------------------------------------------
    for duration in durations:

        for group_idx, speed_group in enumerate(speed_groups):

            fig, axes = plt.subplots(
                4,
                4,
                figsize=(16, 12),
                squeeze=False
            )

            axes = axes.flatten()

            # ----------------------------------------------------
            # Loop over speeds
            # ----------------------------------------------------
            for s, speed in enumerate(speed_group):

                ax = axes[s]

                # ------------------------------------------------
                # Select trials for this speed
                # ------------------------------------------------
                speed_mask = (
                    training_data[duration]["training_speed"] == speed
                )

                x_pos = training_data[duration]["training_x_pos"][speed_mask]
                # x_pred = results[duration]["pred_pos"][speed_mask]

                # ------------------------------------------------
                # Select channel if necessary
                # ------------------------------------------------
                if x_pos.ndim == 3:
                    x_pos = x_pos[:, channel_idx, :]
                    # x_pred = x_pred[:, channel_idx, :]

                n_trials = x_pos.shape[0]

                if max_trials is not None:
                    n_trials = min(n_trials, max_trials)

                x_pos = x_pos[:n_trials]
                # x_pred = x_pred[:n_trials]

                # ------------------------------------------------
                # Count invalid values per trial
                # ------------------------------------------------
                invalid_true = np.sum(
                    ~np.isfinite(x_pos),
                    axis=tuple(range(1, x_pos.ndim))
                )

                # invalid_pred = np.sum(
                #     ~np.isfinite(x_pred),
                #     axis=tuple(range(1, x_pred.ndim))
                # )

                # ------------------------------------------------
                # Combine true + predicted invalid values
                # ------------------------------------------------
                # invalid_total = (
                #     invalid_true + invalid_pred
                # )

                # ------------------------------------------------
                # Convert to percentage if requested
                # ------------------------------------------------
                if percentage:

                    n_values = np.prod(
                        x_pos.shape[1:]
                    )

                    invalid_total = (
                        invalid_true
                        / (2 * n_values)
                        * 100
                    )

                    ylabel = "Invalid values (%)"

                else:

                    ylabel = "NaN / Inf values"

                # ------------------------------------------------
                # Plot each trial
                # ------------------------------------------------
                trial_numbers = np.arange(1, n_trials + 1)

                ax.bar(
                    trial_numbers,
                    invalid_true
                )

                # ------------------------------------------------
                # Formatting
                # ------------------------------------------------
                ax.set_title(
                    f"{speed}°/s",
                    fontsize=14,
                    fontweight="bold"
                )

                ax.set_xlabel(
                    "Trial",
                    fontsize=10
                )

                ax.set_ylabel(
                    ylabel,
                    fontsize=10
                )

                ax.set_xticks(trial_numbers)

                # Avoid overcrowding if many trials
                if n_trials > 20:
                    ax.set_xticks(
                        trial_numbers[::5]
                    )

                ax.grid(
                    axis="y",
                    alpha=0.3
                )

            # ----------------------------------------------------
            # Hide unused panels
            # ----------------------------------------------------
            for s in range(
                len(speed_group),
                16
            ):
                axes[s].axis("off")

            # ----------------------------------------------------
            # Figure title
            # ----------------------------------------------------
            first_speed = speed_group[0]
            last_speed = speed_group[-1]

            fig.suptitle(
                f"Invalid values per trial — "
                f"Duration = {duration:.1f} s\n"
                f"Speeds {first_speed}–{last_speed}°/s",
                fontsize=20,
                fontweight="bold"
            )

            plt.tight_layout(
                rect=[0, 0, 1, 0.94]
            )

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------
            if save_path is not None:

                os.makedirs(
                    save_path,
                    exist_ok=True
                )

                filename_group = (
                    f"{filename}_duration_{duration:.1f}"
                    f"_speeds_{first_speed}_{last_speed}.png"
                )

                plt.savefig(
                    op.join(
                        save_path,
                        filename_group
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )

            plt.show()