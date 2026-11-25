
import numpy as np
from torch.utils.data import DataLoader
import torch
import scipy.io as sio
import torch
from torch.utils.data import Dataset
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

            # # ----------------------------------------
            # # EyeLink position traces
            # # ----------------------------------------
            # eyelink_trial = eyelink_x[
            #     sampleStart:sampleEnd, t, c, :
            # ]

            # time = np.arange(sampleStart, sampleEnd) / fs

            # plt.figure(figsize=(10, 4))

            # for eye in range(eyelink_trial.shape[-1]):
            #     plt.plot(
            #         time,
            #         eyelink_trial[:, eye],
            #         label=f"Eye {eye + 1}"
            #     )

            # plt.xlabel("Time (s)")
            # plt.ylabel("EyeLink horizontal position (deg)")
            # plt.title(f"Condition {c}, Trial {t}, Speed {t_speed}")
            # plt.legend()
            # plt.grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.show()

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

#-----------------------------
# 1: Generate Pytorch Dataset
#-----------------------------
class EOGDataset(Dataset):
    def __init__(self, X, x_pos_norm, x_pos, x_speed):
        # X = X_train
        # x_pos = x_pos_train

        self.X = torch.tensor(X, dtype=torch.float32)
        self.x_pos_norm = torch.tensor(x_pos_norm, dtype=torch.float32)
        self.x_pos = torch.tensor(x_pos, dtype=torch.float32)
        self.x_speed = torch.tensor(x_speed, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.x_pos_norm[idx], self.x_pos[idx], self.x_speed[idx]


#----------------------    
# 2: Model design
#----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class EOGNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(n_channels, 16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8) #kernel_size=17, padding=8

        # 🔥 correct head
        self.head = nn.Conv1d(in_channels=32, out_channels=2, kernel_size=1)

    def forward(self, x):
        # x: (B, channels, T)
        # X = torch.tensor(X, dtype=torch.float32)
        # x = X
        # conv1 = nn.Conv1d(6, 16, kernel_size=9, padding=4)
        # conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        # Split
        x_eog = x[:, :2, :]
        blink_eog   = x_eog * x[:, 2:4, :]
        sacc_eog    = x_eog * x[:, 4:6, :]
        pursuit_eog = x_eog * x[:, 6:8, :]

        # x = torch.cat([blink_eog, sacc_eog, pursuit_eog], dim=1)  # (B, 6, T)
        x = torch.cat([blink_eog, sacc_eog, pursuit_eog], dim=1)  # (B, 6, T)
        
        # CNN
        x = F.relu(self.conv1(x))   # (B, 16, T) x shape: B = 32, 16, T = 250
        x = F.relu(self.conv2(x))   # (B, 32, T) x shape: B = 32, 32, T = 250 

        # 🔥 NO flatten
        x = self.head(x)            # (B, 2, T)

        return x.squeeze(1)         # (B, T)
#-----------------------------------------------
# Interpolate data when there are blinks or eye position is too far away from the screen
#-----------------------------------------------
#Usage:
        # X_train_eog = ch.interpolate_gaps(X_train[:, :2, :],train_mask)
        # x_pos_train = ch.interpolate_gaps(x_pos_train, train_mask)
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


def crop_sequences(X, x_pos, n_samples):
  total_samples = X.shape[2]
  if not (0 < n_samples <= total_samples):
    raise ValueError(
        f"n_samples ({n_samples}) must be between 1 and {total_samples}."
    )

  start = int(np.floor(total_samples / 2 - n_samples / 2))
  start = max(0, min(start, total_samples - n_samples))
  end = start + n_samples

  return X[:, :, start:end], x_pos[:, :, start:end]


#--------------------------------------------
# Model Training
#--------------------------------------------

from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(dataset):
    #dataset = train_dataset 
    loader = DataLoader(dataset, batch_size=32, shuffle=True) #batch_size = number of trials per batch

    model = EOGNet(n_channels=6)# #6 channels here = [blink data, saccade data, pursuit]x[2 eyes]dataset.X.shape[1]-6)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "train_acc": [],
        "train_loss": []
    }

    # loss_pos_fn = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0
        for X, x_pos_norm, x_pos, x_speed in loader:
            optimizer.zero_grad() # reset gradients at new batch (clear old gradients)
            
            pred_pos = model(X)

            # x_evt = X[:, 2:, :]
            blink_mask = X[:, 2:4, :] > 0
            farpos_mask = (x_pos < -30) | (x_pos > 30)
            valid_mask = ~blink_mask & ~farpos_mask

            loss = ((pred_pos - x_pos_norm)**2)[valid_mask].mean() #MSE loss
            # loss = loss_pos_fn(pred_pos, x_pos)

            loss.backward() #compute new gradients
            optimizer.step() #update weights

            total_loss += loss.item()


        history["train_loss"].append(total_loss)

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    return model, history

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


def get_pos_performance_per_speed(pred_pos, x_pos, x_speed, valid, scale=1.0):
    """valid: boolean (N, 2, T), True = usable sample.
    scale: xpos_std of the training fold, to return RMSE in degrees."""
    speed_list = np.unique(x_speed)
    rmse = np.full(len(speed_list), np.nan)
    for i, speed in enumerate(speed_list):
        idx = (x_speed[:, None, None] == speed) & valid
        if idx.any():
            rmse[i] = np.sqrt(np.mean((pred_pos[idx] - x_pos[idx]) ** 2)) * scale
    return rmse

# def get_pos_performance_per_speed(pred_pos, x_pos, x_speed):
#     pred_pos = pred_pos.squeeze()

#     speed_list = np.unique(x_speed)
#     rmse_per_speed = np.full(len(speed_list), np.nan)

#     for i, speed in enumerate(speed_list):

#         idx = (
#         (x_speed[:, None, None] == speed) &
#         (~np.isnan(pred_pos)) &
#         (~np.isnan(x_pos))
#         )

#         if np.any(idx):
#             rmse_per_speed[i] = np.sqrt(
#                 np.mean((pred_pos[idx] - x_pos[idx])**2)
#             )

#     return rmse_per_speed

def per_speed_summary(summary_data, xpos_std):
    pred, true = summary_data["pred_pos"], summary_data["true_pos"]            # normalized
    spd = np.asarray(summary_data["true_speed"]).reshape(len(pred), -1)[:, 0]
    out = {}
    for s in np.unique(spd):
        m = spd == s
        p, t = pred[m].ravel(), true[m].ravel()
        v = np.isfinite(p) & np.isfinite(t)
        p, t = p[v], t[v]
        mse = np.mean((p - t) ** 2)
        out[s] = dict(
            rmse_deg = np.sqrt(mse) * xpos_std,   # use the training-fold mean std
            target_sd = t.std(),                  # spread of the target
            r2 = 1 - mse / t.var(),               # relative to a constant predictor
            slope = np.polyfit(t, p, 1)[0],       # <1 means shrinkage
            r = np.corrcoef(t, p)[0, 1],
        )
    return out

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

# ==================
#Plot various summary stats
# ==================
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_summary_metric(summary_df, metric, speeds=range(5, 21),
                        unit="iteration", color="lightgray", dot_color="black",
                        save_path=None, filename=None, ylim=None, seed=0):
    """
    Bars = mean, error bars = SEM, dots = individual units.

    unit="iteration":   dots are CV iterations (each averaged over test participants).
    unit="participant": dots are test participants (each averaged over CV iterations).
    SEM is computed across the dots.
    """
    METRIC_INFO = {
    "rmse_deg":  dict(label="RMSE (deg)",            ref=None),
    "r2":        dict(label="R² (vs constant)",      ref=0),
    "target_sd": dict(label="Target SD (normalized)", ref=None),
    "slope":     dict(label="Slope (pred vs true)",  ref=1),
    "r":         dict(label="Pearson r",             ref=None),
    }
    info = METRIC_INFO.get(metric, dict(label=metric, ref=None))
    speeds = list(speeds)
    rng = np.random.default_rng(seed)

    group_cols = ["duration", "speed", unit]
    df = (summary_df[summary_df["speed"].isin(speeds)]
          .groupby(group_cols, as_index=False)[metric]
          .mean())

    durations = sorted(df["duration"].unique())
    fig, axes = plt.subplots(len(durations), 1, sharey=True, squeeze=False,
                             figsize=(14, 4 * len(durations)))
    axes = axes.flatten()

    for ax, dur in zip(axes, durations):
        d = df[df["duration"] == dur]

        for i, sp in enumerate(speeds):
            vals = d.loc[d["speed"] == sp, metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0

            ax.bar(i, mean, width=0.7, color=color, edgecolor="black",
                   linewidth=1, alpha=0.8, zorder=1)
            ax.errorbar(i, mean, yerr=sem, color="black", capsize=3,
                        linewidth=1.5, capthick=1.5, zorder=3)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals,
                       s=22, color=dot_color, edgecolor="white",
                       linewidth=0.4, alpha=0.85, zorder=4)

        if info["ref"] is not None:
            ax.axhline(info["ref"], color="gray", linestyle="--", linewidth=1)

        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels(speeds)
        ax.set_xlim(-0.7, len(speeds) - 0.3)
        ax.set_title(f"Duration = {dur:.1f} s", fontsize=16, fontweight="bold")
        ax.set_xlabel("Target speed (deg/s)", fontsize=13)
        ax.set_ylabel(info["label"], fontsize=13)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if ylim is not None:
            ax.set_ylim(ylim)

    fig.suptitle(f"{info['label']} across speeds and durations "
                 f"(mean ± SEM, dots = {unit}s)",
                 fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(op.join(save_path, filename or f"summary_{metric}_{unit}.png"),
                    dpi=300, bbox_inches="tight")
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
#===========================================================
# Check amount of data deviating too far away from the screen.
#===========================================================
import os
import matplotlib.pyplot as plt

def plot_invalid_values( training_data, speeds=None, durations=None, channel_idx=0, save_path=None, filename="invalid_values_by_speed_duration.png", ):
    """ Plot the percentage of values outside +-30 deg range for each speed,
      with one subplot per duration. 
    Each bar represents the percentage of 
    invalid values across all trials for a given speed. 
    Parameters 
    ---------- 
    training_data : dict Dictionary containing: 
    training_data[duration]["training_x_pos"] 
    training_data[duration]["training_speed"] 
    Arrays can have shape: (trials, time) 
    or: (trials, channels, time) 
    speeds : list, optional Speeds to plot. 
    If None, uses all unique speeds. 
    durations : list, optional Durations to plot. 
    If None, uses all durations. 
    channel_idx : int Channel to use if arrays have shape (trials, channels, time). 
    save_path : str, optional Directory in which to save the figure. 
    filename : str Name of the saved figure. """ 
    # ------------------------------------------------------------ # Determine durations # ------------------------------------------------------------ 
    if durations is None: 
        durations = sorted(training_data.keys()) 
    # ------------------------------------------------------------ # Determine speeds # ------------------------------------------------------------
    if speeds is None: 
        speeds = np.unique( training_data[durations[0]]["training_speed"] ) 
    speeds = list(speeds) 
    # ------------------------------------------------------------ # Create one subplot per duration # ------------------------------------------------------------ 

    n_durations = len(durations)

    fig, axes = plt.subplots(
        n_durations,
        1,
        figsize=(14, 4 * n_durations),
        squeeze=False
    )

    axes = axes.flatten()

    # ------------------------------------------------------------
    # Loop over durations
    # ------------------------------------------------------------
    for ax, duration in zip(axes, durations):

        means = []
        sems = []

        # --------------------------------------------------------
        # Loop over speeds
        # --------------------------------------------------------
        for speed in speeds:

            speed_mask = (training_data[duration]["training_speed"] == speed)

            x_pos = (training_data[duration]["training_x_pos"][speed_mask])

            # ----------------------------------------------------
            # Select channel if necessary
            # ----------------------------------------------------
            if x_pos.ndim == 3:
                x_pos = x_pos[:, channel_idx, :]

            # ----------------------------------------------------
            # Calculate invalid percentage PER TRIAL
            # ----------------------------------------------------
            invalid_count = np.sum((x_pos < -30) | (x_pos > 30), axis=tuple(range(1, x_pos.ndim)))

            total_values_per_trial = x_pos.shape[1]

            invalid_percentage_per_trial = (invalid_count / total_values_per_trial * 100)

            # ----------------------------------------------------
            # Mean ± SEM across trials
            # ----------------------------------------------------
            mean_percentage = np.mean(invalid_percentage_per_trial)

            sem_percentage = (np.std(invalid_percentage_per_trial,ddof=1)
                / np.sqrt(len(invalid_percentage_per_trial)))

            means.append(mean_percentage)
            sems.append(sem_percentage)

        # --------------------------------------------------------
        # Plot one bar per speed
        # --------------------------------------------------------
        bars = ax.bar(
            speeds,
            means,
            yerr=sems,
            capsize=5
        )

        # --------------------------------------------------------
        # Formatting
        # --------------------------------------------------------
        ax.set_title(
            f"Duration = {duration:.1f} s",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Speed (°/s)",
            fontsize=13
        )

        ax.set_ylabel(
            "Values outside ±30 (%)",
            fontsize=13
        )

        ax.set_xticks(speeds)

        ax.grid(
            axis="y",
            alpha=0.3
        )

        # --------------------------------------------------------
        # Add mean percentage above each bar
        # --------------------------------------------------------
        for bar, mean_percentage, sem_percentage in zip(
            bars,
            means,
            sems
        ):

            if mean_percentage > 0:

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sem_percentage,
                    f"{mean_percentage:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )


    # ------------------------------------------------------------
    # Figure title — IMPORTANT: outside the duration loop
    # ------------------------------------------------------------
    fig.suptitle(
        "Values outside ±30 across stimulus speeds and durations",
        fontsize=20,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    if save_path is not None:

        os.makedirs(
            save_path,
            exist_ok=True
        )

        plt.savefig(
            op.join(
                save_path,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt


def plot_invalid_values_subj(all_x_pos, all_x_speed, all_subjects, sequence_lengths,
                        speeds=None, channel_idx=0, limit=30,
                        save_path=None,
                        filename="invalid_values_by_speed_duration_participants.png"):
    """
    Percentage of samples outside +-limit (or non-finite) per speed, one subplot
    per duration. Bars = mean across participants (error = SEM across
    participants); dots = each participant's mean over their trials.

    all_x_pos        : (trials, time) or (trials, channels, time)
    all_x_speed      : (trials,) speed label per trial
    all_subjects     : (trials,) participant id per trial
    sequence_lengths : {duration_s: n_samples}
    """
    all_x_pos = np.asarray(all_x_pos)
    speed_arr = np.asarray(all_x_speed)
    if speed_arr.ndim > 1:                 # per-sample speed -> take the first sample
        speed_arr = speed_arr.reshape(len(speed_arr), -1)[:, 0]
    subjects = np.asarray(all_subjects)

    if all_x_pos.ndim == 3:
        all_x_pos = all_x_pos[:, channel_idx, :]

    participants_u = list(dict.fromkeys(subjects))      # keeps original order
    if speeds is None:
        speeds = np.unique(speed_arr)
    speeds = list(speeds)

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(participants_u), 2)))
    rng = np.random.default_rng(0)

    durations = sorted(sequence_lengths.keys())
    fig, axes = plt.subplots(len(durations), 1,
                             figsize=(14, 4 * len(durations)), squeeze=False)
    axes = axes.flatten()
    fs = 125
    for ax, duration in zip(axes, durations):
        n = sequence_lengths[duration]
        # x_crop = all_x_pos[:, :n]
        total_samples = all_x_pos.shape[1]
        n_samples = int(duration*fs)
        start = int(np.floor(total_samples / 2 - n_samples / 2))
        start = max(0, min(start, total_samples - n_samples))
        end = start + n_samples
        x_crop = all_x_pos[:, start:end]
        #invalid = ~np.isfinite(x_crop) | (np.abs(x_crop) > limit)   # NaN/inf count too
        invalid = (np.abs(x_crop) > limit) 

        pct_trial = invalid.mean(axis=1) * 100                      # (trials,)

        # participant x speed table of mean % invalid
        table = np.full((len(participants_u), len(speeds)), np.nan)
        for p, part in enumerate(participants_u):
            for s, speed in enumerate(speeds):
                m = (subjects == part) & (speed_arr == speed)
                if m.any():
                    table[p, s] = pct_trial[m].mean()

        means = np.nanmean(table, axis=0)
        n_valid = np.sum(~np.isnan(table), axis=0)
        sems = np.nanstd(table, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))

        width = 0.8 * np.min(np.diff(speeds)) if len(speeds) > 1 else 0.8
        bars = ax.bar(speeds, means, width=width, yerr=sems, capsize=5,
                      color="lightgray", edgecolor="black", alpha=0.8, zorder=1)

        # one dot per participant
        for p, part in enumerate(participants_u):
            xs = np.array(speeds) + rng.uniform(-width * 0.25, width * 0.25, len(speeds))
            ax.scatter(xs, table[p], s=40, color=colors[p], edgecolor="black",
                       linewidth=0.6, zorder=3, label=str(part))

        for bar, m, e in zip(bars, means, sems):
            if np.isfinite(m) and m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (e if np.isfinite(e) else 0),
                        f"{m:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Duration = {duration:.1f} s", fontsize=16, fontweight="bold")
        ax.set_xlabel("Speed (°/s)", fontsize=13)
        ax.set_ylabel(f"Values outside ±{limit} (%)", fontsize=13)
        ax.set_xticks(speeds)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Participant", loc="center right",
               frameon=False)

    fig.suptitle("Values outside ±%d across stimulus speeds and durations" % limit,
                 fontsize=20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.93, 0.97])

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(op.join(save_path, filename), dpi=300, bbox_inches="tight")
    plt.show()