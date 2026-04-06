
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
#-----------------------------------------------
# 1: Load data and generate the Pytorch dataset
#-----------------------------------------------
def load_mat_data(filepath):

    data = sio.loadmat(filepath)

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][::4,:,8:10,:] #Downsample 4 times with ::4
    blink = data['data'][0, 0]['sub01'][0, 0]['blinkTrs'][::4,:,4:6,:]
    saccade = data['data'][0, 0]['sub01'][0, 0]['saccadesTrs'][::4,:,4:6,:]
    pursuit = data['data'][0, 0]['sub01'][0, 0]['pursuitTrs'][::4,:,4:6,:]

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
    y_dir = []
    y_speed = []

    for c in range(n_cond):
        for t in range(n_trials):

            # EOG → (channels, time)
            eog_trial = eog[:, t, c, :].T

            blink_trial = blink[:, t, c, :].T
            sacc_trial = saccade[:, t, c, :].T
            pursuit_trial = pursuit[:, t, c, :].T

            # Concatenate channels
            trial = np.concatenate([
                eog_trial,
                blink_trial,
                sacc_trial,
                pursuit_trial
            ], axis=0)  # (channels_total (n_variables*n_eyes), time)

            X_list.append(trial)

            y_dir.append(dir_labels[c])
            y_speed.append(speed[t, c])

    X = np.array(X_list)          # (N_trials, channels, time)
    x_mask = ~np.isnan(X)   # True where data exists
    X = np.nan_to_num(X, nan=0.0)
    y_dir = np.array(y_dir)
    y_speed = np.array(y_speed)

    return X, x_mask, y_dir, y_speed

def get_network_output(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    
    all_dir = []
    all_speed = []
    all_targets_dir = []
    all_targets_speed = []

    with torch.no_grad():
        for X, x_mask, y_dir, y_speed in loader:
            X = X * x_mask
            pred_dir, pred_speed = model(X, x_mask)
            # pred_dir, pred_speed = model(X)

            all_dir.append(pred_dir.cpu())
            all_speed.append(pred_speed.cpu())
            all_targets_dir.append(y_dir.cpu())
            all_targets_speed.append(y_speed.cpu())

    pred_dir = torch.cat(all_dir).numpy()
    pred_speed = torch.cat(all_speed).numpy()
    y_dir = torch.cat(all_targets_dir).numpy()
    y_speed = torch.cat(all_targets_speed).numpy()

    return pred_dir, pred_speed, y_dir, y_speed

def get_direction_performance(pred_dir, y_dir):
    pred_labels = np.argmax(pred_dir, axis=1)
    acc = np.mean(pred_labels == y_dir)
    return acc

def get_speed_performance(pred_speed, y_speed):
    rmse = np.sqrt(np.mean((pred_speed.squeeze() - y_speed)**2))
    return rmse


def get_speed_performance_per_condition(pred_speed, y_speed):
    pred_speed = pred_speed.squeeze()
    speed_list= np.unique(y_speed)
    rmse_per_speed = np.full(len(speed_list), np.nan)

    for i, speed in enumerate(speed_list):
        idx = y_speed == speed

        if np.sum(idx) > 0:
            rmse_per_speed[i] = np.sqrt(
                np.mean((pred_speed[idx] - y_speed[idx])**2)
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


def plot_eog_traces(
    X,
    y_dir,
    y_speed,
    subjects,
    save_path,
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
            ax.set_ylim(-4, 4)

    # cleaner legend (only once)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Subjects")

    plt.tight_layout()
    saveDir = op.join(save_path)
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(saveDir, "eog_subject_means.png"), dpi=300, bbox_inches='tight')
    plt.show()
