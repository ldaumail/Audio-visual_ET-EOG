
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

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][:,:,8:10,:]
    blink = data['data'][0, 0]['sub01'][0, 0]['blinkTrs'][:,:,4:6,:]
    saccade = data['data'][0, 0]['sub01'][0, 0]['saccadesTrs'][:,:,4:6,:]
    pursuit = data['data'][0, 0]['sub01'][0, 0]['pursuitTrs'][:,:,4:6,:]

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

# def get_speed_performance_per_condition(pred_speed, y_speed):
#     pred_speed = pred_speed.squeeze()

#     unique_speeds = np.unique(y_speed)
#     rmse_per_speed = np.nan(len(unique_speeds))

#     for s, speed in enumerate(unique_speeds):
#         idx = y_speed == speed

#         if np.sum(idx) == 0:
#             continue

#         rmse = np.sqrt(np.mean((pred_speed[idx] - y_speed[idx])**2))
#         rmse_per_speed[s] = rmse

#     return rmse_per_speed

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