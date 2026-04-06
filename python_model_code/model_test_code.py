import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as op
import sys
import scipy.io as sio

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
import utils.classifier_helpers as ch

data_dir = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'data', 'v05')

participants = {'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'}
all_X = []
all_mask = []
all_y_dir = []
all_y_speed = []
all_subjects = []
for participant in participants:
    filepath = op.join(data_dir, f"{participant}_v05", f"{participant}_model_data_filt_audio_visual_motion_v05.mat")
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

    all_X.append(X)
    all_mask.append(x_mask)
    all_y_dir.append(y_dir)
    all_y_speed.append(y_speed)

    # 👇 critical for LOSO
    all_subjects.append(np.array([participant] * len(X)))

all_X = np.concatenate(all_X, axis=0)
all_mask = np.concatenate(all_mask, axis=0)
all_y_dir = np.concatenate(all_y_dir, axis=0)
all_y_speed = np.concatenate(all_y_speed, axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)


class EOGDataset(Dataset):
    def __init__(self, X, x_mask, y_dir, y_speed):
        # X = X_train
        # x_mask = mask_train
        # y_dir = y_dir_train
        # y_speed = y_speed_train

        self.X = torch.tensor(X, dtype=torch.float32)
        self.x_mask = torch.tensor(x_mask, dtype=torch.float32)
        self.y_dir = torch.tensor(y_dir, dtype=torch.long)
        self.y_speed = torch.tensor(y_speed, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # idx =0
        # X = X[idx]
        # mask = x_mask[idx]
        X = self.X[idx]
        mask = self.x_mask[idx]

        # 🔥 apply mask
        X = X * mask

        # assuming channel order:
        # [EOG (2 eyes), blink (2), saccade (2), pursuit (2)]
        eog = X[:2]
        eog_mask = mask[:2]

        # masked mean/std for EOG only
        mean = (eog.sum(dim=1, keepdim=True) /
                (eog_mask.sum(dim=1, keepdim=True) + 1e-8))

        var = ((eog - mean)**2 * eog_mask).sum(dim=1, keepdim=True) / \
            (eog_mask.sum(dim=1, keepdim=True) + 1e-8)

        std = torch.sqrt(var + 1e-8)

        eog = (eog - mean) / std

        # put back
        X[:2] = eog

        return X, mask, self.y_dir[idx], self.y_speed[idx]

##=========================================================================

from torch.utils.data import DataLoader

test_participant = sorted(participants)[0]
train_idx = all_subjects != test_participant
X_train = all_X[train_idx]
mask_train = all_mask[train_idx]
y_dir_train = all_y_dir[train_idx]
y_speed_train = all_y_speed[train_idx]
train_dataset = EOGDataset(X_train, mask_train, y_dir_train, y_speed_train)
dataset = train_dataset 

import torch.nn as nn
import torch.nn.functional as F


# class EOGNet(nn.Module):
#     def __init__(self, n_channels):
#         super().__init__()
n_channels=dataset.X.shape[1]
conv1 = nn.Conv1d(n_channels, 32, kernel_size=101, padding=50)
conv2 = nn.Conv1d(32, 64, kernel_size=51, padding=25)
pool = nn.MaxPool1d(2)

lstm = nn.LSTM(64, 64, batch_first=True)

# 🔥 NEW: combine temporal features
fc = nn.Linear(64 * 3, 64)

fc_dir = nn.Linear(64, 2)
fc_speed = nn.Linear(64, 1)

# def forward(self, x, x_mask):
    # x_mask = mask_train
loader = DataLoader(dataset, batch_size=16, shuffle=True) #batch_size = number of trials per batch
for x, x_mask, y_dir, y_speed in loader:
    x_mask = x_mask.any(dim=1, keepdim=True).float()
    
    # x: (batch, channels, time)
    # CNN
    x = F.relu(conv1(x))
    x = pool(x)
    x_mask = pool(x_mask)

    x = F.relu(conv2(x))
    x = pool(x)
    x_mask = pool(x_mask)

    # (B, T, F)
    x = x.permute(0, 2, 1)
    x, _ = lstm(x)

    # -------------------------------
    # 🔥 KEY: temporal feature extraction
    # -------------------------------

    mask = x_mask.squeeze(1)  # (B, T)

    # first valid timestep (since mask is binary, the first 1 encountered)
    first_idx = mask.argmax(dim=1) 

    # last valid timestep
    last_idx = mask.shape[1] - 1 - torch.flip(mask, dims=[1]).argmax(dim=1)

    B = x.shape[0]

    first_feat = x[torch.arange(B), first_idx]
    last_feat  = x[torch.arange(B), last_idx]

    # 🔥 displacement (direction signal)
    delta_feat = last_feat - first_feat

    # 🔥 average (global context)
    mask = mask.unsqueeze(-1)
    avg_feat = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    # 🔥 combine all
    feat = torch.cat([avg_feat, delta_feat, last_feat], dim=1)

    feat = F.relu(fc(feat))

    dir_out = fc_dir(feat)
    speed_out = fc_speed(feat)

    return dir_out, speed_out