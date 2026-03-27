import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as op
import sys

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
# from utils.classifier_helpers import get_network_output, get_network_cat_performance, get_direction_performance, get_speed_performance, get_speed_performance_per_condition
import utils.classifier_helpers as ch


#-----------------------------
# 1: Generate Pythorch Dataset
#-----------------------------
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


    # def __getitem__(self, idx):
    #     return self.X[idx],  self.x_mask[idx], self.y_dir[idx], self.y_speed[idx]
    
#----------------------    
# 2: Model design
#----------------------
import torch.nn as nn
import torch.nn.functional as F


class EOGNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=101, padding=50)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=51, padding=25)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(64, 64, batch_first=True)

        # 🔥 NEW: combine temporal features
        self.fc = nn.Linear(64 * 3, 64)

        self.fc_dir = nn.Linear(64, 2)
        self.fc_speed = nn.Linear(64, 1)

    def forward(self, x, x_mask):

        x_mask = x_mask.any(dim=1, keepdim=True).float()
        
        # x: (batch, channels, time)
        # CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x_mask = self.pool(x_mask)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x_mask = self.pool(x_mask)

        # (B, T, F)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

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

        feat = F.relu(self.fc(feat))

        dir_out = self.fc_dir(feat)
        speed_out = self.fc_speed(feat)

        return dir_out, speed_out
#--------------------------------------------
# Model Training
#--------------------------------------------

from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(dataset):
    #dataset = train_dataset 
    loader = DataLoader(dataset, batch_size=16, shuffle=True) #batch_size = number of trials per batch

    model = EOGNet(n_channels=dataset.X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "train_acc": [],
        "train_loss": []
    }

    loss_dir_fn = nn.CrossEntropyLoss()
    loss_speed_fn = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0
        all_preds = []
        all_targets = []
        for X, x_mask, y_dir, y_speed in loader:
            optimizer.zero_grad() # reset gradients at new batch (clear old gradients)
            
            X = X * x_mask
            pred_dir, pred_speed = model(X, x_mask)
            # y_dir.dtype == torch.long
            # print("pred_dir:", pred_dir.shape)
            # print("y_dir:", y_dir.shape)
            # print("y_dir sample:", y_dir[:5])
            loss_dir = loss_dir_fn(pred_dir, y_dir)
            loss_speed = loss_speed_fn(pred_speed.view(-1), y_speed)

            loss = loss_dir + 0.5 * loss_speed
            loss.backward() #compute new gradients
            optimizer.step() #update weights

            total_loss += loss.item()

            all_preds.append(pred_dir.detach())
            all_targets.append(y_dir)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_targets = torch.cat(all_targets).cpu().numpy()

        epoch_acc = ch.get_direction_performance(all_preds, all_targets)

        history["train_acc"].append(epoch_acc)
        history["train_loss"].append(total_loss)

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    return model, history

#-------------------------------
#Prepare all participants data
#-------------------------------
data_dir = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'data', 'v05')

participants = {'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'}
all_X = []
all_mask = []
all_y_dir = []
all_y_speed = []
all_subjects = []
for participant in participants:
    filepath = op.join(data_dir, f"{participant}_v05", f"{participant}_model_data_filt_audio_visual_motion_v05.mat")
    X, x_mask, y_dir, y_speed = ch.load_mat_data(filepath)

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

#

crossvalit_alldir_perfs = np.zeros([6,5]) #6 folds since 6 participants, repeat 5 times
crossvalit_allspeed_perfs = np.zeros([6,5])
crossvalit_dir_perfs = np.zeros([6,2,5]) #direction perf
crossvalit_speed_perfs = np.zeros([6,16,5]) #speed perf
#preallocate mean performance and mean category performances of each crossvalidation iteration 
# crossvalit_mean_alldir_perfs = np.zeros([5,])
# crossvalit_mean_allspeed_perfs = np.zeros([5,])
# crossvalit_mean_dir_perfs = np.zeros([5,2])
# crossvalit_mean_speed_perfs = np.zeros([5,16])
#preallocate standard deviation of performance and standard deviations of category performances of each crossvalidation iteration 
# crossvalit_std_alldir_perfs = np.zeros([5,])
# crossvalit_std_allspeed_perfs = np.zeros([5,])
# crossvalit_std_dir_perfs = np.zeros([5,2])
# crossvalit_std_speed_perfs = np.zeros([5,16])
#preallocate the training accuracy for each fold in every crossvalidation
alldir_training_accuracy = np.zeros([6,5])
allspeed_training_accuracy = np.zeros([6,5])

training_loss = np.zeros([6,5])

run_nb = 0

for it in range(5):
    for t, test_participant in enumerate(participants):

        print(f"\n=== Testing on {test_participant} ===")

        train_idx = all_subjects != test_participant
        test_idx  = all_subjects == test_participant

        X_train = all_X[train_idx]
        mask_train = all_mask[train_idx]
        y_dir_train = all_y_dir[train_idx]
        y_speed_train = all_y_speed[train_idx]

        X_test = all_X[test_idx]
        mask_test = all_mask[test_idx]
        y_dir_test = all_y_dir[test_idx]
        y_speed_test = all_y_speed[test_idx]

        train_dataset = EOGDataset(X_train, mask_train, y_dir_train, y_speed_train)
        test_dataset  = EOGDataset(X_test,  mask_test,  y_dir_test,  y_speed_test)

        model, history = train_model(train_dataset)

        # 🔹 predictions
        pred_dir, pred_speed, y_dir, y_speed = ch.get_network_output(model, test_dataset)

        # 🔹 overall performance
        perf_dir = ch.get_direction_performance(pred_dir, y_dir)
        perf_speed = ch.get_speed_performance(pred_speed, y_speed)

        # 🔹 category performance
        perf_cat_dir = ch.get_network_cat_performance(pred_dir, y_dir, n_classes=2)
        perf_cat_speed = ch.get_speed_performance_per_condition(pred_speed, y_speed)
        # 🔹 store
        crossvalit_alldir_perfs[t, it] = perf_dir
        crossvalit_allspeed_perfs[t, it] = perf_speed #total rmse
        crossvalit_dir_perfs[t, :, it] = perf_cat_dir
        crossvalit_speed_perfs[t, :, it] = perf_cat_speed #rmse per speed

        # 🔹 training accuracy (last epoch)
        alldir_training_accuracy[t, it] = history["train_acc"][-1]
        training_loss[t, it] = history["train_loss"][-1]

        # run_nb += 1
#=========
#Plots
#=========
n_participants = 6
colors = plt.cm.tab10(np.linspace(0, 1, n_participants))

# mapping: participant → color
participant_colors = {i: colors[i] for i in range(n_participants)}
#=========
#Plot Direction Performance
#=========

import numpy as np
import matplotlib.pyplot as plt
import os
save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

def plotperf(crossvalit_perfs):
    """
    crossvalit_perfs shape: (n_subjects, n_crossval)
    e.g. (6, 5)
    """

    data = [crossvalit_perfs[:, i] for i in range(crossvalit_perfs.shape[1])]
    pos = np.arange(1, len(data) + 1)

    fig = plt.figure()

    # --- Violin plot ---
    plt.violinplot(
        data,
        pos,
        points=300,
        widths=0.3,
        showmeans=True,
        showextrema=True,
        showmedians=False
    )

    # --- Scatter (raster) points ---
    for i, p in enumerate(pos):
        for s in range(crossvalit_perfs.shape[0]):
            y = crossvalit_perfs[s, i]
            x = np.random.normal(p, 0.03)
            plt.scatter(x, y, alpha=0.7, color=participant_colors[s])

    # --- Connect same subject across CV ---
    n_subjects = crossvalit_perfs.shape[0]

    for s in range(n_subjects):
        y = crossvalit_perfs[s, :]
        plt.plot(pos, y, alpha=0.7, color=participant_colors[s])

    # --- Labels ---
    plt.title('Distribution of direction performances for every cross-validation')
    plt.xlabel('Cross-validation number')
    plt.ylabel('Performance')
    plt.ylim((0, 1))


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(op.join(save_path, "direction_prediction_performance_eog.png"), dpi=300, bbox_inches='tight')

    plt.show()

# using fnc for class
plotperf(crossvalit_alldir_perfs)

#============
#Plot speed rmse
#============

import numpy as np
import matplotlib.pyplot as plt
import os
save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

def plotperf(crossvalit_perfs):
    """
    crossvalit_perfs shape: (n_subjects, n_crossval)
    e.g. (6, 5)
    """

    data = [crossvalit_perfs[:, i] for i in range(crossvalit_perfs.shape[1])]
    pos = np.arange(1, len(data) + 1)

    fig = plt.figure()

    # --- Violin plot ---
    plt.violinplot(
        data,
        pos,
        points=300,
        widths=0.3,
        showmeans=True,
        showextrema=True,
        showmedians=False
    )

    # --- Scatter (raster) points ---
    for i, p in enumerate(pos):
        for s in range(crossvalit_perfs.shape[0]):
            y = crossvalit_perfs[s, i]
            x = np.random.normal(p, 0.03)
            plt.scatter(x, y, alpha=0.7, color=participant_colors[s])

    # --- Connect same subject across CV ---
    n_subjects = crossvalit_perfs.shape[0]

    for s in range(n_subjects):
        y = crossvalit_perfs[s, :]
        plt.plot(pos, y, alpha=0.7, color=participant_colors[s])

    # --- Labels ---
    plt.title('Distribution of speed RMSE for every cross-validation')
    plt.xlabel('Cross-validation number')
    plt.ylabel('RMSE')
    plt.ylim((0, 1.5))


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(op.join(save_path, "speed_prediction_rmse_eog.png"), dpi=300, bbox_inches='tight')

    plt.show()

# using fnc for class
plotperf(crossvalit_allspeed_perfs)

# ==================
#Plot RMSE per speed across corss-validations
# ================== 

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plotseabperf(crossvalit_perfs):
    # shape: (participants, speeds, crossval)

    n_participants, n_speeds, n_cv = crossvalit_perfs.shape

    # -------------------------
    # 🔥 Average over cross-validation
    # -------------------------
    mean_perfs = np.nanmean(crossvalit_perfs, axis=2)  # (6, 16)

    # -------------------------
    # 🔥 Build dataframe
    # -------------------------
    participants = np.repeat(np.arange(n_participants), n_speeds)
    # np.unique(y_speed)
    speeds = np.tile(np.arange(n_speeds)+5, n_participants)

    df = pd.DataFrame({
        "participant": participants,
        "speed": speeds,
        "RMSE": mean_perfs.reshape(-1)
    })

    # -------------------------
    # 🔥 Plot
    # -------------------------
    plt.figure()
    order = np.sort(df["speed"].unique())
    speed_to_idx = {s: i for i, s in enumerate(order)}
    # violin (distribution across participants)
    sns.violinplot(x="speed", y="RMSE", data=df,order=order)

    # 🔥 add dots (each = participant)
    sns.stripplot(x="speed", y="RMSE", data=df, order=order,jitter=True, size=5)

    # 🔥 connect same participant across speeds
    for p in range(n_participants):
        sub = df[df["participant"] == p]
        x_vals = [speed_to_idx[s] for s in sub["speed"]]
        plt.plot(x_vals, sub["RMSE"], alpha=0.7, color=participant_colors[p])

    plt.title('RMSE per speed (averaged across CV)')
    plt.xlabel('Speed (deg/sec)')
    plt.ylabel('RMSE')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(op.join(save_path, "speed_specific_prediction_rmse_eog.png"), dpi=300, bbox_inches='tight')

    plt.show()

# using fnc for class
plotseabperf(crossvalit_speed_perfs)

# test_loader = DataLoader(test_dataset, batch_size=32)

# model.eval()
# all_preds = []
# all_targets = []

# with torch.no_grad():
#     for X, mask, y_dir, y_speed in test_loader:
#         pred_dir, pred_speed = model(X)

#         all_preds.append(pred_dir.argmax(dim=1))
#         all_targets.append(y_dir)