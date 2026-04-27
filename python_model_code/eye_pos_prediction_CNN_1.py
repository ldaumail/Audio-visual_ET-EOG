import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as op
import sys

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
import utils.cnn_xpos_helpers as ch


#-----------------------------
# 1: Generate Pytorch Dataset
#-----------------------------
class EOGDataset(Dataset):
    def __init__(self, X, x_pos, x_speed):
        # X = X_train
        # x_pos = x_pos_train

        self.X = torch.tensor(X, dtype=torch.float32)
        self.x_pos = torch.tensor(x_pos, dtype=torch.float32)
        self.x_speed = torch.tensor(x_speed, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.x_pos[idx], self.x_speed[idx]


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
        x_evt = x[:, 2:, :]

        blink_eog   = x_eog * x_evt[:, 0:2, :]
        sacc_eog    = x_eog * x_evt[:, 2:4, :]
        pursuit_eog = x_eog * x_evt[:, 4:6, :]

        # x = torch.cat([blink_eog, sacc_eog, pursuit_eog], dim=1)  # (B, 6, T)
        x = torch.cat([blink_eog, sacc_eog, pursuit_eog], dim=1)  # (B, 6, T)
        
        # CNN
        x = F.relu(self.conv1(x))   # (B, 16, T) x shape: B = 32, 16, T = 250
        x = F.relu(self.conv2(x))   # (B, 32, T) x shape: B = 32, 32, T = 250 

        # 🔥 NO flatten
        x = self.head(x)            # (B, 2, T)

        return x.squeeze(1)         # (B, T)
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
        for X, x_pos, x_speed in loader:
            optimizer.zero_grad() # reset gradients at new batch (clear old gradients)
            
            pred_pos = model(X)

            x_evt = X[:, 2:, :]
            blink_mask = x_evt[:, 0:2, :] > 0
            valid_mask = ~blink_mask

            loss = ((pred_pos - x_pos)**2)[valid_mask].mean()
            # loss = loss_pos_fn(pred_pos, x_pos)

            loss.backward() #compute new gradients
            optimizer.step() #update weights

            total_loss += loss.item()


        history["train_loss"].append(total_loss)

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

    return model, history

#-------------------------------
#Prepare all participants data
#-------------------------------
data_dir = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'data', 'v05')

participants = {'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'}
all_X = []
all_x_pos = []
all_x_dir = []
all_x_speed = []
all_subjects = []
for participant in participants:
    filepath = op.join(data_dir, f"{participant}_v05", f"{participant}_model_data_filt_audio_visual_motion_v05_1.mat")
    X, x_pos, x_dir, x_speed = ch.load_mat_data(filepath)

    all_X.append(X)
    all_x_pos.append(x_pos)
    all_x_dir.append(x_dir)
    all_x_speed.append(x_speed)

    # 👇 critical for LOSO
    all_subjects.append(np.array([participant] * len(X)))

all_X = np.concatenate(all_X, axis=0)
all_x_pos = np.concatenate(all_x_pos, axis=0)
all_x_dir = np.concatenate(all_x_dir, axis=0)
all_x_speed = np.concatenate(all_x_speed, axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)

#====================================================
# Check data and conditions
#====================================================

# save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

# ch.plot_eog_traces(all_X, all_x_dir,
#     all_x_speed, all_subjects, save_path,
#     channel_idx=0,
#     time=None,
#     alpha_trials=0.4,
#     alpha_subject=0.8
# )

#====================================================
# Train, and then test the model after each iteration
#====================================================
all_pred_pos = []
all_true_pos = []
all_true_speed = []
all_fold_ids = []
all_cv_ids = []

crossvalit_allpos_perfs = np.zeros([6,5])
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
# alldir_training_accuracy = np.zeros([6,5])
# allspeed_training_accuracy = np.zeros([6,5])
allpos_training_accuracy = np.zeros([6,5])

training_loss = np.zeros([6,5])


for it in range(5):
    for t, test_participant in enumerate(participants):
        #it = 0
        #t = 0
        # test_participant = 'sub-04'

        print(f"\n=== Testing on {test_participant} ===")

        train_idx = all_subjects != test_participant
        test_idx  = all_subjects == test_participant

        X_train = all_X[train_idx]
        x_speed_train = all_x_speed[train_idx]
        x_pos_train = all_x_pos[train_idx]

        X_test = all_X[test_idx]
        x_speed_test = all_x_speed[test_idx]
        x_pos_test = all_x_pos[test_idx]

        #-----------------------------
        #Mask blinks and normalize
        #-----------------------------
        x_evt = X_train[:, 2:, :]  # (B, 6, T)
        blink_mask = x_evt[:, 0:2, :] > 0
        valid_mask = ~blink_mask

        eog_train = X_train[:, :2, :]
        eog_mean = eog_train[valid_mask].mean()
        eog_std  = eog_train[valid_mask].std()

        xpos = x_pos_train
        xpos_mean = xpos[valid_mask].mean()
        xpos_std  = xpos[valid_mask].std()

        X_train[:, :2, :] = (X_train[:, :2, :] - eog_mean) / (eog_std + 1e-8)
        X_test[:,  :2, :] = (X_test[:,  :2, :] - eog_mean) / (eog_std + 1e-8)

        x_pos_train = (x_pos_train - xpos_mean) / (xpos_std + 1e-8)
        x_pos_test  = (x_pos_test  - xpos_mean) / (xpos_std + 1e-8)


        train_dataset = EOGDataset(X_train, x_pos_train, x_speed_train)
        test_dataset  = EOGDataset(X_test,  x_pos_test, x_speed_test)

        model, history = train_model(train_dataset)

        # 🔹 predictions
        pred_pos, x_pos, x_speed = ch.get_network_output(model, test_dataset)

        pred_real = pred_pos * xpos_std + xpos_mean

        # 🔹 save predictions
        all_pred_pos.append(pred_pos)
        all_true_pos.append(x_pos)
        all_true_speed.append(x_speed)

        all_fold_ids.append(np.full_like(pred_pos, t))
        all_cv_ids.append(np.full_like(pred_pos, it))

        # 🔹 overall performance

        perf_speed = ch.get_pos_performance(pred_pos, x_pos)

        # 🔹 category performance
        perf_cat_speed = ch.get_pos_performance_per_speed(pred_pos, x_pos, x_speed)
        # 🔹 store
        crossvalit_allpos_perfs[t, it] = perf_speed #total rmse
        crossvalit_speed_perfs[t, :, it] = perf_cat_speed #rmse per speed

        # 🔹 training accuracy (last epoch)
        # alldir_training_accuracy[t, it] = history["train_acc"][-1]
        training_loss[t, it] = history["train_loss"][-1]


all_true_pos = np.concatenate(all_true_pos)
all_pred_pos = np.concatenate(all_pred_pos)
all_fold_ids = np.concatenate(all_fold_ids)
all_cv_ids = np.concatenate(all_cv_ids)

#===================================================================================================================================
# Figure
#===================================================================================================================================
#Plot predicted vs true trial traces
save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
ch.plot_pred_vs_true(
    all_true_pos[:11,:,:],
    all_pred_pos[:11,:,:],
    subjects=None,
    save_path= save_path,
    channel_idx=0,
    time=None,
    max_trials=20,          # avoid plotting hundreds
)


#============
#Plot position rmse
#============

import numpy as np
import matplotlib.pyplot as plt
import os
n_participants = 6
colors = plt.cm.tab10(np.linspace(0, 1, n_participants))

# mapping: participant → color
participant_colors = {i: colors[i] for i in range(n_participants)}
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
    plt.title('Distribution of eye position RMSE for every cross-validation')
    plt.xlabel('Cross-validation number')
    plt.ylabel('RMSE')
    # plt.ylim((0, 2))


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(op.join(save_path, "xpos_prediction_rmse_filt9_norm-eyelink.png"), dpi=300, bbox_inches='tight')

    plt.show()

# using fnc for class
plotperf(crossvalit_allpos_perfs)

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
    plt.savefig(op.join(save_path, "speed_specific_prediction_rmse_filt9_norm-eyelink.png"), dpi=300, bbox_inches='tight')

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



