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

    all_X.append(X) #EOG data + event codes of blinks, saccades, pursuits
    all_x_pos.append(x_pos) #Eyelink eye position data = target training data
    all_x_dir.append(x_dir)
    all_x_speed.append(x_speed) #target dot drifting speed (not eye speed)

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

save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
save_name = op.join('eyelink_traces_noblink.png')
ch.plot_traces(all_x_pos, all_x_dir,
    all_x_speed, all_subjects, save_path,
    save_name,
    channel_idx=0,
    ylim = [-40, 40],
    time=None,
    alpha_trials=0.4,
    alpha_subject=0.8
)

#====================================================
# Train, and then test the model after each iteration
#====================================================

fs = 125

sequence_lengths = {
    0.5: int(0.5 * fs),
    1.0: int(1.0 * fs),
    1.5: int(1.5 * fs),
    2.0: int(2.0 * fs),
}

def crop_sequences(X, x_pos, n_samples):
    return (
        X[:, :, :n_samples],
        x_pos[:, :, :n_samples],
    )


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
# allpos_training_accuracy = np.zeros([6,5])

training_data = {}
results = {}

for duration, n_samples in sequence_lengths.items():

    print(f"\n==============================")
    print(f"Training with {duration:.1f} s sequences")
    print(f"==============================")
    #training
    all_train_pos = []
    all_train_eog = []
    all_train_speed = []
    #testing
    all_pred_pos = []
    all_true_pos = []
    all_true_speed = []
    all_fold_ids = []
    all_cv_ids = []
    

    crossvalit_allpos_perfs = np.zeros((6,5))
    crossvalit_speed_perfs = np.zeros((6,16,5))
    training_loss = np.zeros((6,5))
    for it in range(5):
        for t, test_participant in enumerate(participants):
            #it = 0
            #t = 0
            # test_participant = 'sub-04'

            print(f"\n=== Testing on {test_participant} ===")

            train_idx = all_subjects != test_participant
            test_idx  = all_subjects == test_participant

            X_train = all_X[train_idx] #EOG data + event codes of blinks, saccades, pursuits for training
            x_speed_train = all_x_speed[train_idx]
            x_pos_train = all_x_pos[train_idx] #Eyelink Eye position training data = target for the model

            X_test = all_X[test_idx] #EOG data + event codes of blinks, saccades, pursuits for testing
            x_speed_test = all_x_speed[test_idx]
            x_pos_test = all_x_pos[test_idx] #Eyelink Eye position test data

            #Crop sequences

            X_train, x_pos_train = crop_sequences(
            X_train,
            x_pos_train,
            n_samples
            )

            X_test, x_pos_test = crop_sequences(
                X_test,
                x_pos_test,
                n_samples
            )

            # ---------------------------------------------------
            # TRAIN
            # ---------------------------------------------------
            #-----------------------------
            #Mask blinks and normalize
            #-----------------------------
            #Training data masks
            #mask 1 = blinks

            # x_evt = X_train[:, 2:, :]  # (B, 6, T)
            blink_mask = X_train[:, 2:4, :] > 0
            # valid_train_mask1 = ~blink_mask #keep everything that is not blink

            #mask 2 = eye position is far away from screen
            farpos_mask = (x_pos_train < -30) | (x_pos_train > 30)
            train_mask = blink_mask | farpos_mask

            X_train_eog = X_train[:, :2, :]
            # X_train_eog[train_mask] = 0
            # x_pos_train[train_mask] = 0
            # X_train_eog = ch.interpolate_gaps(X_train[:, :2, :],train_mask)
            # x_pos_train = ch.interpolate_gaps(x_pos_train, train_mask)

            # ch.plot_trial(x_pos_train,trial_idx=2)
            # ---------------------------------------------------
            # TEST
            # ---------------------------------------------------
            #Test data masks
            #mask 1 = blinks
            # x_evt = X_test[:, 2:, :]  # (B, 6, T)
            test_blink_mask = X_test[:, 2:4, :] > 0
            # valid_test_mask1 = ~test_blink_mask #keep everything that is not blink

            #mask 2 = eye position is far away from screen
            test_farpos_mask = (x_pos_test < -30) | (x_pos_test > 30)
            # valid_test_mask2 = ~test_farpos_mask  #If eye position is far away from screen, remove this part as well

            test_mask = test_blink_mask | test_farpos_mask
            X_test_eog = X_test[:, :2, :]
            # X_test_eog[test_mask] = 0
            # x_pos_test[test_mask] = 0
            # X_test_eog = ch.interpolate_gaps(X_test[:, :2, :],test_mask)
            # x_pos_test = ch.interpolate_gaps(x_pos_test,test_mask)

            # eog_train = X_train[:, :2, :] #EOG data
            eog_mean = X_train_eog[~train_mask].mean()
            eog_std  = X_train_eog[~train_mask].std()

            #Eyelink Eye position training data
            xpos_mean = x_pos_train[~train_mask].mean()
            xpos_std  = x_pos_train[~train_mask].std()

            # Normalize EOG data
            X_train[:, :2, :] = (X_train_eog - eog_mean) / (eog_std + 1e-8) 
            X_test[:,  :2, :] = (X_test_eog - eog_mean) / (eog_std + 1e-8)

            # Normalize Eyelink data
            x_pos_train_norm = (x_pos_train - xpos_mean) / (xpos_std + 1e-8)
            x_pos_test_norm  = (x_pos_test  - xpos_mean) / (xpos_std + 1e-8)

            #save TRAINING data eye X pos and eog X 
            all_train_pos.append(x_pos_train_norm)
            all_train_eog.append(X_train)
            all_train_speed.append(x_speed_train)

            train_dataset = EOGDataset(X_train, x_pos_train_norm,  x_pos_train, x_speed_train) #Makes it pytorch compatible by converting from nyumpy arrays to torch tensors
            test_dataset  = EOGDataset(X_test,  x_pos_test_norm, x_pos_test, x_speed_test)

            model, history = train_model(train_dataset)

            # 🔹 predictions
            pred_pos, x_pos_norm, x_speed = ch.get_network_output(model, test_dataset)

            pred_real = pred_pos * xpos_std + xpos_mean

            # 🔹 save TEST predictions
            all_pred_pos.append(pred_pos)
            all_true_pos.append(x_pos_norm)
            all_true_speed.append(x_speed)

            all_fold_ids.append(np.full_like(pred_pos, t))
            all_cv_ids.append(np.full_like(pred_pos, it))

            # 🔹 overall performance
            pred_pos[test_mask] = np.nan
            x_pos_norm[test_mask] = np.nan
            perf_speed = ch.get_pos_performance(pred_pos, x_pos_norm)

            # 🔹 category performance
            perf_cat_speed = ch.get_pos_performance_per_speed(pred_pos, x_pos_norm, x_speed)
            # 🔹 store
            crossvalit_allpos_perfs[t, it] = perf_speed #total rmse
            crossvalit_speed_perfs[t, :, it] = perf_cat_speed #rmse per speed

            # 🔹 training accuracy (last epoch)
            # alldir_training_accuracy[t, it] = history["train_acc"][-1]
            training_loss[t, it] = history["train_loss"][-1]

    training_data[duration] = {
        "training_x_pos" : np.concatenate(all_train_pos),
        "training_eog": np.concatenate(all_train_eog),
        "training_speed": np.concatenate(all_train_speed)
    }

    results[duration] = {
    "rmse": crossvalit_allpos_perfs.copy(),
    "speed_rmse": crossvalit_speed_perfs.copy(),
    "loss": training_loss.copy(),
    "pred_pos": np.concatenate(all_pred_pos),
    "true_pos": np.concatenate(all_true_pos), #true position = eyelink eye position
    "true_speed": np.concatenate(all_true_speed),
    "fold_ids": np.concatenate(all_fold_ids),
    "cv_ids": np.concatenate(all_cv_ids),
    }


# all_true_pos = np.concatenate(all_true_pos)
# all_pred_pos = np.concatenate(all_pred_pos)
# all_fold_ids = np.concatenate(all_fold_ids)
# all_cv_ids = np.concatenate(all_cv_ids)

#===================================================================================================================================
# Figure
#===================================================================================================================================
#Plot predicted vs true trial traces
save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
ch.plot_pred_vs_true(
    all_true_pos[3600:3610,:,:],
    all_pred_pos[3600:3610,:,:],
    subjects=None,
    save_path= save_path,
    save_name = "pred_vs_true_trials_validmask.png",
    channel_idx=0,
    time=None,
    max_trials=20          # avoid plotting hundreds

)

#Plot predicted as a function of true trial traces, for each speed.
save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

ch.plot_pred_vs_true_position_clouds(
    results,
    speeds=None,
    durations=None,
    subjects=None,
    max_trials=None,
    channel_idx=0,
    save_path=save_path,
    filename="predicted_vs_true_position_clouds_inf.png",
    alpha=0.08,
    point_size=5,
    xlim=None,
    ylim=None,
)


# ==================
#Plot RMSE per speed across cross-validations
# ================== 

import matplotlib.pyplot as plt
import os.path as op

save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
n_participants = 6
colors = plt.cm.tab10(np.linspace(0, 1, n_participants))
participant_colors = {i: colors[i] for i in range(n_participants)}
ch.plot_speed_rmse_duration_comparison(results, participant_colors,
                                   save_path=save_path,
                                   filename="speed_specific_prediction_rmse_by_duration.png")

#==========================================================
#Plot amount of NaN or infinite values within training data
#==========================================================

import os
import matplotlib.pyplot as plt

save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

ch.plot_invalid_values_per_trial(
    training_data,
    speeds=None,
    durations=None,
    channel_idx=0,
    max_trials=None,
    percentage=True,
    save_path=save_path,
    filename="invalid_values_per_trial.png",
)