import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import os.path as op
import sys

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
from utils.classifier_helpers import get_network_output, get_network_cat_performance, get_direction_performance, get_speed_performance, get_speed_performance_per_condition

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

#Dataset
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
        return self.X[idx],  self.x_mask[idx], self.y_dir[idx], self.y_speed[idx]
    
#----------------------    
# 2: Model design
#----------------------
import torch.nn as nn
import torch.nn.functional as F

class EOGNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        #n_channels =train_dataset.X.shape[1]
        #self = EOGNet(n_channels=8)
        #x_mask =  mask_train 
        #x = X
        # ---- CNN layers ----
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=101, padding=50)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=51, padding=25)

        # self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=101, stride=2, padding=50) #add stride = remove pooling layer
        # self.conv2 = nn.Conv1d(32, 64, kernel_size=51, stride=2, padding=25)
        self.pool = nn.MaxPool1d(2)

        # ---- LSTM ----
        self.lstm = nn.LSTM(64, 64, batch_first=True)

        # ---- Heads ----
        self.fc_dir = nn.Linear(64, 2)   # left vs right
        self.fc_speed = nn.Linear(64, 1)

    def forward(self, x, x_mask):
        x_mask = x_mask.any(dim=1, keepdim=True).float() #add a dimension for the channels dimension of x
        # x: (batch, channels, time)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x_mask = self.pool(x_mask)
        # print("X After pooling:", x.shape)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x_mask = self.pool(x_mask)

        # reshape for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)   # (B, T, F) B = batch, T = time, F = features
        x, _ = self.lstm(x)
        # print("X After LSTM:", x.shape)

        # 🔥 masked temporal pooling → ONE vector per trial
        x_mask = x_mask.squeeze(1).unsqueeze(-1) # (B, T) ; then # (B, T, 1)
        # print("X mask After Pooling:", x_mask.shape)
        x = (x * x_mask).sum(dim=1) / (x_mask.sum(dim=1) + 1e-8)

        dir_out = self.fc_dir(x)      # (B, 2)
        # print("Final dir_out:", dir_out.shape)
        speed_out = self.fc_speed(x)  # (B, 1)

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

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        epoch_acc = get_direction_performance(all_preds, all_targets)

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
    X, x_mask, y_dir, y_speed = load_mat_data(filepath)

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
        pred_dir, pred_speed, y_dir, y_speed = get_network_output(model, test_dataset)

        # 🔹 overall performance
        perf_dir = get_direction_performance(pred_dir, y_dir)
        perf_speed = get_speed_performance(pred_speed, y_speed)

        # 🔹 category performance
        perf_cat_dir = get_network_cat_performance(pred_dir, y_dir, n_classes=2)
        perf_cat_speed = get_speed_performance_per_condition(pred_speed, y_speed)
        # 🔹 store
        crossvalit_alldir_perfs[t, it] = perf_dir
        crossvalit_allspeed_perfs[t, it] = perf_speed #total rmse
        crossvalit_dir_perfs[t, :, it] = perf_cat_dir
        crossvalit_speed_perfs[t, :, it] = perf_cat_speed #rmse per speed

        # 🔹 training accuracy (last epoch)
        alldir_training_accuracy[t, it] = history["train_acc"][-1]
        training_loss[t, it] = history["train_loss"][-1]

        # run_nb += 1





test_loader = DataLoader(test_dataset, batch_size=32)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X, mask, y_dir, y_speed in test_loader:
        pred_dir, pred_speed = model(X)

        all_preds.append(pred_dir.argmax(dim=1))
        all_targets.append(y_dir)