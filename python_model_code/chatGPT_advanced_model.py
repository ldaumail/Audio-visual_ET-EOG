import numpy as np
import scipy.io as sio

def load_mat_data(filepath):
    data = sio.loadmat(filepath)

    eog = data['EOG']                  # (n_trials, time, channels)
    velocity = data['eyelink_vel']     # (n_trials, time)

    direction = data['direction']      # (n_trials,)
    speed = data['speed']              # (n_trials,)

    blink = data['blink']
    saccade = data['saccade']
    pursuit = data['pursuit']

    return eog, velocity, direction, speed, blink, saccade, pursuit


import torch
from torch.utils.data import Dataset

class EOGDataset(Dataset):
    def __init__(self, eog, velocity, direction, speed, blink, saccade, pursuit):
        self.X = []
        self.vel = velocity
        self.y_dir = direction
        self.y_speed = speed

        for i in range(eog.shape[0]):
            trial = np.concatenate([
                eog[i].T,
                blink[i][None, :],
                saccade[i][None, :],
                pursuit[i][None, :]
            ], axis=0)

            # normalize per trial
            trial = (trial - trial.mean()) / (trial.std() + 1e-8)

            self.X.append(trial)

        self.X = np.array(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.vel[idx], dtype=torch.float32),
            torch.tensor(self.y_dir[idx], dtype=torch.long),
            torch.tensor(self.y_speed[idx], dtype=torch.float32)
        )
    
import torch.nn as nn
import torch.nn.functional as F

class EOGEncoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(64, 64, batch_first=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)  # (batch, time, features)

        x, _ = self.lstm(x)

        return x  # full sequence

class VelocityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, time, features)
        return self.fc(x).squeeze(-1)  # (batch, time)
    

class TaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_dir = nn.Linear(64, 2)
        self.fc_speed = nn.Linear(64, 1)

    def forward(self, x):
        x_last = x[:, -1, :]
        return self.fc_dir(x_last), self.fc_speed(x_last)
    
import torch.optim as optim

def pretrain_velocity(dataset):

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    encoder = EOGEncoder(n_channels=dataset.X.shape[1])
    head = VelocityHead()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0

        for X, vel, _, _ in loader:
            optimizer.zero_grad()

            features = encoder(X)
            pred_vel = head(features)

            # match lengths (due to pooling)
            min_len = min(pred_vel.shape[1], vel.shape[1])
            loss = loss_fn(pred_vel[:, :min_len], vel[:, :min_len])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Pretrain] Epoch {epoch+1} - Loss: {total_loss:.4f}")

    return encoder

def finetune_task(dataset, encoder):

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    head = TaskHead()

    # 🔥 Option: freeze encoder first
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(head.parameters(), lr=1e-3)

    loss_dir_fn = nn.CrossEntropyLoss()
    loss_speed_fn = nn.MSELoss()

    for epoch in range(15):
        total_loss = 0

        for X, _, y_dir, y_speed in loader:
            optimizer.zero_grad()

            with torch.no_grad():
                features = encoder(X)

            pred_dir, pred_speed = head(features)

            loss_dir = loss_dir_fn(pred_dir, y_dir)
            loss_speed = loss_speed_fn(pred_speed.squeeze(), y_speed)

            loss = loss_dir + 0.5 * loss_speed
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Finetune] Epoch {epoch+1} - Loss: {total_loss:.4f}")

    return encoder, head

def full_finetune(dataset, encoder, head):

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for param in encoder.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=1e-4
    )

    loss_dir_fn = nn.CrossEntropyLoss()
    loss_speed_fn = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0

        for X, _, y_dir, y_speed in loader:
            optimizer.zero_grad()

            features = encoder(X)
            pred_dir, pred_speed = head(features)

            loss = loss_dir_fn(pred_dir, y_dir) + 0.5 * loss_speed_fn(pred_speed.squeeze(), y_speed)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Full FT] Epoch {epoch+1} - Loss: {total_loss:.4f}")

from torch.utils.data import DataLoader

eog, vel, direction, speed, blink, saccade, pursuit = load_mat_data("data.mat")

dataset = EOGDataset(eog, vel, direction, speed, blink, saccade, pursuit)

# Stage 1
encoder = pretrain_velocity(dataset)

# Stage 2
encoder, head = finetune_task(dataset, encoder)

# Optional best step
full_finetune(dataset, encoder, head)