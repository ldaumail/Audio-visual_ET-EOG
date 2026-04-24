   
import scipy.io as sio
import os.path as op
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = op.join(
    '/Users', 'ldaumail3', 'Documents', 'research',
    'eyetracking_EOG', 'Audio_Visual_EOG-main', 'data', 'v05'
)

participants = ['sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']

# Original sampling rate
fs_orig = 2000

# Target sampling rates to test
target_fs_list = [2000, 1000, 500, 250, 125, 62.5, 31.25]

def downsample(data, factor):
    """Simple decimation (no filtering). Replace with proper filter if needed."""
    return data[::factor]

def plot_signals(eog, eyelink, fs, title=""):
    """Plot a single trial comparison"""
    t = np.arange(eog.shape[0]) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, eog[:], label="EOG", alpha=0.7)
    plt.plot(t, eyelink[:], label="Eyelink", alpha=0.7)
    plt.title(f"{title} (fs={fs} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_participant = 'sub-07'
all_data = {}
# where to save figures
save_dir = op.join('/Users','ldaumail3','Documents','research','eyetracking_EOG','Audio_Visual_EOG-main','analysis_plots', "plots_downsampling")
os.makedirs(save_dir, exist_ok=True)

for target_fs in target_fs_list:
    factor = int(fs_orig / target_fs)
    
    all_X = []
    all_x_pos = []
    all_subjects = []

    filepath = op.join(
        data_dir,
        f"{plot_participant}_v05",
        f"{plot_participant}_model_data_filt_audio_visual_motion_v05.mat"
    )
    
    data = sio.loadmat(filepath)

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][:, :, 8:10, :]
    eyelink_x = data['data'][0, 0]['sub01'][0, 0]['origx'][:, :, 4:6, :]

    # Downsample
    eog_ds = eog[::factor]
    eyelink_ds = eyelink_x[::factor]

    all_X.append(eog_ds)
    all_x_pos.append(eyelink_ds)
    all_subjects.append(np.array([plot_participant] * eog_ds.shape[0]))

    # === Extract one example trace ===
    eog_trace = eog_ds[:, 0, 0, 0]
    eyelink_trace = eyelink_ds[:, 0, 0, 0]

    t = np.arange(eog_trace.shape[0]) / target_fs

    # === Create subplots ===
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # EOG subplot
    axes[0].plot(t, eog_trace)
    axes[0].set_title(f"EOG — {plot_participant} (fs={target_fs} Hz)")
    axes[0].set_ylabel("EOG position")

    # Eyelink subplot
    axes[1].plot(t, eyelink_trace)
    axes[1].set_title("Eyelink")
    axes[1].set_ylabel("Eye position")
    axes[1].set_xlabel("Time (s)")

    plt.tight_layout()

    # === Save figure ===
    save_path = op.join(save_dir, f"{plot_participant}_fs{target_fs}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)  # important when looping

    # Store data
    all_data[target_fs] = {
        "X": np.concatenate(all_X, axis=0),
        "x_pos": np.concatenate(all_x_pos, axis=0),
        "subjects": np.concatenate(all_subjects, axis=0)
    }