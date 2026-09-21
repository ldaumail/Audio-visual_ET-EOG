
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as op
import numpy as np



def load_mat_data(filepath):

    data = sio.loadmat(filepath)

    eog = data['data'][0, 0]['sub01'][0, 0]['eog_filt_xpos'][::16,:,4:10,:] #Downsample 16 times with ::4. 4:8 = auditory only condition, eyes open. 8:10 = visible only, eyes open

    blink = data['data'][0, 0]['sub01'][0, 0]['blinkTrs'][::16,:,0:6,:]
    saccade = data['data'][0, 0]['sub01'][0, 0]['saccadesTrs'][::16,:,0:6,:]
    pursuit = data['data'][0, 0]['sub01'][0, 0]['pursuitTrs'][::16,:,0:6,:]
    
    eyelink_x = data['data'][0, 0]['sub01'][0, 0]['origx'][::16,:,0:6,:] #blinkFreeX

    conditions = data['data'][0,0]['eyelink_conds'][0][0:6]
    speed = data['data'][0, 0]['sub01'][0, 0]['driftingSpeeds'][:,4:10]

    # -------------------------
    # Clean direction labels
    # -------------------------
    dir_labels = []
    cond_labels = []
    for item in conditions:
        string_val = str(item[0])
        if 'Leftward' in string_val:
            dir_labels.append(0)
        elif 'Rightward' in string_val:
            dir_labels.append(1)
        if 'Vis' in string_val:
            cond_labels.append(1)
        else:
            cond_labels.append(0)
        

    dir_labels = np.array(dir_labels)  # (n_conditions,)
    cond_labels = np.array(cond_labels)

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
    cond = []
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
            cond.append(cond_labels[c])
            x_speed.append(speed[t, c])

    X = np.array(X_list)          # (N_trials, channels, time)
    # X = np.nan_to_num(X, nan=0.0)
    x_pos = np.array(x_pos)
    x_dir = np.array(x_dir)
    cond = np.array(cond)
    x_speed = np.array(x_speed)

    return X, x_pos, x_dir, cond, x_speed




data_dir = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'data', 'v05')

participants = {'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'}
all_X = []
all_x_pos = []
all_x_dir = []
all_cond = []
all_x_speed = []
all_subjects = []
for participant in participants:
    filepath = op.join(data_dir, f"{participant}_v05", f"{participant}_model_data_filt_audio_visual_motion_v05_1.mat")
    X, x_pos, x_dir, cond, x_speed = load_mat_data(filepath)

    all_X.append(X) #EOG data + event codes of blinks, saccades, pursuits
    all_x_pos.append(x_pos) #Eyelink eye position data
    all_x_dir.append(x_dir)
    all_cond.append(cond)
    all_x_speed.append(x_speed)

    # 👇 critical for LOSO
    all_subjects.append(np.array([participant] * len(X)))

all_X = np.concatenate(all_X, axis=0)
all_x_pos = np.concatenate(all_x_pos, axis=0)
all_x_dir = np.concatenate(all_x_dir, axis=0)
all_cond = np.concatenate(all_cond, axis=0)
all_x_speed = np.concatenate(all_x_speed, axis=0)
all_subjects = np.concatenate(all_subjects, axis=0)


# Plot average EOG across participants ± SEM
import numpy as np
import matplotlib.pyplot as plt

def plot_average_eog(
    all_X,
    all_x_speed,
    all_x_dir,
    all_cond,
    all_subjects,
    speed_to_plot,
    eye=0,
    save_file="average_EOG.svg",
    fs=125,
):

    eye_names = ["Left eye", "Right eye"]
    direction_names = ["Leftward", "Rightward"]
    condition_names = ["Audio", "Visual"]   # 0=Audio, 1=Visual

    time = np.arange(all_X.shape[-1]) / fs

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True,
        sharey=True
    )

    for direction in [0, 1]:

        ax = axes[direction]

        for condition in [0, 1]:

            mask = (
                (all_x_speed == speed_to_plot) &
                (all_x_dir == direction) &
                (all_cond == condition)
            )

            X = all_X[mask]
            subjects = all_subjects[mask]

            participant_means = []

            for subject in np.unique(subjects):

                subj_mask = subjects == subject

                participant_mean = X[subj_mask, eye, :].mean(axis=0)

                participant_means.append(participant_mean)

            participant_means = np.stack(participant_means)

            mean_trace = participant_means.mean(axis=0)
            sem_trace = (
                participant_means.std(axis=0, ddof=1)
                / np.sqrt(participant_means.shape[0])
            )

            ax.plot(
                time,
                mean_trace,
                lw=2,
                label=condition_names[condition]
            )

            ax.fill_between(
                time,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                alpha=0.25
            )

        ax.axhline(0, color="k", ls="--", lw=0.75)

        ax.set_title(direction_names[direction])
        ax.set_ylabel("EOG (mA)")
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle(
        f"{eye_names[eye]}\n"
        f"Speed = {speed_to_plot}°/s\n"
        "Mean across participants ± SEM",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(save_file, format="svg", bbox_inches="tight")
    plt.show()

save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
plot_average_eog(
    all_X,
    all_x_speed,
    all_x_dir,
    all_cond,
    all_subjects,
    speed_to_plot=20,
    save_file=op.join(save_path,"Average_EOG_speed20deg.svg"),
)