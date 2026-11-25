import numpy as np
import os.path as op
import sys

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
import utils.cnn_xpos_helpers as ch

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

# save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')
# save_name = op.join('eyelink_traces_noblink.png')
# ch.plot_traces(all_x_pos, all_x_dir,
#     all_x_speed, all_subjects, save_path,
#     save_name,
#     channel_idx=0,
#     ylim = [-40, 40],
#     time=None,
#     alpha_trials=0.4,
#     alpha_subject=0.8
# )

# # Proportion of invalid data (>< 30 deg)
fs = 125

sequence_lengths = {
    0.5: int(0.5 * fs),
    1.0: int(1.0 * fs),
    1.5: int(1.5 * fs),
    2.0: int(2.0 * fs),
}
ch.plot_invalid_values_subj(all_x_pos, all_x_speed, all_subjects, sequence_lengths,
                    save_path=op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots'))

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


training_data = {}
test_data = {}
eye_data = {}
results = {}
summary_rows = []

for duration, n_samples in sequence_lengths.items():

    print(f"\n==============================")
    print(f"Training with {duration:.1f} s sequences")
    print(f"==============================")
    #training
    all_train_pos = []
    all_train_eog = []
    all_train_speed = []
    #testing
    all_test_pos = []
    all_test_eog = []
    all_test_speed = []

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

            X_train, x_pos_train = ch.crop_sequences(X_train, x_pos_train, n_samples)

            X_test, x_pos_test = ch.crop_sequences(X_test, x_pos_test, n_samples)

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

            #save TEST data eye X pos and eog X 
            all_test_pos.append(x_pos_test_norm)
            all_test_eog.append(X_test)
            all_test_speed.append(x_speed_test)

            train_dataset = ch.EOGDataset(X_train, x_pos_train_norm,  x_pos_train, x_speed_train) #Makes it pytorch compatible by converting from nyumpy arrays to torch tensors
            test_dataset  = ch.EOGDataset(X_test,  x_pos_test_norm, x_pos_test, x_speed_test)

            model, history = ch.train_model(train_dataset)

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
            # perf_cat_speed = ch.get_pos_performance_per_speed(pred_pos, x_pos_norm, x_speed)
            perf_cat_speed = ch.get_pos_performance_per_speed(pred_pos, x_pos_norm, x_speed, valid=~test_mask, scale=xpos_std)

            summary_data = dict(pred_pos=pred_pos, true_pos=x_pos_norm, true_speed=x_speed)
            summary_perf = ch.per_speed_summary(summary_data, xpos_std)
            # 🔹 store
            crossvalit_allpos_perfs[t, it] = perf_speed #total rmse
            crossvalit_speed_perfs[t, :, it] = perf_cat_speed #rmse per speed
            for speed, metrics in summary_perf.items():
                summary_rows.append(dict(duration=duration, iteration=it, participant=test_participant, speed=speed, **metrics))
            

            # 🔹 training accuracy (last epoch)
            # alldir_training_accuracy[t, it] = history["train_acc"][-1]
            training_loss[t, it] = history["train_loss"][-1]

    training_data[duration] = {
        "training_x_pos" : np.concatenate(all_train_pos),
        "training_eog": np.concatenate(all_train_eog),
        "training_speed": np.concatenate(all_train_speed)
    }
    test_data[duration] = {
        "training_x_pos" : np.concatenate(all_test_pos),
        "training_eog": np.concatenate(all_test_eog),
        "training_speed": np.concatenate(all_test_speed)
    }

    results[duration] = {
    "rmse": crossvalit_allpos_perfs.copy(),
    "speed_rmse": crossvalit_speed_perfs.copy(),
    "loss": training_loss.copy(),
    "pred_pos": np.concatenate(all_pred_pos),
    "true_pos": np.concatenate(all_true_pos), #true position = eyelink eye position
    "true_speed": np.concatenate(all_true_speed), #true speed = target stimulus speed
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

import pandas as pd
# one figure per metric
summary_df = pd.DataFrame(summary_rows)
for m in ["rmse_deg", "r2", "target_sd", "slope", "r"]:
    ch.plot_summary_metric(summary_df, m, unit="participant",
                        save_path=op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots'))
#===============================================================
#Plot amount of data beyond +- 30 deg range within training data
#===============================================================

import os
import matplotlib.pyplot as plt

save_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots')

ch.plot_invalid_values(
    test_data,
    speeds=None,
    durations=None,
    channel_idx=0,
    save_path=save_path,
    filename="invalid_test_values_per_trial.png",
)


#### Plot training data ! here we need to check if normalization impacted some trials.

# ------------------------------------------------------------
# Determine durations
# ------------------------------------------------------------

durations = sorted(training_data.keys())

# ------------------------------------------------------------
# Determine speeds
# ------------------------------------------------------------

n_speeds = np.unique(training_data[durations[0]]["training_speed"]).shape[0]
speeds = np.arange(n_speeds) + 5
speeds = list(speeds)

# ------------------------------------------------------------
# Split speeds into groups of 16
# ------------------------------------------------------------
speed_groups = [
    speeds[i:i + 16]
    for i in range(0, len(speeds), 16)
]

# ------------------------------------------------------------
# One figure per duration × speed group
# ------------------------------------------------------------
for duration in durations:

    for group_idx, speed_group in enumerate(speed_groups):

        n_speeds_group = len(speed_group)

        # ----------------------------------------------------
        # Create 4x4 figure
        # ----------------------------------------------------
        fig, axes = plt.subplots(
            4,
            4,
            figsize=(16, 16),
            squeeze=False
        )

        axes = axes.flatten()

        # ----------------------------------------------------
        # Plot each speed
        # ----------------------------------------------------
        for s, speed in enumerate(speed_group):

            ax = axes[s]

            # ------------------------------------------------
            # Select trials with this speed
            # ------------------------------------------------
            speed_mask = (
                training_data[duration]["training_speed"] == speed
            )

            time = np.arange(training_data[duration]["training_x_pos"].shape[2])
            x_pos = training_data[duration]["training_x_pos"][speed_mask]
            
            channel_idx = 0
            # Handle (trials, channels, time)
            if x_pos.ndim == 3:
                x_pos = x_pos[:, channel_idx, :]
                

            n_trials = x_pos.shape[0]

            x_pos = x_pos[:n_trials]

            # ------------------------------------------------
            # Colors: one color per trial
            # ------------------------------------------------
            trial_colors = plt.cm.turbo(
                np.linspace(0, 1, n_trials)
            )

            all_true = []
            # ------------------------------------------------
            # Plot each trial
            # ------------------------------------------------
            for trial in range(n_trials):

                true_values = x_pos[trial].reshape(-1)

                valid = (
                    np.isfinite(true_values)
                )
                true_values = true_values[valid]

                if len(true_values) == 0:
                    continue

                all_true.append(true_values)
                point_size=5
                ax.scatter(
                    time,
                    true_values,
                    s=point_size,
                    alpha=0.15,
                    color=trial_colors[trial],
                    rasterized=True
                )

            # ------------------------------------------------
            # Overall statistics
            # ------------------------------------------------
            if len(all_true) > 0:

                true_all = np.concatenate(all_true)

            # ------------------------------------------------
            # Formatting
            # ------------------------------------------------
            ax.set_title(
                f"{speed}°/s",
                fontsize=14,
                fontweight="bold"
            )

            ax.set_xlabel(
                "Time",
                fontsize=10
            )

            ax.set_ylabel(
                " x position",
                fontsize=10
            )

            # if xlim is not None:
            #     ax.set_xlim(xlim)

            # if ylim is not None:
            ax.set_ylim(-3,3)

            ax.set_aspect("equal", adjustable="box")

        # ----------------------------------------------------
        # Hide unused panels
        # ----------------------------------------------------
        for s in range(n_speeds_group, 16):
            axes[s].axis("off")

        # ----------------------------------------------------
        # Figure title
        # ----------------------------------------------------
        first_speed = speed_group[0]
        last_speed = speed_group[-1]

        fig.suptitle(
            f"Duration = {duration:.1f} s | "
            f"Speeds {first_speed}–{last_speed}°/s",
            fontsize=20,
            fontweight="bold"
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------
        if save_path is not None:

            os.makedirs(save_path, exist_ok=True)

            # filename_group = (
            #     f"{filename}_duration_{duration:.1f}"
            #     f"_speeds_{first_speed}_{last_speed}.png"
            # )

            # plt.savefig(
            #     op.join(save_path, filename_group),
            #     dpi=300,
            #     bbox_inches="tight"
            # )

        plt.show()