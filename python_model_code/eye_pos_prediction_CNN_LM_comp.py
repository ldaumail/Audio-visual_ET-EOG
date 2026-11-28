import numpy as np
import pandas as pd
import os.path as op
import sys
from sklearn.linear_model import LinearRegression
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
# fs = 125

# sequence_lengths = {
#     0.5: int(0.5 * fs),
#     1.0: int(1.0 * fs),
#     1.5: int(1.5 * fs),
#     2.0: int(2.0 * fs),
# }
# ch.plot_invalid_values_subj(all_x_pos, all_x_speed, all_subjects, sequence_lengths,
#                     save_path=op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots'))

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
# ---------- feature builders (per eye e; channel layout: 0-1 EOG, 2-3 blink, 4-5 sacc, 6-7 pursuit) ----------
def feat_eog(X, e):
    return X[:, e, :][..., None]                                   # (N, T, 1)

def feat_gated(X, e):                                              # same gating as EOGNet
    eog = X[:, e, :]
    return np.stack([eog * X[:, 2 + e, :],
                     eog * X[:, 4 + e, :],
                     eog * X[:, 6 + e, :]], axis=-1)               # (N, T, 3)

FEATURES = {"lr_eog": feat_eog, "lr_gated": feat_gated}

def fit_predict_lr(feat_fn, X_train, y_train_norm, valid_train, X_test):
    """One LinearRegression per eye, fit on valid samples only. Returns (N_test, 2, T) predictions."""
    N, _, T = X_test.shape
    pred = np.zeros((N, 2, T))
    for e in range(2):
        Ftr = feat_fn(X_train, e)
        nf = Ftr.shape[-1] #features count
        Ftr = Ftr.reshape(-1, nf)
        ytr = y_train_norm[:, e, :].reshape(-1)
        v = valid_train[:, e, :].reshape(-1)
        lr = LinearRegression().fit(Ftr[v], ytr[v])
        pred[:, e, :] = lr.predict(feat_fn(X_test, e).reshape(-1, nf)).reshape(N, T)
    return pred

def evaluate(pred, true_norm, true_deg, speed, test_mask, xpos_mean, xpos_std):
    valid = ~test_mask
    p = np.where(valid, pred, np.nan)            # masked COPIES, originals untouched
    t = np.where(valid, true_norm, np.nan)
    overall = ch.get_pos_performance(p, t)
    per_speed = ch.get_pos_performance_per_speed(p, t, speed, valid=valid, scale=xpos_std)
    summ = ch.per_speed_summary(dict(pred_pos=p, true_pos=t, true_speed=speed), xpos_std)
    bins = ch.rmse_by_position_bin(pred, true_deg, speed, valid, xpos_mean, xpos_std)
    return overall, per_speed, summ, bins

# ---------- setup ----------
participants = sorted(participants)              # fixed order
MODELS = ["cnn", *FEATURES.keys()]
n_subj, n_cvit, n_speeds = len(participants), 5, 16

summary_rows, bin_rmse_rows = [], []
perf_overall = {}          # perf_overall[duration][model] -> (n_subj, n_cvit)
perf_speed = {}            # perf_speed[duration][model]   -> (n_subj, n_speeds, n_cvit)
preds = {}                 # preds[duration][model] -> list of unmasked predictions
training_loss = {}

for duration, n_samples in sequence_lengths.items():
    print(f"\n=== Training with {duration:.1f} s sequences ===")
    perf_overall[duration] = {m: np.full((n_subj, n_cvit), np.nan) for m in MODELS}
    perf_speed[duration]   = {m: np.full((n_subj, n_speeds, n_cvit), np.nan) for m in MODELS}
    preds[duration]        = {m: [] for m in MODELS}
    true_list, speed_list, mask_list = [], [], []
    training_loss[duration] = np.zeros((n_subj, n_cvit))

    for it in range(n_cvit):
        for t, test_participant in enumerate(participants):
            print(f"Testing on {test_participant} (it {it})")
            train_idx = all_subjects != test_participant
            test_idx  = all_subjects == test_participant

            X_train, x_pos_train = ch.crop_sequences(all_X[train_idx], all_x_pos[train_idx], n_samples)
            X_test,  x_pos_test  = ch.crop_sequences(all_X[test_idx],  all_x_pos[test_idx],  n_samples)
            x_speed_train, x_speed_test = all_x_speed[train_idx], all_x_speed[test_idx]
            X_train, X_test = X_train.copy(), X_test.copy()      # crop returns views

            # masks (blinks or |pos| > 30 deg), shape (N, 2, T)
            train_mask = (X_train[:, 2:4, :] > 0) | (np.abs(x_pos_train) > 30)
            test_mask  = (X_test[:, 2:4, :] > 0)  | (np.abs(x_pos_test)  > 30)

            # normalization stats from valid training samples only
            eog_mean = X_train[:, :2, :][~train_mask].mean()
            eog_std  = X_train[:, :2, :][~train_mask].std()
            xpos_mean = x_pos_train[~train_mask].mean()
            xpos_std  = x_pos_train[~train_mask].std()

            X_train[:, :2, :] = (X_train[:, :2, :] - eog_mean) / (eog_std + 1e-8)
            X_test[:,  :2, :] = (X_test[:,  :2, :] - eog_mean) / (eog_std + 1e-8)
            y_train_norm = (x_pos_train - xpos_mean) / (xpos_std + 1e-8)
            y_test_norm  = (x_pos_test  - xpos_mean) / (xpos_std + 1e-8)

            # ---------- predictions from every model ----------
            train_ds = ch.EOGDataset(X_train, y_train_norm, x_pos_train, x_speed_train)
            test_ds  = ch.EOGDataset(X_test,  y_test_norm,  x_pos_test,  x_speed_test)
            model, history = ch.train_model(train_ds)
            pred_cnn, _, _ = ch.get_network_output(model, test_ds)
            training_loss[duration][t, it] = history["train_loss"][-1]

            pred_by_model = {"cnn": pred_cnn}
            for name, fn in FEATURES.items():
                pred_by_model[name] = fit_predict_lr(fn, X_train, y_train_norm, ~train_mask, X_test)

            # ---------- evaluation (same data, same mask, same metrics for all) ----------
            for name, pred in pred_by_model.items():
                overall, per_sp, summ, bins = evaluate(
                    pred, y_test_norm, x_pos_test, x_speed_test, test_mask, xpos_mean, xpos_std)
                perf_overall[duration][name][t, it] = overall
                perf_speed[duration][name][t, :, it] = per_sp
                for speed, metrics in summ.items():
                    summary_rows.append(dict(duration=duration, model=name, iteration=it,
                                             participant=test_participant, speed=speed,
                                             xpos_std=xpos_std, **metrics))
                for r in bins:
                    bin_rmse_rows.append(dict(duration=duration, model=name, iteration=it,
                                              participant=test_participant, **r))
                preds[duration][name].append(pred.copy())

            true_list.append(y_test_norm); speed_list.append(x_speed_test); mask_list.append(test_mask)

    preds[duration]["true_pos"]   = np.concatenate(true_list)
    preds[duration]["true_speed"] = np.concatenate(speed_list)
    preds[duration]["test_mask"]  = np.concatenate(mask_list)

summary_df = pd.DataFrame(summary_rows)
bin_rmse_df = pd.DataFrame(bin_rmse_rows)


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
# Plot RMSE per speed across cross-validations
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
mod = 'lr_gated'#'lr_eog' #'lr_gated'#'cnn'
df = summary_df[summary_df["model"] == mod]
for m in ["rmse_deg", "r2", "target_sd", "slope", "r"]:
    ch.plot_summary_metric(df, m, mod, unit="participant",
                        save_path=op.join('/Users', 'ldaumail3', 'Documents', 'research', 'eyetracking_EOG', 'Audio_Visual_EOG-main', 'analysis_plots'))

import pandas as pd
# one figure per metric
df = bin_rmse_df[bin_rmse_df["model"] == mod]
ch.plot_bin_rmse(df, mod, unit="participant",
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