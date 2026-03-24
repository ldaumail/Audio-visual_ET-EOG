
import numpy as np
from torch.utils.data import DataLoader
import torch


# def get_network_performance(model, test_pats, test_targets):
#     # this presents the test patterns and targets to see how well the trained network generalizes
#     # to the test patterns (which it did not see during training)
#     # this just gives you overall performance
#     eval_loss, eval_perf = model.evaluate(test_pats, test_targets)
#     # returns just the overall performance
#     return eval_perf


def get_network_output(model, dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    
    model.eval()
    
    all_dir = []
    all_speed = []
    all_targets_dir = []
    all_targets_speed = []

    with torch.no_grad():
        for X, mask, y_dir, y_speed in loader:
            pred_dir, pred_speed = model(X)

            all_dir.append(pred_dir.cpu())
            all_speed.append(pred_speed.cpu())
            all_targets_dir.append(y_dir.cpu())
            all_targets_speed.append(y_speed.cpu())

    pred_dir = torch.cat(all_dir).numpy()
    pred_speed = torch.cat(all_speed).numpy()
    y_dir = torch.cat(all_targets_dir).numpy()
    y_speed = torch.cat(all_targets_speed).numpy()

    return pred_dir, pred_speed, y_dir, y_speed

def get_direction_performance(pred_dir, y_dir):
    pred_labels = np.argmax(pred_dir, axis=1)
    acc = np.mean(pred_labels == y_dir)
    return acc

def get_speed_performance(pred_speed, y_speed):
    rmse = np.sqrt(np.mean((pred_speed.squeeze() - y_speed)**2))
    return rmse

def get_network_cat_performance(pred_out, targets, n_classes):
    pred_labels = np.argmax(pred_out, axis=1)

    cat_perf = np.zeros(n_classes)

    for i in range(n_classes):
        idx = targets == i

        if np.sum(idx) == 0:
            continue

        correct = np.sum(pred_labels[idx] == i)
        cat_perf[i] = correct / np.sum(idx)

    return cat_perf