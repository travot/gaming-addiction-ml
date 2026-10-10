import numpy as np
import json
import copy


# General utilities
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (np.array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (np.array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels


def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (np.array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (np.array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)


def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        (np.array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]), data], axis=1)
    return data


def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.

    Arguments:
        data (np.array): of shape (N,D)
        means (np.array): of shape (1,D)
        stds (np.array): of shape (1,D)
    Returns:
        (np.array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds


def normalize_and_append_bias(train_data, *other_arrays):
    """
    Normalize arrays using train_data statistics, then append bias term.

    Arguments:
        train_data (np.array): reference training array of shape (N,D)
        *other_arrays (np.array): any number of arrays of shape (_,D)
    Returns:
        tuple: transformed arrays in the same order as input
    """
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std = np.where(std == 0, 1.0, std)

    outputs = [append_bias_term(normalize_fn(train_data, mean, std))]
    for arr in other_arrays:
        outputs.append(append_bias_term(normalize_fn(arr, mean, std)))
    return tuple(outputs)


def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.

    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.


def macrof1_fn(pred_labels, gt_labels):
    """
    Return the macro F1-score.

    Arguments:
        pred_labels (np.array):
        gt_labels (np.array):
    Returns:
        float: macro F1-score
    """
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels == val)

        tp = sum(predpos * gtpos)
        fp = sum(predpos * ~gtpos)
        fn = sum(~predpos * gtpos)
        if tp == 0:
            continue
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        macrof1 += 2 * (precision * recall) / (precision + recall)

    return macrof1 / len(class_ids)


def per_class_recall_fn(pred_labels, gt_labels):
    """
    Return recall for each class present in gt_labels.

    Arguments:
        pred_labels (np.array): predicted labels
        gt_labels (np.array): ground-truth labels
    Returns:
        dict: mapping class_id -> recall
    """
    class_ids = np.unique(gt_labels)
    recalls = {}
    for class_id in class_ids:
        gtpos = (gt_labels == class_id)
        tp = np.sum((pred_labels == class_id) * gtpos)
        fn = np.sum((pred_labels != class_id) * gtpos)
        denom = tp + fn
        recalls[int(class_id)] = 0.0 if denom == 0 else float(tp / denom)
    return recalls


def mse_fn(pred, gt):
    """
    Mean Squared Error

    Arguments:
        pred: Nx1 (or N,) prediction array
        gt: Nx1 (or N,) groundtruth values for each prediction
    Returns:
        returns the computed MSE loss
    """
    loss = (pred - gt) ** 2
    loss = np.mean(loss)
    return loss


def load_config(config_path):
    """
    Load configuration from a JSON file.

    Arguments:
        config_path (str): path to the configuration file
    Returns:
        dict: configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def run_cross_validation(model_obj, features, labels, cv_folds, task, seed=100, verbose=True):
    """
    Run K-fold cross-validation on an unfitted model object and print aggregated metrics.
    """
    n_samples = len(features)
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
    if cv_folds > n_samples:
        raise ValueError("cv_folds cannot be greater than the number of training samples")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, cv_folds)

    fold_scores_1 = []
    fold_scores_2 = []
    for fold_id in range(cv_folds):
        val_idx = folds[fold_id]
        train_idx = np.concatenate([folds[i] for i in range(cv_folds) if i != fold_id])

        x_train, y_train = features[train_idx], labels[train_idx]
        x_val, y_val = features[val_idx], labels[val_idx]

        x_train, x_val = normalize_and_append_bias(x_train, x_val)

        fold_model = copy.deepcopy(model_obj)
        fold_model.fit(x_train, y_train)
        val_preds = fold_model.predict(x_val)

        if task == "classification":
            fold_scores_1.append(accuracy_fn(val_preds, y_val))
            fold_scores_2.append(macrof1_fn(val_preds, y_val))
        else:
            fold_scores_1.append(mse_fn(val_preds, y_val))

    if task == "classification":
        acc_mean, acc_std = np.mean(fold_scores_1), np.std(fold_scores_1)
        f1_mean, f1_std = np.mean(fold_scores_2), np.std(fold_scores_2)
        if verbose:
            print("Cross-validation "
                f"({cv_folds}-fold): accuracy = {acc_mean:.3f}% +/- {acc_std:.3f}, "
                f"F1 = {f1_mean:.6f} +/- {f1_std:.6f}")
        return acc_mean, f1_mean
    mse_mean, mse_std = np.mean(fold_scores_1), np.std(fold_scores_1)
    if verbose:
        print("Cross-validation "
            f"({cv_folds}-fold): MSE = {mse_mean:.6f} +/- {mse_std:.6f}")
    return mse_mean
        