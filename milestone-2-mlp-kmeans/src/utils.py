import os
import numpy as np


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


def standardize_fit(data):
    """
    Fit z-score standardization parameters on the training data.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        (tuple): per-feature mean and std arrays
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1.0
    return means, stds


def standardize_transform(data, means, stds):
    """
    Apply z-score standardization using precomputed means and stds.

    Arguments:
        data (np.array): of shape (N,D)
        means (np.array): per-feature means
        stds (np.array): per-feature standard deviations
    Returns:
        (np.array): shape (N,D)
    """
    return normalize_fn(data, means, stds)


def robust_scale_fit(data):
    """
    Fit a robust scaler using the median and interquartile range.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        (tuple): median and IQR arrays of shape (D,)
    """
    medians = np.median(data, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return medians, iqr


def robust_scale_transform(data, medians, iqr):
    """
    Apply robust scaling using precomputed median and IQR.

    Arguments:
        data (np.array): of shape (N,D)
        medians (np.array): per-feature medians
        iqr (np.array): per-feature interquartile ranges
    Returns:
        (np.array): shape (N,D)
    """
    return (data - medians) / iqr


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


def mse_fn(pred, gt):
    """
    Mean Squared Error
    """
    pred = np.asarray(pred).flatten()
    gt = np.asarray(gt).flatten()
    return np.mean((pred - gt) ** 2)


def rmse_fn(pred, gt):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse_fn(pred, gt))


def mae_fn(pred, gt):
    """
    Mean Absolute Error
    """
    pred = np.asarray(pred).flatten()
    gt = np.asarray(gt).flatten()
    return np.mean(np.abs(pred - gt))


def r2_fn(pred, gt):
    """
    Coefficient of Determination (R2 score)
    """
    pred = np.asarray(pred).flatten()
    gt = np.asarray(gt).flatten()
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def nfold_cv_mlp(X, y, mlp_model, loss, n_folds=5, is_classification=True, loss_kwargs=None):
    """
    Perform n-fold cross-validation for MLP.
    
    Arguments:
        X (np.array): features of shape (N, D)
        y (np.array): labels (class indices for classification, or values for regression)
        mlp_model (MLP): untrained MLP instance
        loss: loss function class (MSE or CrossEntropy)
        n_folds (int): number of folds (default 5)
        is_classification (bool): whether this is classification task
    
    Returns:
        dict: with keys 'mean_acc'/'mean_mse', 'std_acc'/'std_mse', and 'fold_scores'
    """
    N = X.shape[0]
    fold_size = N // n_folds
    indices = np.random.permutation(N)
    fold_scores = []
    
    for fold in range(n_folds):
        # Split data
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train MLP on this fold
        if is_classification:
            loss_obj = loss(**(loss_kwargs or {}))
            y_train_oh = label_to_onehot(y_train)
            mlp_model.fit(X_train, y_train_oh, loss=loss_obj, epochs=100, batch_size=32)
            pred_oh = mlp_model.predict(X_test)
            pred = onehot_to_label(pred_oh)
            score = accuracy_fn(pred, y_test)
        else:
            mlp_model.fit(X_train, y_train.reshape(-1, 1), loss=loss, epochs=100, batch_size=32)
            pred = mlp_model.predict(X_test).flatten()
            score = mse_fn(pred, y_test)
        
        fold_scores.append(score)
    
    fold_scores = np.array(fold_scores)
    metric_name = 'accuracy' if is_classification else 'mse'
    return {
        f'mean_{metric_name}': np.mean(fold_scores),
        f'std_{metric_name}': np.std(fold_scores),
        'fold_scores': fold_scores
    }


def grid_search_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    mlp_class,
    loss_class,
    param_grid,
    is_classification=True,
    verbose=True,
    init_method="basic",
    momentum=0.0,
    loss_kwargs=None,
    trial_plot_dir=None,
    optimizer="sgd",
    weight_decay=0.0,
    dropout_rate=0.0,
):
    """
    Grid search over MLP hyperparameters.
    
    Arguments:
        X_train, y_train: training data and labels
        X_test, y_test: test data and labels
        mlp_class: MLP class (not instance)
        loss_class: loss function class (MSE or CrossEntropy)
        param_grid (dict): dict with keys 'lr', 'epochs', 'batch_size', 'hidden_size'
            e.g. {'lr': [1e-5, 1e-4], 'epochs': [100, 200], 'batch_size': [16, 32], 'hidden_size': [32, 64]}
        is_classification (bool): task type
        verbose (bool): print progress
        init_method (str): weight initialization method
        momentum (float): momentum factor
        optimizer (str): optimizer to use (sgd, momentum, adam)
        weight_decay (float): L2 regularization strength
        dropout_rate (float): dropout rate applied during training
    
    Returns:
        dict with keys: 'results' (list of dicts), 'best_config', 'best_score', 'scores_array'
    """
    lrs = param_grid.get('lr', [1e-5])
    epochs_list = param_grid.get('epochs', [100])
    batch_sizes = param_grid.get('batch_size', [32])
    hidden_sizes = param_grid.get('hidden_size', [64])
    
    results = []
    total = len(lrs) * len(epochs_list) * len(batch_sizes) * len(hidden_sizes)
    count = 0
    
    for lr in lrs:
        for epochs in epochs_list:
            for bs in batch_sizes:
                for hidden in hidden_sizes:
                    count += 1
                    if verbose:
                        print(f"[{count}/{total}] lr={lr:.0e}, epochs={epochs}, bs={bs}, hidden={hidden}", end='')
                    
                    # Initialize MLP
                    input_dim = X_train.shape[1]
                    output_dim = len(np.unique(y_train)) if is_classification else 1
                    from src.activations import ReLU, Softmax, Identity
                    if is_classification:
                        mlp = mlp_class([input_dim, hidden, output_dim], [ReLU, Softmax], init_method=init_method)
                    else:
                        mlp = mlp_class([input_dim, hidden, output_dim], [ReLU, Identity], init_method=init_method)
                    
                    # Train
                    if is_classification:
                        loss_obj = loss_class(**(loss_kwargs or {}))
                        y_train_oh = label_to_onehot(y_train)
                        mlp.fit(
                            X_train,
                            y_train_oh,
                            loss=loss_obj,
                            epochs=epochs,
                            batch_size=bs,
                            learning_rate=lr,
                            optimizer=optimizer,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            dropout_rate=dropout_rate,
                        )
                        pred_oh = mlp.predict(X_test)
                        pred = onehot_to_label(pred_oh)
                        score = accuracy_fn(pred, y_test)
                        metric = 'accuracy'
                    else:
                        loss_obj = loss_class(**(loss_kwargs or {}))
                        mlp.fit(
                            X_train,
                            y_train.reshape(-1, 1),
                            loss=loss_obj,
                            epochs=epochs,
                            batch_size=bs,
                            learning_rate=lr,
                            optimizer=optimizer,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            dropout_rate=dropout_rate,
                        )
                        pred = mlp.predict(X_test).flatten()
                        score = mse_fn(pred, y_test)
                        metric = 'mse'

                    if is_classification and trial_plot_dir is not None:
                        trial_name = (
                            f"trial_{count:03d}"
                            f"_lr={lr:g}"
                            f"_ep={epochs}"
                            f"_bs={bs}"
                            f"_hs={hidden}"
                        )
                        save_dir = os.path.join(trial_plot_dir, trial_name)
                        os.makedirs(save_dir, exist_ok=True)
                        labels = sorted(list(np.unique(y_test)))
                        plot_confusion_matrix(y_test, pred, labels, os.path.join(save_dir, "confusion.png"), title=f"Confusion: {trial_name}")
                        plot_classification_report(y_test, pred, labels, os.path.join(save_dir, "report.png"), title=f"Per-class PRF: {trial_name}")
                    
                    results.append({
                        'lr': lr, 'epochs': epochs, 'batch_size': bs, 'hidden_size': hidden,
                        metric: score
                    })
                    if verbose:
                        print(f" → {metric}={score:.4f}")
    
    # Find best
    scores = np.array([r[metric] for r in results])
    if is_classification:
        best_idx = np.argmax(scores)
    else:
        best_idx = np.argmin(scores)
    best_config = results[best_idx]
    best_score = scores[best_idx]
    
    return {
        'results': results,
        'best_config': best_config,
        'best_score': best_score,
        'scores_array': scores.reshape(len(lrs), len(epochs_list), len(batch_sizes), len(hidden_sizes)),
        'param_names': ['lr', 'epochs', 'batch_size', 'hidden_size']
    }


# --- Feature selection helpers ---
def variance_threshold(X, threshold=0.05):
    """
    Remove features whose variance is below `threshold`.

    Returns:
        X_reduced: array with selected features
        keep_idx: indices of kept features (relative to input X)
        variances: per-feature variances
    """
    variances = X.var(axis=0)
    keep_idx = np.where(variances >= threshold)[0]
    return X[:, keep_idx], keep_idx, variances


def correlation_filter(X, threshold=0.9):
    """
    Remove one feature from any highly correlated pair (abs(corr) > threshold).
    Greedily removes the higher-indexed feature in a correlated pair.

    Returns:
        X_reduced, keep_idx, corr_matrix
    """
    corr = np.corrcoef(X, rowvar=False)
    D = corr.shape[0]
    to_remove = set()
    for i in range(D):
        for j in range(i + 1, D):
            if abs(corr[i, j]) > threshold:
                to_remove.add(j)
    keep_idx = np.array([i for i in range(D) if i not in to_remove])
    return X[:, keep_idx], keep_idx, corr


def fscore_select_k(X, y, k=10):
    """
    Univariate ANOVA-like F-score to rank features for classification.

    Returns:
        X_reduced, topk_idx, f_scores
    """
    classes = np.unique(y)
    N, D = X.shape
    overall_mean = X.mean(axis=0)
    ss_between = np.zeros(D)
    ss_within = np.zeros(D)
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        mean_c = Xc.mean(axis=0)
        ss_between += nc * (mean_c - overall_mean) ** 2
        ss_within += ((Xc - mean_c) ** 2).sum(axis=0)

    dfb = max(len(classes) - 1, 1)
    dfw = max(N - len(classes), 1)
    msb = ss_between / dfb
    msw = ss_within / dfw
    f_scores = msb / (msw + 1e-12)
    topk_idx = np.argsort(-f_scores)[:k]
    return X[:, topk_idx], topk_idx, f_scores


def polynomial_feature_expansion(X, degree=2, include_bias=False):
    """
    Simple polynomial feature expansion up to given degree.

    - degree=1: returns original features
    - degree=2: returns original features and all pairwise products (including squares)

    Arguments:
        X (np.array): input features of shape (N, D)
        degree (int): maximum polynomial degree (>=1)
        include_bias (bool): whether to include a constant 1 column

    Returns:
        np.array: expanded features of shape (N, D_expanded)
    """
    from itertools import combinations_with_replacement

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if degree < 1:
        raise ValueError("degree must be >= 1")

    N, D = X.shape
    cols = []
    # include original features (degree=1)
    for d in range(D):
        cols.append(X[:, d].reshape(N, 1))

    # higher degrees
    for deg in range(2, degree + 1):
        for comb in combinations_with_replacement(range(D), deg):
            prod = np.ones(N)
            for idx in comb:
                prod = prod * X[:, idx]
            cols.append(prod.reshape(N, 1))

    if not cols:
        expanded = np.zeros((N, 0))
    else:
        expanded = np.concatenate(cols, axis=1)

    if include_bias:
        expanded = np.concatenate([np.ones((N, 1)), expanded], axis=1)

    return expanded


def plot_regression_history(history, save_path):
    from scripts.regression_plots import plot_regression_history as _plot_regression_history

    return _plot_regression_history(history, save_path)


def plot_grid_search(gs_results, save_path, is_classification=False):
    
    """
    Plot grid search results as a beautiful heatmap showing all configurations.
    Columns: Learning Rate x Batch Size
    Rows: Hidden Layer Size x Epochs
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    results = gs_results['results']
    metric = 'accuracy' if is_classification else 'mse'
    
    # Get unique parameter values in sorted order
    lrs = sorted(list(set(r['lr'] for r in results)))
    batch_sizes = sorted(list(set(r['batch_size'] for r in results)))
    hidden_sizes = sorted(list(set(r['hidden_size'] for r in results)))
    epochs_list = sorted(list(set(r['epochs'] for r in results)))
    
    # Build 2D composite coordinate grids
    col_pairs = [(lr, bs) for lr in lrs for bs in batch_sizes]
    row_pairs = [(hs, ep) for hs in hidden_sizes for ep in epochs_list]
    
    grid = np.zeros((len(row_pairs), len(col_pairs)))
    
    for i, (hs, ep) in enumerate(row_pairs):
        for j, (lr, bs) in enumerate(col_pairs):
            matching = [r[metric] for r in results if r['lr'] == lr and r['batch_size'] == bs and r['hidden_size'] == hs and r['epochs'] == ep]
            grid[i, j] = matching[0] if matching else float('nan')
            
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle colormap and log-scale normalization
    cmap = 'viridis' if is_classification else 'viridis_r'
    
    grid_min, grid_max = np.nanmin(grid), np.nanmax(grid)
    
    if not is_classification:
        from matplotlib.colors import LogNorm
        # Ensure minimum value is strictly positive for LogNorm
        norm_min = max(grid_min, 1e-5)
        norm = LogNorm(vmin=norm_min, vmax=grid_max)
        threshold = np.exp((np.log(norm_min) + np.log(grid_max)) / 2.0)
    else:
        norm = None
        threshold = (grid_min + grid_max) / 2.0
        
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric.upper(), rotation=-90, va="bottom", fontsize=11)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(col_pairs)))
    ax.set_yticks(np.arange(len(row_pairs)))
    
    col_labels = [f"lr={lr:.0e}\nbs={bs}" for lr, bs in col_pairs]
    row_labels = [f"hs={hs}\nep={ep}" for hs, ep in row_pairs]
    
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=9)
    
    ax.set_xlabel('Optimization (Learning Rate & Batch Size)', fontsize=11, labelpad=10)
    ax.set_ylabel('Architecture & Training (Hidden Size & Epochs)', fontsize=11, labelpad=10)
    
    # Annotate values inside the cells with smart contrast
    for i in range(len(row_pairs)):
        for j in range(len(col_pairs)):
            val = grid[i, j]
            if np.isnan(val):
                continue
            
            # Determine contrasting text color based on log or linear thresholds
            if is_classification:
                text_color = "black" if val > threshold else "white"
            else:
                text_color = "white" if val > threshold else "black"
                
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=8)
            
    # Highlight the best configuration with a red rectangle
    best_config = gs_results.get('best_config', {})
    if best_config:
        best_lr = best_config.get('lr')
        best_bs = best_config.get('batch_size')
        best_hs = best_config.get('hidden_size')
        best_ep = best_config.get('epochs')
        
        best_col_idx = None
        best_row_idx = None
        for idx, (lr, bs) in enumerate(col_pairs):
            if lr == best_lr and bs == best_bs:
                best_col_idx = idx
                break
        for idx, (hs, ep) in enumerate(row_pairs):
            if hs == best_hs and ep == best_ep:
                best_row_idx = idx
                break
                
        if best_col_idx is not None and best_row_idx is not None:
            import matplotlib.patches as patches
            # Draw red rectangle around the cell (center is at (col, row), boundaries are col-0.5 to col+0.5)
            rect = patches.Rectangle((best_col_idx - 0.5, best_row_idx - 0.5), 1.0, 1.0, linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
    title_suffix = "Validation Accuracy (Higher is Better)" if is_classification else "Validation MSE (Lower is Better)"
    ax.set_title(f"Grid Search Hyperparameter Tuning Heatmap (All 36 Configs)\n{title_suffix}", fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved grid search heatmap to {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path, title=None):
    """Plot and save a confusion matrix heatmap.

    y_true, y_pred: 1D arrays of labels
    labels: list or array of label names/ids
    save_path: path to save PNG
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # compute confusion matrix without sklearn
    labels = list(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title or 'Confusion Matrix')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # annotate counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_classification_report(y_true, y_pred, labels, save_path, title=None):
    """Save a small bar plot of per-class precision/recall/f1."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # compute precision, recall, f1 without sklearn
    labels = list(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    tp = np.zeros(len(labels), dtype=int)
    fp = np.zeros(len(labels), dtype=int)
    fn = np.zeros(len(labels), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            if t == p:
                tp[label_to_idx[t]] += 1
            else:
                fp[label_to_idx[p]] += 1
                fn[label_to_idx[t]] += 1
    p = np.zeros(len(labels), dtype=float)
    r = np.zeros(len(labels), dtype=float)
    f1 = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        p[i], r[i], f1[i] = prec, rec, f

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, p, width, label='Precision', color='#4E79A7')
    ax.bar(x, r, width, label='Recall', color='#F28E2B')
    ax.bar(x + width, f1, width, label='F1', color='#E15759')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Class')
    ax.set_ylim(0, 1.0)
    ax.set_title(title or 'Per-class Precision/Recall/F1')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved classification report to {save_path}")
