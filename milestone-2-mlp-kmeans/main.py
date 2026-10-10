import argparse
import json
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU, Identity, Softmax
from src.methods.kmeans import KMeans
from src.utils import (
    robust_scale_fit,
    robust_scale_transform,
    standardize_fit,
    standardize_transform,
    append_bias_term,
    accuracy_fn,
    macrof1_fn,
    mse_fn,
    rmse_fn,
    mae_fn,
    r2_fn,
    label_to_onehot,
    onehot_to_label,
    nfold_cv_mlp,
    grid_search_mlp,
    variance_threshold,
    correlation_filter,
    fscore_select_k,
    plot_regression_history,
    plot_grid_search,
)
from src.utils import plot_confusion_matrix, plot_classification_report
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    # Reshape regression targets to (N, 1) to match output layer shape
    train_labels_reg = train_labels_reg.reshape(-1, 1)
    test_labels_reg = test_labels_reg.reshape(-1, 1)

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  robust-scale, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        val_size = int(0.2 * len(train_features))
        indices = np.random.permutation(len(train_features))
        val_features = train_features[indices[:val_size]]
        train_features = train_features[indices[val_size:]]
        val_labels_clf, train_labels_classif = train_labels_classif[indices[:val_size]], train_labels_classif[indices[val_size:]]
        val_labels_reg, train_labels_reg = train_labels_reg[indices[:val_size]], train_labels_reg[indices[val_size:]]

    # Optional polynomial feature expansion (applied before scaling/selection)
    if args.feature_expansion and args.method == "mlp":
        from src.utils import polynomial_feature_expansion
        print(f"Applying polynomial feature expansion (degree={args.exp_degree}) before selection")
        train_features = polynomial_feature_expansion(train_features, degree=args.exp_degree)
        test_features = polynomial_feature_expansion(test_features, degree=args.exp_degree)
        if not args.test:
            val_features = polynomial_feature_expansion(val_features, degree=args.exp_degree)

    # Feature scaling choice
    if args.preprocessing == "standardization":
        means, stds = standardize_fit(train_features)
        train_features = standardize_transform(train_features, means, stds)
        test_features = standardize_transform(test_features, means, stds)
        if not args.test:
            val_features = standardize_transform(val_features, means, stds)
        print("Using standardization (z-score) preprocessing")
    else:
        medians, iqr = robust_scale_fit(train_features)
        train_features = robust_scale_transform(train_features, medians, iqr)
        test_features = robust_scale_transform(test_features, medians, iqr)
        if not args.test:
            val_features = robust_scale_transform(val_features, medians, iqr)
        print("Using robust scaling preprocessing")

    ### WRITE YOUR CODE HERE to do any other data processing
    # Feature selection pipeline: variance -> correlation -> ANOVA F-statistic
    if args.task == 'classification' and args.feature_selection:
        print("Applying feature selection: variance -> correlation -> ANOVA F-statistic")

        # Variance threshold (remove near-constant features)
        train_features, keep_var_idx, _ = variance_threshold(train_features, threshold=args.fs_variance_threshold)
        test_features = test_features[:, keep_var_idx]
        print(f"  After variance threshold ({args.fs_variance_threshold}): {train_features.shape[1]} features kept")

        # Correlation filter (remove redundant features)
        train_features, keep_corr_idx, _ = correlation_filter(train_features, threshold=args.fs_correlation_threshold)
        test_features = test_features[:, keep_corr_idx]
        print(f"  After correlation filter ({args.fs_correlation_threshold}): {train_features.shape[1]} features kept")

        # ANOVA F-statistic select top-k features (univariate)
        fs_k = args.fs_top_k if args.fs_top_k > 0 else train_features.shape[1]
        k = min(fs_k, train_features.shape[1])
        train_features, keep_k_idx, _ = fscore_select_k(train_features, train_labels_classif, k=k)
        test_features = test_features[:, keep_k_idx]
        print(f"  After ANOVA F-statistic select top {k}: {train_features.shape[1]} features kept")
    elif args.task == 'classification':
        print("Feature selection disabled (--no-feature-selection or config)")


    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        if args.task == "classification":
            n_out = len(np.unique(train_labels_classif))
            activations = [ReLU, Softmax]
        else:
            n_out = 1
            activations = [ReLU, Identity]
        # build hidden layer sizes from args.hidden_sizes (comma-separated)
        try:
            hidden_list = [int(h) for h in args.hidden_sizes.split(',')] if isinstance(args.hidden_sizes, str) else [int(args.hidden_sizes)]
        except Exception:
            hidden_list = [64]
        dims = [train_features.shape[1]] + hidden_list + [n_out]
        method_obj = MLP(dims, activations, init_method=args.init_method, l2_lambda=args.l2, dropout_rate=args.dropout_rate)
    
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    # Grid search if requested
    run_dir = args.init_method if args.optimizer == "sgd" else f"{args.init_method}_{args.optimizer}"
    if args.grid_search and args.method == "mlp":
        print("\n=== GRID SEARCH ===")
        param_grid = args.param_grid or {'lr': [1e-5, 1e-4, 1e-3], 'epochs': [100, 200], 'batch_size': [16, 32], 'hidden_size': [32, 64, 128]}
        class_weights = None
        if args.task == "classification":
            class_counts = np.bincount(train_labels_classif.astype(int))
            class_weights = {i: len(train_labels_classif) / (len(class_counts) * count) for i, count in enumerate(class_counts) if count > 0}
        # Use CrossEntropy for classification and MSE for regression
        loss_cls = CrossEntropy if args.task == 'classification' else MSE
        loss_kwargs = {'class_weights': class_weights} if (loss_cls is CrossEntropy and class_weights is not None) else None

        gs_results = grid_search_mlp(
            train_features,
            train_labels_classif if args.task == "classification" else train_labels_reg,
            test_features,
            test_labels_classif if args.task == "classification" else test_labels_reg,
            MLP,
            loss_cls,
            param_grid,
            is_classification=(args.task == "classification"),
            verbose=True,
            init_method=args.init_method,
            momentum=args.momentum,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            dropout_rate=args.dropout_rate,
            loss_kwargs=loss_kwargs,
            trial_plot_dir=(f"plots/{args.task}/{run_dir}/grid_search_trials"
                            if args.task == "classification" and args.save_grid_search_plots else None),
        )
        print(f"\n Best config: {gs_results['best_config']}")
        print(f"  Best score: {gs_results['best_score']:.4f}")
        
        # Save the grid-search summary only when explicitly requested.
        if args.save_grid_search_plots:
            gs_plot_path = f"plots/{args.task}/{run_dir}/grid_search.png"
            plot_grid_search(gs_results, gs_plot_path, is_classification=(args.task == "classification"))
        if args.use_best_config:
            args.lr = gs_results['best_config']['lr']
            args.max_iters = gs_results['best_config']['epochs']
            best_hidden = gs_results['best_config']['hidden_size']
            args.batch_size = gs_results['best_config'].get('batch_size', 32)
            method_obj = MLP([train_features.shape[1], best_hidden, n_out], activations, init_method=args.init_method, l2_lambda=args.l2, dropout_rate=args.dropout_rate)
            # update args.hidden_sizes so saved plot filenames reflect the actual trained architecture
            args.hidden_sizes = str(best_hidden)
            print(f"\nUsing best config for final training: lr={args.lr}, epochs={args.max_iters}, hidden_size={best_hidden}...\n")
        else:
            print("(Use --use_best_config to train with best config)\n")
            return
 
    if args.task == "classification":
        # Classification flow: support kmeans branch and MLP with CV, oversampling, and class-weights
        if args.method == "kmeans":
            if not args.test:
                all_train_features = np.concatenate([train_features, val_features], axis=0)
                all_train_labels = np.concatenate([train_labels_classif, val_labels_clf], axis=0)
            else:
                all_train_features = train_features
                all_train_labels = train_labels_classif
            print(f"\nFitting KMeans with K={args.K}...")
            train_preds = method_obj.fit(all_train_features, all_train_labels)
            preds = method_obj.predict(test_features)
            print(f"Train Acc : {accuracy_fn(train_preds, all_train_labels):.2f}%")
            print(f"Test  Acc : {accuracy_fn(preds, test_labels_classif):.2f}% | F1: {macrof1_fn(preds, test_labels_classif):.4f}")
        else:
            if args.cv and args.method == "mlp" and not args.test:
                class_counts = np.bincount(train_labels_classif.astype(int))
                class_weights = {i: len(train_labels_classif) / (len(class_counts) * count) for i, count in enumerate(class_counts) if count > 0}
                loss_cls = CrossEntropy
                loss_kwargs = {'class_weights': class_weights}
                cv_results = nfold_cv_mlp(train_features, train_labels_classif, method_obj, loss_cls, n_folds=args.cv_folds, is_classification=True,
                                          loss_kwargs=loss_kwargs)
                print(f"CV Acc: {cv_results['mean_accuracy']:.2f}% +/- {cv_results['std_accuracy']:.2f}%")

            # Optionally oversample minority classes by simple upsampling
            if args.oversample and not args.test:
                print("Applying simple oversampling to balance classes in training set")
                classes, counts = np.unique(train_labels_classif, return_counts=True)
                max_count = counts.max()
                indices_per_class = {c: np.where(train_labels_classif == c)[0] for c in classes}
                new_X = []
                new_y = []
                for c in classes:
                    idx = indices_per_class[c]
                    needed = max_count - len(idx)
                    if needed <= 0:
                        samples_idx = idx
                    else:
                        extra = np.random.choice(idx, size=needed, replace=True)
                        samples_idx = np.concatenate([idx, extra])
                    new_X.append(train_features[samples_idx])
                    new_y.append(train_labels_classif[samples_idx])
                train_features = np.vstack(new_X)
                train_labels_classif = np.concatenate(new_y)

            y_train = label_to_onehot(train_labels_classif)
            class_counts = np.bincount(train_labels_classif.astype(int))
            class_weights = {i: len(train_labels_classif) / (len(class_counts) * count) for i, count in enumerate(class_counts) if count > 0}
            loss_obj = CrossEntropy(class_weights=class_weights)
            method_obj.fit(
                train_features,
                y_train,
                loss_obj,
                epochs=args.max_iters,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                optimizer=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                dropout_rate=args.dropout_rate,
            )
            preds = onehot_to_label(method_obj.predict(test_features))
            acc = accuracy_fn(preds, test_labels_classif)
            f1 = macrof1_fn(preds, test_labels_classif)
            print(f"Test Acc: {acc:.2f}% | F1: {f1:.4f}")

            # Save classification plots in a compact per-run folder.
            run_tag = (
                f"lr={args.lr:g}"
                f"_ep={args.max_iters}"
                f"_bs={args.batch_size}"
                f"_hs={str(args.hidden_sizes).replace(',', 'x')}"
                f"_l2={args.l2:g}"
                f"_do={args.dropout_rate:g}"
                f"_mom={args.momentum:g}"
            )
            plot_dir = os.path.join("plots", "classification", run_dir, run_tag)
            os.makedirs(plot_dir, exist_ok=True)
            cm_path = os.path.join(plot_dir, "confusion.png")
            report_path = os.path.join(plot_dir, "report.png")
            labels = sorted(list(np.unique(test_labels_classif)))
            plot_confusion_matrix(test_labels_classif, preds, labels, cm_path, title=f"Confusion: {run_tag}")
            plot_classification_report(test_labels_classif, preds, labels, report_path, title=f"Per-class PRF: {run_tag}")
    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"
        x_val = val_features if not args.test else None
        y_val = val_labels_reg if not args.test else None
        
        print("Training MLP regression model...")
        history = method_obj.fit(
            train_features, train_labels_reg, MSE(),
            epochs=args.max_iters, batch_size=32, learning_rate=args.lr,
            x_val=x_val, y_val=y_val, optimizer=args.optimizer, momentum=args.momentum,
            weight_decay=args.weight_decay, dropout_rate=args.dropout_rate
        )
        
        # Plot and save history
        plot_path = f"plots/{args.task}/{run_dir}/loss_curve.png"
        plot_regression_history(history, plot_path)
        
        # Evaluate on test set
        preds = method_obj.predict(test_features).flatten()
        mse = mse_fn(preds, test_labels_reg)
        rmse = rmse_fn(preds, test_labels_reg)
        mae = mae_fn(preds, test_labels_reg)
        r2 = r2_fn(preds, test_labels_reg)
        
        print("\nFinal Test Metrics (Baseline):")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}\n")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Load config
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
    
    parser.add_argument(
        "--task",
        default=config.get('task', 'classification'),
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default=config.get('method', 'mlp'),
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default=config.get('data_path', 'data/features.npz'),
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.get('training', {}).get('lr', 1e-5),
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=config.get('training', {}).get('epochs', 100),
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        default=config.get('cv', {}).get('enabled', False),
        help="run n-fold cross-validation for MLP",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=config.get('cv', {}).get('n_folds', 5),
        help="number of folds for cross-validation",
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        default=config.get('grid_search', {}).get('enabled', False),
        help="run grid search to optimize MLP hyperparameters",
    )
    parser.add_argument(
        "--save_grid_search_plots",
        action=argparse.BooleanOptionalAction,
        default=config.get('grid_search', {}).get('save_plots', False),
        help="save per-trial grid-search plots and the summary heatmap",
    )
    parser.add_argument(
        "--use_best_config",
        action="store_true",
        default=config.get('grid_search', {}).get('use_best_config', False),
        help="after grid search, train with best config",
    )

    parser.add_argument(
        "--init_method",
        type=str,
        default=config.get('training', {}).get('init_method', 'basic'),
        choices=["basic", "he", "xavier"],
        help="weight initialization method for MLP",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="momentum coefficient for SGD (beta)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.get('training', {}).get('weight_decay', config.get('training', {}).get('l2', 0.0)),
        help="L2 regularization strength",
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.get('training', {}).get('optimizer', 'sgd'),
        choices=["sgd", "momentum", "adam"],
        help="optimizer for MLP training",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default=config.get('preprocessing', 'standardization'),
        choices=["standardization", "robust"],
        help="feature scaling to apply before training",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get('training', {}).get('batch_size', 32),
        help="mini-batch size for training",
    )

    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=str(config.get('training', {}).get('hidden_sizes', config.get('training', {}).get('hidden_size', '64'))),
        help="comma-separated hidden layer sizes for MLP (e.g. '64' or '128,64')",
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=config.get('training', {}).get('dropout_rate', 0.0),
        help="dropout rate to apply to hidden layers (0.0 disables)",
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=config.get('training', {}).get('l2', 0.0),
        help="L2 weight decay coefficient",
    )


    parser.add_argument(
        "--oversample",
        action="store_true",
        default=config.get('oversample', False),
        help="apply simple oversampling to balance classes in training set",
    )

    fs_cfg = config.get('feature_selection', {})
    parser.add_argument(
        "--feature-selection",
        action=argparse.BooleanOptionalAction,
        default=fs_cfg.get('enabled', True),
        help="enable/disable feature selection pipeline for classification",
    )
    parser.add_argument(
        "--fs-variance-threshold",
        type=float,
        default=fs_cfg.get('variance_threshold', 0.05),
        help="variance threshold for feature filtering",
    )
    parser.add_argument(
        "--fs-correlation-threshold",
        type=float,
        default=fs_cfg.get('correlation_threshold', 0.9),
        help="absolute correlation threshold for redundancy filtering",
    )
    parser.add_argument(
        "--fs-top-k",
        type=int,
        default=fs_cfg.get('top_k', 10),
        help="number of top ANOVA F-statistic features to keep (<=0 keeps all)",
    )

    # Feature expansion options
    parser.add_argument(
        "--feature-expansion",
        action=argparse.BooleanOptionalAction,
        default=config.get('feature_expansion', False),
        help="enable/disable polynomial feature expansion",
    )
    parser.add_argument(
        "--exp-degree",
        type=int,
        default=config.get('exp_degree', 2),
        help="degree for polynomial feature expansion (>=1)",
    )

    args = parser.parse_args()
    
    # Use param_grid from config if available
    if hasattr(args, 'grid_search') and args.grid_search and 'grid_search' in config:
        args.param_grid = config['grid_search'].get('param_grid', {})
    else:
        args.param_grid = None
    
    main(args)
