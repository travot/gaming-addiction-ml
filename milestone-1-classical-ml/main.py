import argparse
import time
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_and_append_bias, accuracy_fn, macrof1_fn, mse_fn, load_config, run_cross_validation
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
    train_features_cv = np.array(train_features)

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.


    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test and not args.cross_validate:
        val_size = int(0.2 * len(train_features))
        indices = np.random.permutation(len(train_features))

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        val_features = train_features[val_idx]
        val_labels_reg = train_labels_reg[val_idx]
        val_labels_classif = train_labels_classif[val_idx]

        train_features = train_features[train_idx]
        train_labels_reg = train_labels_reg[train_idx]
        train_labels_classif = train_labels_classif[train_idx]

    if not args.test and not args.cross_validate:
        train_features, val_features, test_features = normalize_and_append_bias(train_features, val_features, test_features)
    else:
        train_features, test_features = normalize_and_append_bias(train_features, test_features)

    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K, task_kind=args.task, dist=args.dist, P=args.P, alpha=args.knn_alpha)

    elif args.method == "logistic_regression":
        lr_value = args.lr[0] if isinstance(args.lr, list) else args.lr
        it_value = args.max_iters[0] if isinstance(args.max_iters, list) else args.max_iters
        cw_value = args.class_weight[0] if isinstance(args.class_weight, list) else args.class_weight
        method_obj = LogisticRegression(lr=lr_value, max_iters=it_value, class_weight=cw_value, class_weight_clip=args.class_weight_clip)

    elif args.method == "linear_regression":
        method_obj = LinearRegression()

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"

        if args.method == "logistic_regression" and args.cross_validate:
            lr_grid = args.lr if isinstance(args.lr, list) else [args.lr]
            it_grid = args.max_iters if isinstance(args.max_iters, list) else [args.max_iters]
            cw_grid = args.class_weight if isinstance(args.class_weight, list) else [args.class_weight]
            if len(lr_grid) > 1 or len(it_grid) > 1 or len(cw_grid) > 1:
                best_f1 = -np.inf
                best_acc = -np.inf
                best_lr, best_iters, best_cw = lr_grid[0], it_grid[0], cw_grid[0]
                for lr in lr_grid:
                    for iters in it_grid:
                        for cw in cw_grid:
                            candidate = LogisticRegression(lr=lr, max_iters=iters, class_weight=cw, class_weight_clip=args.class_weight_clip)
                            cv_acc, cv_f1 = run_cross_validation(candidate, train_features_cv, train_labels_classif, args.cv_folds, "classification", verbose=False)
                            if (cv_f1 > best_f1) or (np.isclose(cv_f1, best_f1) and cv_acc > best_acc):
                                best_f1 = cv_f1
                                best_acc = cv_acc
                                best_lr, best_iters, best_cw = lr, iters, cw
                print(f"Selected logistic params from CV grid: lr={best_lr}, max_iters={best_iters}, class_weight={best_cw}, F1={best_f1:.6f}, accuracy={best_acc:.3f}%")
                method_obj = LogisticRegression(lr=best_lr, max_iters=best_iters, class_weight=best_cw, class_weight_clip=args.class_weight_clip)

        if args.cross_validate:
            run_cross_validation(method_obj, train_features_cv, train_labels_classif, args.cv_folds, "classification")

        # Fit the method on training data
        start_train = time.time()
        preds_train = method_obj.fit(train_features, train_labels_classif)
        train_time = time.time() - start_train

        start_train_pred = time.time()
        preds_train_eval = method_obj.predict(train_features)
        train_pred_time = time.time() - start_train_pred

        if not args.test and not args.cross_validate:
            start_val = time.time()
            val_preds = method_obj.predict(val_features)
            val_time = time.time() - start_val

            val_acc = accuracy_fn(val_preds, val_labels_classif)
            val_f1 = macrof1_fn(val_preds, val_labels_classif)
            print(f"Validation set: accuracy = {val_acc:.3f}% - F1 = {val_f1:.6f}")
            print(f"Validation time: {val_time:.6f}s")

        # Predict on unseen data
        start_test = time.time()
        preds = method_obj.predict(test_features)
        test_time = time.time() - start_test

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        print(f"Training time: {train_time:.6f}s")
        print(f"Train prediction time: {train_pred_time:.6f}s")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        print(f"Test prediction time: {test_time:.6f}s")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"

        if args.cross_validate:
            run_cross_validation(method_obj, train_features_cv, train_labels_reg, args.cv_folds, "regression")

        # Fit the method on training data
        start_train = time.time()
        preds_train = method_obj.fit(train_features, train_labels_reg)
        train_time = time.time() - start_train

        start_train_pred = time.time()
        preds_train_eval = method_obj.predict(train_features)
        train_pred_time = time.time() - start_train_pred

        if not args.test and not args.cross_validate:
            start_val = time.time()
            val_preds = method_obj.predict(val_features)
            val_time = time.time() - start_val

            val_mse = mse_fn(val_preds, val_labels_reg)
            print(f"Validation set: MSE = {val_mse:.6f}")
            print(f"Validation time: {val_time:.6f}s")

        # Predict on unseen data
        start_test = time.time()
        preds = method_obj.predict(test_features)
        test_time = time.time() - start_test

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")
        print(f"Training time: {train_time:.6f}s")
        print(f"Train prediction time: {train_pred_time:.6f}s")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")
        print(f"Test prediction time: {test_time:.6f}s")


    else:
        raise ValueError(f"Unknown task: {args.task}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="path to a JSON config file")
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=19,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="weighted_clipped",
        choices=["none", "weighted", "balanced", "sqrt_weighted", "log_weighted", "weighted_clipped"],
        help="class reweighting for logistic regression",
    )
    parser.add_argument(
        "--class_weight_clip",
        type=float,
        default=4.0,
        help="max class weight used by weighted_clipped mode",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="euclidean",
        help="the distance function for KNN method",
    )
    parser.add_argument(
        "--P",
        type=float,
        default=1.5,
        help="distance-weight exponent for KNN classification",
    )
    parser.add_argument(
        "--knn_alpha",
        type=float,
        default=0.525,
        help="class-weight exponent for KNN classification",
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="run K-fold cross-validation on training data before final train/test evaluation",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="number of folds to use when --cross_validate is enabled",
    )

    # Feel free to add more arguments here if you need!

    cfg, _ = parser.parse_known_args()
    if cfg.config: parser.set_defaults(**load_config(cfg.config))
    args = parser.parse_args()
    main(args)
