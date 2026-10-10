import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label

def softmax(res):
    z = res - np.max(res, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, class_weight="none", class_weight_clip=4.0):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            class_weight (str): weighting mode for class-imbalanced training
                supported: none / weighted / sqrt_weighted / log_weighted / weighted_clipped
            class_weight_clip (float): max class weight for weighted_clipped mode
        """
        self.lr = lr
        self.max_iters = max_iters
        if class_weight == "balanced":
            class_weight = "weighted"
        self.class_weight = class_weight
        self.class_weight_clip = class_weight_clip

    def _compute_sample_weights(self, labels_int, n_classes):
        """
        Compute per-class and per-sample weights according to the selected mode.

        Arguments:
            labels_int (np.array): class labels as ints, shape (N,)
            n_classes (int): number of classes
        Returns:
            sample_weights (np.array): per-sample weights, shape (N,)
        """
        counts = np.bincount(labels_int, minlength=n_classes).astype(float)
        counts = np.maximum(counts, 1.0)
        inv_freq = labels_int.shape[0] / (n_classes * counts)

        class_weight_fns = {
            "weighted": lambda w: w,
            "sqrt_weighted": np.sqrt,
            "log_weighted": lambda w: 1.0 + np.log(w),
            "weighted_clipped": lambda w: np.minimum(w, self.class_weight_clip),
        }
        class_weights = class_weight_fns.get(self.class_weight, lambda w: np.ones_like(w))(inv_freq)

        self.class_weights_ = class_weights
        return self.class_weights_[labels_int]

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        n_classes = get_n_classes(training_labels)
        labels_int = training_labels.astype(int)
        target = label_to_onehot(labels_int)
        X = training_data
        self.w = np.zeros((X.shape[1], n_classes)) #initialize the weights
        sample_weights = self._compute_sample_weights(labels_int, n_classes)

        normalizer = np.sum(sample_weights)
        for i in range(self.max_iters):
            res = X @ self.w
            y_hat = softmax(res)
            weighted_error = (y_hat - target) * sample_weights[:, None]
            grad = X.T @ weighted_error / normalizer
            self.w -= self.lr*grad

        res = X @ self.w
        y_hat = softmax(res)
        pred_labels = onehot_to_label(y_hat)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        X = test_data
        res = X @ self.w
        y_hat = softmax(res)
        return np.argmax(y_hat, axis=1)
