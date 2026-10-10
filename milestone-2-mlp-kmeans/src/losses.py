import numpy as np


class MSE:
    @staticmethod
    def loss(y_true, y_pred, sample_weights=None):
        """
        :param y_true: (array) Ground-truth target values.
        :param y_pred: (array) Predicted values from the model.
        :param sample_weights: (optional array) Per-sample weights to apply to the loss.
        :return: (float) Mean squared error (optionally weighted).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
        if y_true.size == 0:
            raise ValueError("y_true and y_pred must not be empty")

        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        if sample_weights is None:
            return float(np.mean((y_pred_flat - y_true_flat) ** 2))
        w = np.asarray(sample_weights).reshape(-1)
        if w.shape != y_true_flat.shape:
            raise ValueError('sample_weights must have same length as y_true')
        return float(np.sum(w * (y_pred_flat - y_true_flat) ** 2) / np.sum(w))

    @staticmethod
    def gradient(y_true, y_pred, sample_weights=None):
        """
        Compute the gradient of MSE with respect to predictions.

        Returns an array shaped like `y_true` containing dLoss/dy_pred.
        Supports optional `sample_weights` which weight per-sample contributions.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        orig_shape = y_true.shape
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        n = y_true_flat.size
        if sample_weights is None:
            grad_flat = 2.0 * (y_pred_flat - y_true_flat) / n
        else:
            w = np.asarray(sample_weights).reshape(-1)
            if w.shape != y_true_flat.shape:
                raise ValueError('sample_weights must have same length as y_true')
            grad_flat = 2.0 * (y_pred_flat - y_true_flat) * w / np.sum(w)
        return grad_flat.reshape(orig_shape)


class CrossEntropy:
    """Multiclass cross-entropy with optional per-class weighting.

    Supports one-hot labels and softmax outputs. Pass a `class_weights`
    mapping (dict) class_id -> weight to emphasize specific classes.
    """

    def __init__(self, class_weights=None):
        self.class_weights = class_weights

    def _sample_weights(self, y_true):
        if self.class_weights is None:
            return np.ones(y_true.shape[0], dtype=float)
        class_ids = np.argmax(y_true, axis=1)
        return np.asarray([self.class_weights.get(int(c), 1.0) for c in class_ids], dtype=float)

    def loss(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.clip(np.asarray(y_pred), 1e-7, 1 - 1e-7)
        sample_weights = self._sample_weights(y_true)
        per_sample_loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(sample_weights * per_sample_loss))

    def gradient(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sample_weights = self._sample_weights(y_true).reshape(-1, 1)
        return (y_pred - y_true) * sample_weights / y_true.shape[0]


class MAE:
    @staticmethod
    def loss(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def gradient(y_true, y_pred):
        y_true = np.asarray(y_true)
        orig_shape = y_true.shape
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = np.asarray(y_pred).reshape(-1)
        if y_true_flat.shape != y_pred_flat.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        n = y_true_flat.size
        grad_flat = np.sign(y_pred_flat - y_true_flat) / n
        return grad_flat.reshape(orig_shape)


class Huber:
    """Huber loss for regression with parameter `delta`."""

    @staticmethod
    def loss(y_true, y_pred, sample_weights=None, delta=1.0):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        r = y_pred - y_true
        abs_r = np.abs(r)
        mask = abs_r <= delta
        loss_vals = np.where(mask, 0.5 * (r ** 2), delta * (abs_r - 0.5 * delta))
        return float(np.mean(loss_vals))

    @staticmethod
    def gradient(y_true, y_pred, sample_weights=None, delta=1.0):
        y_true = np.asarray(y_true)
        orig_shape = y_true.shape
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = np.asarray(y_pred).reshape(-1)
        if y_true_flat.shape != y_pred_flat.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        n = y_true_flat.size
        r = y_pred_flat - y_true_flat
        abs_r = np.abs(r)
        grad_flat = np.where(abs_r <= delta, r, delta * np.sign(r))
        return (grad_flat / n).reshape(orig_shape)
