import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        self.weights = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Closed-form solution (normal equation) as seen in the course:
            w* = (X^T X)^{-1} X^T y

        Arguments:
            training_data (np.array): shape (N, D)
            training_labels (np.array): shape (N,)
        Returns:
            pred_labels (np.array): shape (N,)
        """

        training_labels = training_labels.reshape(-1)

        X = training_data

        XtX = X.T @ X
        XtY = X.T @ training_labels

        self.weights = np.linalg.solve(XtX, XtY)

        pred_labels = X @ self.weights
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): shape (N, D)
        Returns:
            pred_labels (np.array): shape (N,)
        """
        if self.weights is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        pred_labels = test_data @ self.weights
        return pred_labels