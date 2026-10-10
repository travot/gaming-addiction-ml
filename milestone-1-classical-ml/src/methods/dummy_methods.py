import numpy as np


class DummyClassifier(object):
    """
    This method is a dummy classifier that always predicts the most frequent
    class in the training set. It serves as a baseline and example of how
    to structure your method classes.
    """

    def __init__(self, arg1=None, arg2=None):
        """
        Initialize the object. You can add any number of arguments here.
        """
        self.arg1 = arg1
        self.arg2 = arg2

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # Find the most frequent class
        unique, counts = np.unique(training_labels, return_counts=True)
        self.most_frequent = unique[np.argmax(counts)]

        # Predict that class for all training samples
        pred_labels = np.full(training_labels.shape, self.most_frequent)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        pred_labels = np.full(test_data.shape[0], self.most_frequent)
        return pred_labels
