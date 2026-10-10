import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=19, task_kind="classification", dist="euclidean", P=1.0, alpha=0.5):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        if (not (dist == "euclidean" or dist == "chi-square")):
            raise ValueError(f"Unknown distance method: {dist}. Supported methods: euclidean, chi-square")
        if P < 0:
            raise ValueError("P must be >= 0")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.dist = dist
        self.P = P
        self.alpha = alpha
        self.eps = 1e-8
        # Adding variables for storing training data & lables,
        # and initiating with placeholder (unvalid) value.
        self.training_data = None
        self.training_labels = None
        self.class_counts = None
        self.training_size = 0

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        # Make safety checks, I don't know if this is really useful, but it's best practice.
        if (training_labels is None or training_data is None) or (training_data.size == 0 or training_labels.size == 0) :
            raise ValueError("Inputs cannot be None. Please provide valid numerical data.")
        if training_data.shape[0] != training_labels.shape[0]:
            raise ValueError(f"Incompatible shapes: {training_data.shape[0]} samples vs {training_labels.shape[0]} labels.")

        # Save the training data and training labels as indicated, 
        # so that we can use them later in predict function.
        self.training_data = np.array(training_data) # use np.array to deep copy and avoid potential pb
        self.training_labels = np.array(training_labels)
        unique, counts = np.unique(self.training_labels, return_counts=True)
        self.class_counts = {int(label): int(count) for label, count in zip(unique, counts)}
        self.training_size = int(self.training_labels.shape[0])

        # The milestone tests expect fit() to return predictions on the training set.
        return self.predict(self.training_data)

    def get_euclidean_dist(self, source, target):
        return np.sqrt(np.square(source - target).sum())

    def get_chi_square_dist(self, source, target):
        denominator = source + target
        fraction = np.divide(
            np.square(source - target),
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator > 0,
        )
        return np.sqrt(np.sum(fraction))
    
    def distance(self, source, target):
        if self.dist == "euclidean":
            return self.get_euclidean_dist(source, target)
        elif self.dist == "chi-square":
            return self.get_chi_square_dist(source, target)
        else:
            raise ValueError(f"Unknown distance method: {self.dist}. Supported methods: euclidean, chi-square")
    
    def weighted_dist(self, dist):
        # Inverse-distance weighting with tunable exponent P.
        # When P=0, all neighbors have equal weight (no distance-based weighting).
        if self.P == 0:
            return 1.0
        return 1.0 / ((dist + self.eps) ** self.P)

    def label_weight(self, label):
        count = self.class_counts.get(int(label), 0) if self.class_counts is not None else 0
        if count <= 0 or self.training_size <= 0:
            return 1.0
        # Inverse-frequency class weight with tunable exponent alpha.
        return (self.training_size / count) ** self.alpha

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        if self.training_data is None or self.training_labels is None:
            raise ValueError("This KNN instance is not fitted yet! Call .fit() first.")
        if test_data.shape[1] != self.training_data.shape[1]:
            raise ValueError(f"Incompatible shapes: {test_data.shape[1]} test columns vs {self.training_data.shape[1]} training columns.")

        N = test_data.shape[0]
        test_labels = np.zeros(N)

        # Actual KNN logic
        for i in range(N):
            closest_points = [(self.distance(test_data[i], self.training_data[row]), row) for row in range(self.training_data.shape[0])]
            closest_points.sort()

            if self.task_kind == "classification":
                histogram = {}
                weighted_score = {}
                dist_sum = {}
                for (dist, row) in closest_points[:self.k]:
                    label = int(self.training_labels[row])
                    if label not in weighted_score:
                        histogram[label] = 1
                        weighted_score[label] = 0.0
                        dist_sum[label] = 0.0
                    histogram[label] += 1
                    weighted_score[label] += 1.0 * self.weighted_dist(dist) * self.label_weight(label)
                    dist_sum[label] += dist

                max_count = max(weighted_score.values())
                candidates = [label for (label, count) in weighted_score.items() if count == max_count]
                if len(candidates) > 1:
                    # When tied on votes, choose the class with smallest average neighbor distance.
                    test_labels[i] = min(candidates, key=lambda label: dist_sum[label]/histogram[label])
                else: # we take the first and only one
                    test_labels[i] = candidates[0]
            else: # is regression
                average = 0.0
                weight_sum = 0.0
                for (dist, row) in closest_points[:self.k]:
                    if dist == 0:
                        # If there's an element at the exact same point, then simply return that value
                        average = self.training_labels[row]
                        weight_sum = 1.0
                        break
                    inverse_dist = 1.0 / dist
                    average += self.training_labels[row] * inverse_dist
                    weight_sum += inverse_dist

                average /= weight_sum
                test_labels[i] = average

        return test_labels
