import copy

import numpy as np
from src.activations import Sigmoid, ReLU, Identity, Softmax


class MLP:
    """Minimal multilayer perceptron with flexible training options.

    - Constructor keeps structural arguments (dimensions, activations, init_method).
    - Regularization, dropout and optimization options are provided to `fit()`.
    """

    def __init__(self, dimensions, activations, init_method="basic", l2_lambda=0.0, dropout_rate=0.0):
        self.dimensions = np.asarray(dimensions)
        self.activations = np.asarray(activations)
        self.init_method = init_method
        self.l2_lambda = float(l2_lambda)
        self.dropout_rate = float(dropout_rate)
        self._dropout_masks = {}
        self.known_activations = (Sigmoid, ReLU, Identity, Softmax)

        self._validate_init_args()
        self._initialize_weights()

    def feed_forward(self, x, training=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        z = {0: x}
        a = {}
        if training:
            self._dropout_masks = {}
        for i in range(self.activations.size):
            a_i = z[i] @ self.weights[i] + self.biases[i]
            a[i + 1] = a_i
            z_i = self.activations[i].forward(a_i)
            # Apply inverted dropout on hidden layers only when training
            if training and (i < self.activations.size - 1) and (self.dropout_rate > 0.0):
                keep_prob = 1.0 - self.dropout_rate
                mask = (np.random.rand(*z_i.shape) < keep_prob) / keep_prob
                z_i = z_i * mask
                self._dropout_masks[i + 1] = mask
            z[i + 1] = z_i
        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        prev_training = getattr(self, '_training', False)
        self._training = False
        try:
            z, _ = self.feed_forward(x, training=False)
        finally:
            self._training = prev_training
        return z[len(self.dimensions) - 1]

    def back_prop(self, z, a, y_true, loss, sample_weights=None):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return: weight_gradients, bias_gradients
        """
        pre_activation = a
        post_activation = z
        weight_gradients = []
        bias_gradients = []

        # Output layer
        output_layer_idx = len(self.dimensions) - 1
        # Try to pass sample_weights to loss.gradient when supported
        try:
            output_loss_gradient = loss.gradient(y_true, post_activation[output_layer_idx], sample_weights=sample_weights)
        except TypeError:
            output_loss_gradient = loss.gradient(y_true, post_activation[output_layer_idx])

        # If output activation is Softmax and loss gradient already accounts for it, skip derivative
        if self.activations[output_layer_idx - 1] is Softmax:
            output_final_delta = output_loss_gradient
        else:
            output_activation_derivative = self.activations[output_layer_idx - 1].gradient(pre_activation[output_layer_idx])
            output_final_delta = output_loss_gradient * output_activation_derivative

        weight_gradients.insert(0, post_activation[output_layer_idx - 1].T @ output_final_delta)
        bias_gradients.insert(0, output_final_delta.mean(axis=0))
        layer_deltas = [output_final_delta]

        # Hidden layers
        for layer_idx in range((len(self.dimensions) - 1) - 1, 0, -1):
            curr_layer_delta = layer_deltas[0] @ self.weights[layer_idx].T
            curr_activation_derivative = self.activations[layer_idx - 1].gradient(pre_activation[layer_idx])
            curr_final_delta = curr_layer_delta * curr_activation_derivative
            # Apply dropout mask if present
            if hasattr(self, '_dropout_masks') and (layer_idx in self._dropout_masks):
                curr_final_delta = curr_final_delta * self._dropout_masks[layer_idx]
            weight_gradients.insert(0, post_activation[layer_idx - 1].T @ curr_final_delta)
            bias_gradients.insert(0, curr_final_delta.mean(axis=0))
            layer_deltas.insert(0, curr_final_delta)

        return weight_gradients, bias_gradients

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        self.weights[index] -= dw
        # L2 weight decay applied after gradient step
        if getattr(self, 'l2_lambda', 0.0) > 0.0 and getattr(self, '_last_lr', None) is not None:
            self.weights[index] -= self._last_lr * self.l2_lambda * self.weights[index]
        self.biases[index] -= delta

    def fit(
        self,
        x,
        y_true,
        loss,
        epochs,
        batch_size,
        learning_rate=1e-5,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        track_test_loss=False,
        optimizer="sgd",
        momentum=0.0,
        weight_decay=0.0,
        dropout_rate=0.0,
        patience=None,
        min_delta=0.0,
        restore_best_weights=False,
        sample_weights=None,
    ):
        """
        Train the MLP using mini-batch updates.

        Parameters (high-level):
        - x, y_true: training data and targets.
        - loss: loss object with `.loss()` and `.gradient()` methods. May accept `sample_weights`.
        - epochs, batch_size: training loop size controls.
        - learning_rate: base step size for parameter updates (default aligned with `main.py`).
        - optimizer: one of 'sgd', 'momentum', 'adam'. 'momentum' enables classical momentum, 'adam' enables Adam.
        - momentum: momentum coefficient used when `optimizer` is 'momentum' or 'adam' (Adam uses its own internal betas).
        - weight_decay: coefficient applied to the gradients each update (classic "weight decay" / L2 on gradients).
        - dropout_rate: dropout applied to hidden activations during training (this value will be assigned to the model for the session).
        - patience, min_delta, restore_best_weights: early-stopping controls using validation loss.
        - sample_weights: per-sample weights passed through to loss functions when supported.

        Notes:
        - There are two L2-related mechanisms in the implementation: `weight_decay` (added to gradients during update)
          and the model-level `l2_lambda` stored on the instance (applied inside `update_w_b` using `self._last_lr`).
          Both may be set; `weight_decay` acts immediately on gradients, while `l2_lambda` is a deferred multiplier applied after gradient step.
        - If `dropout_rate` is provided here it will override the model's stored `dropout_rate` for the training session.
        """

        n_samples = x.shape[0]
        history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
        if optimizer not in {"sgd", "momentum", "adam"}:
            raise ValueError(f"optimizer must be one of ('sgd', 'momentum', 'adam'), got '{optimizer}'")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        use_momentum = optimizer in {"momentum", "adam"}
        use_adam = optimizer == "adam"

        v_weights = [np.zeros_like(W) for W in self.weights]
        v_biases = [np.zeros_like(b) for b in self.biases]
        m_weights = [np.zeros_like(W) for W in self.weights]
        m_biases = [np.zeros_like(b) for b in self.biases]
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8
        adam_t = 0

        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        epochs_without_improvement = 0
        should_stop = False

        # store last learning rate for L2 application
        self._last_lr = learning_rate

        # Ensure dropout specified at fit-time is applied to the model
        self.dropout_rate = float(dropout_rate)

        for epoch in range(epochs):
            # Begin epoch: set training mode (affects dropout behavior)
            self._training = True

            # Shuffle the training data at the start of each epoch to form mini-batches
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_true_shuffled = y_true[indices]
            if sample_weights is not None:
                sw_shuffled = sample_weights[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_true_batch = y_true_shuffled[start_idx:end_idx]
                sample_weights_batch = None
                if sample_weights is not None:
                    sample_weights_batch = sw_shuffled[start_idx:end_idx]

                # If using Adam, increment timestep for bias-correction
                if use_adam:
                    adam_t += 1

                # Forward pass (with dropout if training=True)
                z, a = self.feed_forward(x_batch, training=True)

                # Backpropagate to compute gradients for this batch
                weight_gradients, bias_gradients = self.back_prop(
                    z, a, y_true_batch, loss, sample_weights=sample_weights_batch
                )

                for layer_idx in range(len(self.dimensions) - 1):
                    # Apply explicit weight decay (L2) to the gradient if requested.
                    # Note: this is distinct from `self.l2_lambda`, which is applied
                    # in `update_w_b` after the parameter update (scaled by last lr).
                    if weight_decay > 0.0:
                        weight_gradients[layer_idx] = weight_gradients[layer_idx] + weight_decay * self.weights[layer_idx]

                    # Optimizer-specific parameter updates
                    if use_adam:
                        m_weights[layer_idx] = (
                            adam_beta1 * m_weights[layer_idx] + (1.0 - adam_beta1) * weight_gradients[layer_idx]
                        )
                        m_biases[layer_idx] = (
                            adam_beta1 * m_biases[layer_idx] + (1.0 - adam_beta1) * bias_gradients[layer_idx]
                        )
                        v_weights[layer_idx] = (
                            adam_beta2 * v_weights[layer_idx] + (1.0 - adam_beta2) * (weight_gradients[layer_idx] ** 2)
                        )
                        v_biases[layer_idx] = (
                            adam_beta2 * v_biases[layer_idx] + (1.0 - adam_beta2) * (bias_gradients[layer_idx] ** 2)
                        )

                        m_hat_w = m_weights[layer_idx] / (1.0 - adam_beta1 ** adam_t)
                        m_hat_b = m_biases[layer_idx] / (1.0 - adam_beta1 ** adam_t)
                        v_hat_w = v_weights[layer_idx] / (1.0 - adam_beta2 ** adam_t)
                        v_hat_b = v_biases[layer_idx] / (1.0 - adam_beta2 ** adam_t)

                        # Compute Adam parameter updates with bias-correction
                        weight_update = learning_rate * m_hat_w / (np.sqrt(v_hat_w) + adam_epsilon)
                        bias_update = learning_rate * m_hat_b / (np.sqrt(v_hat_b) + adam_epsilon)
                        # Apply updates (update_w_b also applies model-level l2 if configured)
                        self.update_w_b(layer_idx, weight_update, bias_update)
                    elif use_momentum and momentum > 0.0:
                        # Classical momentum: accumulate velocity and apply it
                        v_weights[layer_idx] = momentum * v_weights[layer_idx] + learning_rate * weight_gradients[layer_idx]
                        v_biases[layer_idx] = momentum * v_biases[layer_idx] + learning_rate * bias_gradients[layer_idx]
                        self.update_w_b(layer_idx, v_weights[layer_idx], v_biases[layer_idx])
                    else:
                        # Plain SGD update (no momentum)
                        self.update_w_b(layer_idx, learning_rate * weight_gradients[layer_idx], learning_rate * bias_gradients[layer_idx])

            # === Epoch metrics & validation ===
            train_pred = self.predict(x)
            try:
                train_loss = loss.loss(y_true, train_pred, sample_weights=sample_weights) if sample_weights is not None else loss.loss(y_true, train_pred)
            except TypeError:
                train_loss = loss.loss(y_true, train_pred)
            history['train_loss'].append(train_loss)

            # Compute validation loss and handle early stopping bookkeeping
            if x_val is not None and y_val is not None:
                val_pred = self.predict(x_val)
                val_loss = loss.loss(y_val, val_pred)
                history['val_loss'].append(val_loss)
                if patience is not None:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_weights = [W.copy() for W in self.weights]
                        best_biases = [b.copy() for b in self.biases]
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

            # Optionally evaluate test loss each epoch (kept separate from val for debugging)
            if track_test_loss and x_test is not None and y_test is not None:
                test_pred = self.predict(x_test)
                history['test_loss'].append(loss.loss(y_test, test_pred))

            # Check early stopping condition
            if patience is not None and x_val is not None and y_val is not None and epochs_without_improvement >= patience:
                should_stop = True

            if should_stop:
                break

        if restore_best_weights and best_weights is not None and best_biases is not None:
            self.weights = [W.copy() for W in best_weights]
            self.biases = [b.copy() for b in best_biases]

        return history

    # === Helper methods ===
    def _validate_init_args(self):
        valid_inits = ("basic", "he", "xavier")
        if self.init_method not in valid_inits:
            raise ValueError(f"init_method must be one of {valid_inits}, got '{self.init_method}'")

        if self.dimensions.ndim != 1:
            raise ValueError(f"dimensions must be 1 dimensional, got {self.dimensions.ndim}")
        if self.dimensions.size < 3:
            raise ValueError(f"the MLP should have at least 3 layers, got {self.dimensions.size}")
        for dim in self.dimensions:
            if not isinstance(dim, (int, np.integer)):
                raise TypeError(f"each dimension must be an integer, got {type(dim)}")
            if dim <= 0:
                raise ValueError(f"each dimension must be a strictly positive integer, got {dim}")

        if self.activations.ndim != 1:
            raise ValueError(f"activations must be 1 dimensional, got {self.activations.ndim}")
        if self.activations.size != self.dimensions.size - 1:
            raise ValueError(
                f"activations must have N-1 elements where N = len(dimensions), got {self.activations.size} activations for {self.dimensions.size} dimensions"
            )
        for activation in self.activations:
            if not isinstance(activation, type):
                raise TypeError(f"each activation must be a class, got {type(activation)}")
            if activation not in self.known_activations:
                raise ValueError(
                    f"activation '{activation.__name__}' not recognized. Must be one of {[act.__name__ for act in self.known_activations]}"
                )

    def _initialize_weights(self):
        self.weights = []
        self.biases = []
        for i in range(self.dimensions.size - 1):
            in_dim = int(self.dimensions[i])
            out_dim = int(self.dimensions[i + 1])

            if self.init_method == "he":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            elif self.init_method == "xavier":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))
            else:
                W = np.random.randn(in_dim, out_dim) * 0.01

            self.weights.append(W)
            b = np.zeros(out_dim)
            self.biases.append(b)

    def _get_dropout_mask(self, activation_array, dropout_rate, dropout_masks, layer_idx):
        if dropout_masks is not None and dropout_masks.get(layer_idx) is not None:
            return dropout_masks[layer_idx]
        return (np.random.rand(*activation_array.shape) >= dropout_rate).astype(activation_array.dtype)
