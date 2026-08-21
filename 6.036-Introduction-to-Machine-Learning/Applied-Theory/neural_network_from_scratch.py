"""
6.036 Applied Theory — A Feedforward Neural Network from First Principles
============================================================================

Implements forward propagation, backpropagation, and gradient-descent
training entirely by hand: every partial derivative below is derived from
the chain rule and written out explicitly. No autograd, no `torch`, no
`sklearn`. `numpy` is used only for matrix arithmetic.

------------------------------------------------------------------------------
THEORY RECAP (see Pset 7-8 in Psets/pset_roadmap.md)
------------------------------------------------------------------------------
For a network with L layers, layer l computes:
    z[l] = W[l] @ a[l-1] + b[l]        (pre-activation, "logits" of the layer)
    a[l] = activation(z[l])            (post-activation)
with a[0] = the input x.

Forward pass: apply this layer by layer to get the final output a[L] and the
scalar loss J = loss(a[L], y).

Backward pass (the chain rule, applied mechanically):
    dJ/da[L]                              -- depends on the loss function
    dJ/dz[l] = dJ/da[l] * activation'(z[l])            (elementwise)
    dJ/dW[l] = dJ/dz[l] @ a[l-1]^T
    dJ/db[l] = dJ/dz[l]
    dJ/da[l-1] = W[l]^T @ dJ/dz[l]         -- propagate the error one layer back

This is EXACTLY backpropagation: each layer receives an "error signal"
dJ/dz[l] from the layer after it, uses it to compute its own parameter
gradients, and passes a transformed error signal to the layer before it.
------------------------------------------------------------------------------
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Activation functions and their derivatives (needed explicitly for backprop)
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_prime),
    "tanh": (tanh, tanh_prime),
}


# ---------------------------------------------------------------------------
# The network itself
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    A fully connected feedforward network for binary classification.
    `layer_sizes` e.g. [2, 8, 8, 1] means: 2 inputs, two hidden layers of
    8 units each (tanh activation), one sigmoid output unit.

    Loss: binary cross-entropy,
        J = -[ y*log(a_L) + (1-y)*log(1-a_L) ]
    chosen (as in Pset 6) because it is the negative log-likelihood of a
    Bernoulli model -- i.e. this loss IS maximum likelihood estimation,
    not an arbitrary design choice.

    A convenient identity used below: for a sigmoid output layer combined
    with cross-entropy loss, dJ/dz_L simplifies exactly to (a_L - y),
    with no leftover sigmoid-derivative term. This is a standard,
    provable simplification (the two derivatives cancel algebraically)
    and is used here explicitly rather than hidden inside a library.
    """

    def __init__(self, layer_sizes, hidden_activation="tanh", seed=36):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight layers
        self.hidden_act, self.hidden_act_prime = ACTIVATIONS[hidden_activation]

        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for l in range(self.L):
            fan_in, fan_out = layer_sizes[l], layer_sizes[l + 1]
            # Xavier-style initialization: keeps activations from
            # saturating (all-zero or huge random initialization both
            # break gradient flow, as noted in Pset 9's training-dynamics
            # unit).
            scale = np.sqrt(1.0 / fan_in)
            self.W.append(rng.standard_normal((fan_out, fan_in)) * scale)
            self.b.append(np.zeros((fan_out, 1)))

    def forward(self, X):
        """
        X: shape (n_features, n_examples).
        Returns final activation a_L and caches every z[l], a[l] needed
        for the backward pass.
        """
        a = X
        cache = {"a0": a}
        for l in range(self.L):
            z = self.W[l] @ a + self.b[l]
            if l < self.L - 1:
                a = self.hidden_act(z)
            else:
                a = sigmoid(z)  # output layer always sigmoid for binary classification
            cache[f"z{l+1}"] = z
            cache[f"a{l+1}"] = a
        return a, cache

    def compute_loss(self, a_L, y):
        """Binary cross-entropy, averaged over examples. Clipped for numerical safety."""
        eps = 1e-12
        a_L = np.clip(a_L, eps, 1 - eps)
        m = y.shape[1]
        return float(-(1.0 / m) * np.sum(y * np.log(a_L) + (1 - y) * np.log(1 - a_L)))

    def backward(self, y, cache):
        """
        Full backpropagation, layer by layer, following exactly the chain-rule
        recap at the top of this file.
        """
        m = y.shape[1]
        grads_W = [None] * self.L
        grads_b = [None] * self.L

        a_L = cache[f"a{self.L}"]
        dz = a_L - y  # dJ/dz_L, using the sigmoid+cross-entropy simplification

        for l in reversed(range(self.L)):
            a_prev = cache[f"a{l}"] if l > 0 else cache["a0"]
            grads_W[l] = (1.0 / m) * (dz @ a_prev.T)
            grads_b[l] = (1.0 / m) * np.sum(dz, axis=1, keepdims=True)

            if l > 0:
                da_prev = self.W[l].T @ dz             # propagate error one layer back
                z_prev = cache[f"z{l}"]
                dz = da_prev * self.hidden_act_prime(z_prev)  # apply chain rule through activation

        return grads_W, grads_b

    def train_step(self, X, y, learning_rate):
        a_L, cache = self.forward(X)
        loss = self.compute_loss(a_L, y)
        grads_W, grads_b = self.backward(y, cache)
        for l in range(self.L):
            self.W[l] -= learning_rate * grads_W[l]
            self.b[l] -= learning_rate * grads_b[l]
        return loss

    def predict(self, X):
        a_L, _ = self.forward(X)
        return (a_L > 0.5).astype(int)

    # -- parameter (de)serialization, used by the gradient checker below --
    def get_flat_params(self):
        return np.concatenate([w.flatten() for w in self.W] + [b.flatten() for b in self.b])

    def set_flat_params(self, flat):
        idx = 0
        for l in range(self.L):
            size = self.W[l].size
            self.W[l] = flat[idx:idx + size].reshape(self.W[l].shape)
            idx += size
        for l in range(self.L):
            size = self.b[l].size
            self.b[l] = flat[idx:idx + size].reshape(self.b[l].shape)
            idx += size


# ---------------------------------------------------------------------------
# Numerical gradient checking (Pset 8's standard backprop-debugging technique)
# ---------------------------------------------------------------------------

def numerical_gradient_check(net: NeuralNetwork, X, y, epsilon=1e-5, num_checks=30):
    """
    For a handful of randomly chosen parameters, approximates dJ/dparam via
    the symmetric finite-difference formula:
        dJ/dparam ~= [J(param + eps) - J(param - eps)] / (2 * eps)
    and compares it against the analytically computed backprop gradient.
    This is the textbook way to verify a from-scratch backprop implementation
    is actually correct, rather than merely "loss goes down" (which can be
    true even with a subtly wrong gradient, e.g. if the sign is right but the
    magnitude is off due to a chain-rule slip).
    """
    _, cache = net.forward(X)
    grads_W, grads_b = net.backward(y, cache)
    analytic_grad = np.concatenate([g.flatten() for g in grads_W] + [g.flatten() for g in grads_b])

    flat_params = net.get_flat_params()
    rng = np.random.default_rng(0)
    check_indices = rng.choice(len(flat_params), size=min(num_checks, len(flat_params)), replace=False)

    max_relative_error = 0.0
    for idx in check_indices:
        original = flat_params[idx]

        flat_params[idx] = original + epsilon
        net.set_flat_params(flat_params)
        a_plus, _ = net.forward(X)
        loss_plus = net.compute_loss(a_plus, y)

        flat_params[idx] = original - epsilon
        net.set_flat_params(flat_params)
        a_minus, _ = net.forward(X)
        loss_minus = net.compute_loss(a_minus, y)

        flat_params[idx] = original  # restore
        net.set_flat_params(flat_params)

        numeric_grad = (loss_plus - loss_minus) / (2 * epsilon)
        analytic = analytic_grad[idx]
        rel_error = abs(numeric_grad - analytic) / max(abs(numeric_grad) + abs(analytic), 1e-8)
        max_relative_error = max(max_relative_error, rel_error)

    return max_relative_error


# ---------------------------------------------------------------------------
# Synthetic non-linearly-separable dataset ("two rings"): a task a single
# linear layer (Pset 2/6) provably cannot solve, motivating hidden layers.
# ---------------------------------------------------------------------------

def make_two_rings(n_per_class=200, seed=36):
    rng = np.random.default_rng(seed)
    theta_inner = rng.uniform(0, 2 * np.pi, n_per_class)
    r_inner = rng.normal(1.0, 0.15, n_per_class)
    inner = np.stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    theta_outer = rng.uniform(0, 2 * np.pi, n_per_class)
    r_outer = rng.normal(2.5, 0.15, n_per_class)
    outer = np.stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.concatenate([inner, outer], axis=1)  # shape (2, 2*n_per_class)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).reshape(1, -1)
    return X, y


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------

def _self_test():
    print("=" * 70)
    print("SELF-TEST 1: Numerical gradient check on random data")
    print("(confirms the hand-derived backprop gradients are correct BEFORE")
    print("using them to train anything)")
    print("=" * 70)
    rng = np.random.default_rng(36)
    X_toy = rng.standard_normal((4, 20))
    y_toy = (rng.random((1, 20)) > 0.5).astype(float)
    net = NeuralNetwork(layer_sizes=[4, 6, 5, 1], hidden_activation="tanh", seed=1)
    max_rel_error = numerical_gradient_check(net, X_toy, y_toy, epsilon=1e-5, num_checks=40)
    print(f"Max relative error (analytic vs. finite-difference gradient): {max_rel_error:.2e}")
    assert max_rel_error < 1e-4, "Backprop gradients do not match finite-difference approximation!"
    print("PASSED: hand-derived backpropagation matches numerical differentiation.\n")

    print("=" * 70)
    print("SELF-TEST 2: A single linear layer (logistic regression) CANNOT")
    print("separate the two-rings dataset -- demonstrating why depth matters,")
    print("not just asserting it")
    print("=" * 70)
    X, y = make_two_rings(n_per_class=200, seed=36)
    linear_model = NeuralNetwork(layer_sizes=[2, 1], seed=2)  # no hidden layer = logistic regression
    for _ in range(2000):
        linear_model.train_step(X, y, learning_rate=0.5)
    linear_preds = linear_model.predict(X)
    linear_accuracy = float(np.mean(linear_preds == y))
    print(f"Logistic regression (no hidden layer) accuracy on two-rings: {linear_accuracy:.3f}")
    assert linear_accuracy < 0.75, "Linear model should NOT be able to solve this non-linear task."
    print("Confirms: a linear decision boundary cannot separate concentric rings.\n")

    print("=" * 70)
    print("SELF-TEST 3: A 2-hidden-layer network solves the same task well")
    print("=" * 70)
    deep_net = NeuralNetwork(layer_sizes=[2, 16, 16, 1], hidden_activation="tanh", seed=3)
    n_epochs = 3000
    losses = []
    for epoch in range(n_epochs):
        loss = deep_net.train_step(X, y, learning_rate=0.3)
        if epoch % 500 == 0:
            losses.append(loss)
            print(f"  epoch {epoch:4d}   loss = {loss:.4f}")
    deep_preds = deep_net.predict(X)
    deep_accuracy = float(np.mean(deep_preds == y))
    print(f"Final training loss: {loss:.4f}")
    print(f"2-hidden-layer network accuracy on two-rings: {deep_accuracy:.3f}")
    assert deep_accuracy > 0.95, "Deep network should solve the non-linear task with high accuracy."
    assert losses[-1] < losses[0], "Loss should decrease over training."
    print("PASSED: non-linear hidden layers solve a task linear models provably cannot.\n")

    print("All self-tests passed. Backpropagation is verified against finite")
    print("differences, and the resulting network demonstrably solves a task")
    print("that is provably out of reach for a linear model.")


if __name__ == "__main__":
    _self_test()
