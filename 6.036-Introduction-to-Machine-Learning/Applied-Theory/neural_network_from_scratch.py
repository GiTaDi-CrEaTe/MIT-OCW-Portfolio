"""
6.036 Applied Theory  --  A Feedforward Neural Network from First Principles
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

