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

