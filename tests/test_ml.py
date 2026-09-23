"""
Tests for 6.036 Neural Networks from Scratch
"""

import numpy as np
import pytest
from conftest import load_course_module

ml = load_course_module("ml_scratch", "6.036-Introduction-to-Machine-Learning/Applied-Theory/neural_network_from_scratch.py")


def test_neural_network_numerical_gradient_check():
    rng = np.random.default_rng(36)
    if hasattr(ml, "make_two_rings"):
        X, y = ml.make_two_rings(n_per_class=10, seed=1)
        net = ml.NeuralNetwork(layer_sizes=[2, 4, 1], hidden_activation="tanh", seed=1)
        err = ml.numerical_gradient_check(net, X, y, epsilon=1e-5, num_checks=20)
        assert err < 1e-4
    else:
        net = ml.NeuralNetwork([2, 4, 2], seed=42)
        X = rng.standard_normal((2, 5))
        Y = np.zeros((2, 5))
        Y[0, :3] = 1
        Y[1, 3:] = 1
        errs = ml.numerical_gradient_check(net, X, Y, epsilon=1e-7)
        assert all(v < 1e-4 for v in errs.values())


def test_neural_network_learns_nonlinear_data():
    if hasattr(ml, "make_two_rings"):
        X, y = ml.make_two_rings(n_per_class=100, seed=36)
        net = ml.NeuralNetwork(layer_sizes=[2, 12, 12, 1], hidden_activation="tanh", seed=3)
        l_start = net.train_step(X, y, learning_rate=0.4)
        for _ in range(400):
            l_end = net.train_step(X, y, learning_rate=0.4)
        assert l_end < l_start
    elif hasattr(ml, "NeuralNetwork"):
        X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
        Y_xor = np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
        net = ml.NeuralNetwork([2, 8, 2], seed=42)
        losses = net.train(X_xor, Y_xor, epochs=500, learning_rate=0.5, print_every=1000)
        assert losses[-1] < losses[0]



