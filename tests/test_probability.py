"""
Tests for 6.041 Probabilistic Systems Analysis
"""

import numpy as np
import pytest
from conftest import load_course_module

prob = load_course_module("prob_scratch", "6.041-Probabilistic-Systems-Analysis/Applied-Theory/bayesian_inference_and_markov_chains.py")


def test_mle_bernoulli():
    data = [1, 1, 0, 1, 0, 1, 1, 1]  # 6 / 8 = 0.75
    theta_hat = prob.mle_bernoulli(data)
    assert np.isclose(theta_hat, 0.75)


def test_bayesian_update_convergence():
    if hasattr(prob, "bayesian_update"):
        hypotheses = np.linspace(0.01, 0.99, 99)
        prior = np.ones_like(hypotheses) / len(hypotheses)
        post = prob.bayesian_update(prior, hypotheses, data=70, n_trials=100)
        mean_est = prob.posterior_mean(hypotheses, post)
        assert abs(mean_est - 0.70) < 0.05
    elif hasattr(prob, "bayesian_update_beta_binomial"):
        a, b, _ = prob.bayesian_update_beta_binomial(1.0, 1.0, [1]*70 + [0]*30)
        mean_est = a / (a + b)
        assert abs(mean_est - 0.70) < 0.05


