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


def test_markov_chain_stationary_distribution():
    P = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.4, 0.5],
    ], dtype=float)

    if hasattr(prob, "stationary_distribution_analytic"):
        pi = prob.stationary_distribution_analytic(P)
        assert np.isclose(np.sum(pi), 1.0)
        assert np.allclose(pi @ P, pi, atol=1e-8)
    elif hasattr(prob, "analytical_stationary_distribution"):
        pi = prob.analytical_stationary_distribution(P)
        assert np.isclose(np.sum(pi), 1.0)
        assert np.allclose(pi @ P, pi, atol=1e-8)



def test_mle_bernoulli_extreme_cases():
    """MLE should handle all-success and all-failure data correctly."""
    assert np.isclose(prob.mle_bernoulli([1, 1, 1, 1]), 1.0)
    assert np.isclose(prob.mle_bernoulli([0, 0, 0, 0]), 0.0)


def test_stationary_distribution_sums_to_one():
    """Stationary distribution must sum to exactly 1."""
    if hasattr(prob, "stationary_distribution_analytic"):
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi = prob.stationary_distribution_analytic(P)
        assert np.isclose(np.sum(pi), 1.0, atol=1e-10)
        assert np.all(pi >= 0)
