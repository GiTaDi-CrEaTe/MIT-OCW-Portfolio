"""
6.041 Applied Theory  --  Bayesian Inference, MLE, and Markov Chains from Scratch
================================================================================

Three pieces, chained to make a single argument: how belief should update
under evidence (Bayes), how that relates to the "just count and divide"
frequentist estimate (MLE), and how probabilistic models evolve over time
(Markov chains). Only `numpy` and `random` are used, purely as array/RNG
tools -- every probability computation is hand-derived.
"""

import numpy as np
import random

np.set_printoptions(precision=4, suppress=True)


# ===========================================================================
# PART 1  --  Bayesian inference on a discrete hypothesis space (Pset 7)
# ===========================================================================

def bayesian_update(prior: np.ndarray, hypotheses: np.ndarray, data: int, n_trials: int):
    """
    Discrete Bayesian updating for a Bernoulli-type experiment.

    `hypotheses` is a grid of candidate success probabilities theta in [0, 1].
    `prior[i]` = P(theta = hypotheses[i]) before seeing data.
    `data` = number of successes observed in `n_trials` Bernoulli trials.

    Theory: Bayes' rule, derived directly from the definition of conditional
    probability P(A|B) = P(A and B) / P(B):

        P(theta | data) = P(data | theta) * P(theta) / P(data)

    where P(data | theta) is the Binomial likelihood
        C(n_trials, data) * theta^data * (1-theta)^(n_trials - data)
    and P(data) = sum over all theta of the numerator (law of total
    probability) -- this is exactly the normalizing constant.

    The binomial coefficient C(n_trials, data) is the SAME for every
    hypothesis, so it cancels in the normalization and is omitted below --
    a small but real efficiency/clarity gain that falls straight out of
    the algebra.
    """
    likelihood = (hypotheses ** data) * ((1 - hypotheses) ** (n_trials - data))
    unnormalized_posterior = likelihood * prior
    evidence = unnormalized_posterior.sum()  # law of total probability
    posterior = unnormalized_posterior / evidence
    return posterior


def posterior_mean(hypotheses: np.ndarray, posterior: np.ndarray) -> float:
    """E[theta | data] = sum_i hypotheses[i] * P(theta = hypotheses[i] | data)."""
    return float(np.sum(hypotheses * posterior))


# ===========================================================================
# PART 2  --  Maximum Likelihood Estimation (Pset 8), compared to the Bayesian
# posterior mean to show MLE as the large-n limit of Bayesian updating.
# ===========================================================================

def mle_bernoulli(data_sequence) -> float:
    """
    MLE for a Bernoulli parameter theta given i.i.d. coin-flip data.

    Theory: the likelihood of the full sequence is
        L(theta) = theta^k * (1-theta)^(n-k)     (k successes out of n)
    Maximizing L is equivalent to maximizing log L (log is monotonic), and
        d/d(theta) [k*log(theta) + (n-k)*log(1-theta)] = 0
    solves to theta_hat = k / n -- the sample proportion. This "just count
    and divide" answer is not a heuristic; it is the exact calculus solution
    to the likelihood-maximization problem.
    """
    data_sequence = list(data_sequence)
    n = len(data_sequence)
    k = sum(data_sequence)
    return k / n


# ===========================================================================
# PART 3  --  Markov chains: simulation vs. analytically solved stationary
# distribution (Pset 11)
# ===========================================================================

