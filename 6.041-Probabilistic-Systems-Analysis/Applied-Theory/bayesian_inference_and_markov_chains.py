"""
6.041 Applied Theory — Bayesian Inference, MLE, and Markov Chains from Scratch
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
# PART 1 — Bayesian inference on a discrete hypothesis space (Pset 7)
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
# PART 2 — Maximum Likelihood Estimation (Pset 8), compared to the Bayesian
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
# PART 3 — Markov chains: simulation vs. analytically solved stationary
# distribution (Pset 11)
# ===========================================================================

def simulate_markov_chain(transition_matrix: np.ndarray, start_state: int,
                           n_steps: int, rng: random.Random):
    """
    Simulates a discrete-time Markov chain for n_steps and returns the
    empirical visitation frequency of each state -- the "run it and count"
    approach.
    transition_matrix[i, j] = P(next state = j | current state = i).
    """
    n_states = transition_matrix.shape[0]
    counts = np.zeros(n_states)
    state = start_state
    for _ in range(n_steps):
        counts[state] += 1
        probs = transition_matrix[state]
        state = rng.choices(range(n_states), weights=probs, k=1)[0]
    return counts / n_steps


def stationary_distribution_analytic(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Solves for the stationary distribution pi satisfying pi = pi @ P and
    sum(pi) = 1, directly as a linear system -- the "solve it exactly"
    approach, to be checked against the simulation above.

    Theory: pi P = pi means pi (P - I) = 0, i.e. pi is a left null vector of
    (P - I). Combined with the normalization constraint sum(pi) = 1, this
    pins down pi uniquely for an irreducible, aperiodic chain. We solve it by
    replacing one equation of (P^T - I) pi^T = 0 with the normalization
    constraint, turning it into a standard solvable linear system.
    """
    n = transition_matrix.shape[0]
    A = transition_matrix.T - np.eye(n)
    A[-1, :] = 1.0  # replace last equation with the normalization constraint
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return pi


# ===========================================================================
# Self-verification
# ===========================================================================

def _self_test():
    print("=" * 70)
    print("SELF-TEST 1: Bayesian posterior concentrates around the true theta")
    print("as more coin-flip evidence accumulates")
    print("=" * 70)
    true_theta = 0.7
    hypotheses = np.linspace(0.01, 0.99, 99)
    prior = np.ones_like(hypotheses) / len(hypotheses)  # uniform prior: "no idea"

    rng_np = np.random.default_rng(6041)
    posterior = prior.copy()
    sample_sizes_to_report = {10, 50, 200, 1000}
    total_flips = 0
    for batch_idx, batch_size in enumerate([10, 40, 150, 800]):
        flips = rng_np.binomial(1, true_theta, size=batch_size)
        successes = int(flips.sum())
        posterior = bayesian_update(posterior, hypotheses, successes, batch_size)
        total_flips += batch_size
        if total_flips in sample_sizes_to_report:
            mean_est = posterior_mean(hypotheses, posterior)
            # "concentration" proxy: posterior probability mass within 0.05 of truth
            mass_near_truth = posterior[np.abs(hypotheses - true_theta) < 0.05].sum()
            print(f"After {total_flips:4d} flips: posterior mean = {mean_est:.4f}  "
                  f"(true theta = {true_theta}),  "
                  f"P(|theta - true| < 0.05) = {mass_near_truth:.4f}")

    final_mean = posterior_mean(hypotheses, posterior)
    assert abs(final_mean - true_theta) < 0.03, "Posterior mean should converge near the true theta."
    print(f"PASSED: final posterior mean {final_mean:.4f} is within 0.03 of true theta {true_theta}.\n")

    print("=" * 70)
    print("SELF-TEST 2: MLE matches the Bayesian posterior mean under a")
    print("uniform (uninformative) prior, as sample size grows")
    print("(this is the Bernstein-von Mises phenomenon taught qualitatively")
    print("in 6.041's discussion of the relationship between the two schools)")
    print("=" * 70)
    py_rng = random.Random(41)
    coin_flips = [1 if py_rng.random() < true_theta else 0 for _ in range(2000)]
    mle_estimate = mle_bernoulli(coin_flips)
    print(f"MLE estimate from 2000 flips: {mle_estimate:.4f}")
    print(f"Bayesian posterior mean from 1000 flips (above): {final_mean:.4f}")
    assert abs(mle_estimate - true_theta) < 0.03
    assert abs(mle_estimate - final_mean) < 0.05
    print("PASSED: MLE and the Bayesian posterior mean agree closely at large n.\n")

    print("=" * 70)
    print("SELF-TEST 3: Long-run Markov chain simulation matches the")
    print("analytically solved stationary distribution")
    print("=" * 70)
    # A simple 3-state weather chain: Sunny, Cloudy, Rainy
    P = np.array([
        [0.6, 0.3, 0.1],   # from Sunny
        [0.2, 0.5, 0.3],   # from Cloudy
        [0.1, 0.4, 0.5],   # from Rainy
    ])
    pi_analytic = stationary_distribution_analytic(P)
    print(f"Analytic stationary distribution:  {pi_analytic}")
    assert np.allclose(pi_analytic @ P, pi_analytic, atol=1e-8), "pi must be a fixed point of P."
    assert abs(pi_analytic.sum() - 1.0) < 1e-8

    sim_rng = random.Random(2024)
    pi_simulated = simulate_markov_chain(P, start_state=0, n_steps=500_000, rng=sim_rng)
    print(f"Simulated long-run frequencies:    {pi_simulated}")
    diff = np.max(np.abs(pi_analytic - pi_simulated))
    print(f"max |analytic - simulated| = {diff:.4f}")
    assert diff < 0.01, "Simulation should match the analytic stationary distribution closely."
    print("PASSED: 500,000-step simulation matches the exact linear-algebra solution.\n")

    print("All self-tests passed. Bayesian updating, MLE, and Markov chain")
    print("theory are each verified against an independent ground truth.")


if __name__ == "__main__":
    _self_test()
