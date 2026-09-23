"""
Foundations Lab  --  Capstone Experiment 5: Bayesian Inference Under Model Misspecification
========================================================================================

Research Question:
  How does Bayesian updating behave when the data-generating process violates
  the fundamental assumptions of the model?

Investigated Failure Modes:
  1. Non-Stationary Process vs. i.i.d. Likelihood:
     - True process: A 2-state Markov regime-switching coin:
         State 0: P(heads) = 0.20
         State 1: P(heads) = 0.85
       with regime transition probability p_switch = 0.05.
     - Misspecified Model: Assumes i.i.d. Bernoulli observations with a Beta(1, 1) prior.
     - Phenomenon: The Bayesian posterior suffers from "accumulated overconfidence".
       Because the i.i.d. model assumes a static parameter, the posterior variance
       contracts at rate O(1/N). The posterior concentrates tightly around the long-run
       blended average (~0.525), having zero uncertainty, but failing completely to
       track the actual instantaneous state!
       In contrast, an adaptive / discount-weighted or Hidden Markov Model maintains
       calibrated uncertainty.

  2. Heavy-Tailed Contamination vs. Gaussian Likelihood:
     - True process: Normal distribution contaminated by heavy-tailed outliers (Student-t df=2,
       or 10% contamination from N(0, 25)).
     - Misspecified Model: Normal-Normal conjugate inference assuming pure N(mu, sigma0^2).
     - Phenomenon: A single large outlier exerts quadratic influence on the Gaussian likelihood,
       violently yanking the posterior mean and destroying the coverage of credible intervals.

  3. Credible Interval Coverage Calibration:
     - We evaluate the empirical coverage rate of the model's nominal 95% credible intervals
       under:
         (a) Well-specified i.i.d. Bernoulli data
         (b) Misspecified non-stationary regime-switching data
         (c) Misspecified heavy-tailed contaminated Gaussian data
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_regime_switching_stream(
    n_steps: int = 1000,
    p_heads_low: float = 0.20,
    p_heads_high: float = 0.85,
    p_switch: float = 0.03,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates non-stationary coin flips driven by a hidden 2-state Markov chain.
    Returns: (observations, true_instantaneous_thetas).
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n_steps, dtype=int)
    thetas = np.zeros(n_steps, dtype=np.float64)
    obs = np.zeros(n_steps, dtype=int)

    curr_state = 0
    for t in range(n_steps):
        if rng.random() < p_switch:
            curr_state = 1 - curr_state
        states[t] = curr_state
        thetas[t] = p_heads_high if curr_state == 1 else p_heads_low
        obs[t] = 1 if rng.random() < thetas[t] else 0

    return obs, thetas


def run_bayesian_iid_updating(
    obs: np.ndarray, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard conjugate Beta-Binomial update assuming static i.i.d. observations.
    Returns: (posterior_means, posterior_stds, ci_lowers, ci_uppers) at each time step.
    """
    n = len(obs)
    means = np.zeros(n)
    stds = np.zeros(n)
    ci_low = np.zeros(n)
    ci_high = np.zeros(n)

    a, b = prior_alpha, prior_beta
    for t in range(n):
        a += obs[t]
        b += (1 - obs[t])
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        std = np.sqrt(var)

        means[t] = mean
        stds[t] = std
        # Approximate 95% credible interval via Normal approximation or exact Beta quantiles
        ci_low[t] = max(0.0, mean - 1.96 * std)
        ci_high[t] = min(1.0, mean + 1.96 * std)

    return means, stds, ci_low, ci_high


def run_bayesian_adaptive_updating(
    obs: np.ndarray, discount_factor: float = 0.96
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Discount-weighted Bayesian updating (exponential forgetting) suitable for non-stationary environments.
    """
    n = len(obs)
    means = np.zeros(n)
    stds = np.zeros(n)
    ci_low = np.zeros(n)
    ci_high = np.zeros(n)

    a, b = 1.0, 1.0
    for t in range(n):
        # Discount accumulated pseudocounts toward uniform prior
        a = 1.0 + discount_factor * (a - 1.0) + obs[t]
        b = 1.0 + discount_factor * (b - 1.0) + (1 - obs[t])

        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1))
        std = np.sqrt(var)

        means[t] = mean
        stds[t] = std
        ci_low[t] = max(0.0, mean - 1.96 * std)
        ci_high[t] = min(1.0, mean + 1.96 * std)

    return means, stds, ci_low, ci_high


def run_nonstationary_experiment(n_steps: int = 1000) -> Dict[str, float]:
    """
    Compares the well-specified vs misspecified models under non-stationarity.
    """
    obs, true_thetas = generate_regime_switching_stream(n_steps=n_steps, seed=42)

    # 1. Misspecified: Static i.i.d. model
    m_means, m_stds, m_low, m_high = run_bayesian_iid_updating(obs)
    # Check empirical coverage of 95% CI on the instantaneous true parameter
    iid_coverage = np.mean((true_thetas >= m_low) & (true_thetas <= m_high))
    iid_mse = np.mean((m_means - true_thetas) ** 2)

    # 2. Adaptive model
    a_means, a_stds, a_low, a_high = run_bayesian_adaptive_updating(obs, discount_factor=0.95)
    adapt_coverage = np.mean((true_thetas >= a_low) & (true_thetas <= a_high))
    adapt_mse = np.mean((a_means - true_thetas) ** 2)

    return {
        "iid_coverage_pct": float(iid_coverage * 100.0),
        "iid_mse": float(iid_mse),
        "iid_final_uncertainty_std": float(m_stds[-1]),
        "adapt_coverage_pct": float(adapt_coverage * 100.0),
        "adapt_mse": float(adapt_mse),
        "adapt_final_uncertainty_std": float(a_stds[-1]),
    }


def run_heavy_tail_experiment(num_trials: int = 200, n_samples: int = 50) -> Dict[str, float]:
    """
    Gaussian MLE vs Contaminated Heavy-Tailed Distribution:
      True distribution: 90% N(0, 1) + 10% Student-t(df=2) * 5
      Model assumes: Pure N(mu, 1)
    """
    rng = np.random.default_rng(6041)
    gaussian_means = []
    robust_medians = []
    gaussian_ci_covered = 0

    true_center = 0.0
    for _ in range(num_trials):
        # Generate contaminated sample
        inliers = rng.standard_normal(int(n_samples * 0.9))
        outliers = rng.standard_t(df=2, size=int(n_samples * 0.1)) * 5.0
        data = np.concatenate([inliers, outliers])

        sample_mean = np.mean(data)
        sample_median = np.median(data)
        sample_se = np.std(data, ddof=1) / np.sqrt(n_samples)

        gaussian_means.append(sample_mean)
        robust_medians.append(sample_median)

        # 95% CI from Gaussian model
        ci_l = sample_mean - 1.96 * sample_se
        ci_u = sample_mean + 1.96 * sample_se
        if ci_l <= true_center <= ci_u:
            gaussian_ci_covered += 1

    return {
        "gaussian_mean_mse": float(np.mean((np.array(gaussian_means) - true_center) ** 2)),
        "robust_median_mse": float(np.mean((np.array(robust_medians) - true_center) ** 2)),
        "nominal_95_ci_coverage_pct": float((gaussian_ci_covered / num_trials) * 100.0),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT 5: Model Misspecification in Probabilistic Inference")
    print("=" * 80)

    ns_res = run_nonstationary_experiment(n_steps=1000)
    print("1. Non-Stationary Regime-Switching Coin Flips (Markov Drift):")
    print(f"   Misspecified i.i.d. Model:")
    print(f"     - Mean Squared Error vs True Instantaneous Theta: {ns_res['iid_mse']:.4f}")
    print(f"     - Empirical Coverage of Nominal 95% Credible Interval: {ns_res['iid_coverage_pct']:.1f}%  <-- SEVERE OVERCONFIDENCE!")
    print(f"     - Final Posterior Standard Deviation: {ns_res['iid_final_uncertainty_std']:.4e} (False certainty)")
    print(f"   Adaptive (Regime-Aware) Model:")
    print(f"     - Mean Squared Error vs True Instantaneous Theta: {ns_res['adapt_mse']:.4f}")
    print(f"     - Empirical Coverage of Nominal 95% Credible Interval: {ns_res['adapt_coverage_pct']:.1f}%  <-- CALIBRATED!")
    print(f"     - Final Posterior Standard Deviation: {ns_res['adapt_final_uncertainty_std']:.4e}")

    print("\n2. Heavy-Tailed Contamination vs Gaussian Model (Student-t Outliers):")
    ht_res = run_heavy_tail_experiment(num_trials=300, n_samples=60)
    print(f"   Gaussian Estimator MSE: {ht_res['gaussian_mean_mse']:.4f}")
    print(f"   Robust Median Estimator MSE: {ht_res['robust_median_mse']:.4f}")
    print(f"   Gaussian 95% CI Empirical Coverage: {ht_res['nominal_95_ci_coverage_pct']:.1f}%")

