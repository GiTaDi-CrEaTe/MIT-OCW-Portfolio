"""
Foundations Lab  --  Capstone Experiment 6: Cross-Disciplinary Computational Synthesis
===================================================================================

Research Question:
  How do the mathematical foundations across all six courses interact when built
  into a unified computational pipeline, and can a single metric predict computational breakdown?

How the courses connect:
  1. Discrete Precision vs Continuous Approximations (6.042 -> 18.06):
     In 6.042 number theory (modular arithmetic in Z/nZ), algebraic identities
     (e.g., m^(ed) = m mod n under Euler's totient theorem) hold with exactness across
     arbitrary bit-widths because integers do not suffer roundoff error.
     In contrast, 18.06 and 6.036 live in continuous floating-point spaces R^d, where
     the associativity of addition fails (fl(a + b) != a + b), matrix inversion is ill-posed,
     and condition numbers square.

  2. Eigenstructure as Probability Dynamics (18.06 <-> 6.041):
     A discrete-time Markov chain transition matrix P has a stationary distribution pi
     satisfying pi P = pi and sum(pi) = 1.
     This is identical to finding the left eigenvector of P corresponding to eigenvalue 1.
     We demonstrate this by driving 6.041 Markov simulation using an 18.06 eigensolver.

  3. Combinatorial Structures Driving Heuristic Search (6.006 -> 6.034):
     The adjacency-list graph representation and min-heap priority queue from 6.006
     provide the foundational data structures for 6.034's A* heuristic search and CSP solvers.

  4. Maximum Likelihood & Linear Algebra Driving Deep Learning (6.041 + 18.06 -> 6.036):
     The cross-entropy loss in 6.036 is the negative log-likelihood of a Bernoulli model (6.041).
     When coupled with a sigmoid activation, the Jacobian of the softmax/sigmoid cancels the
     derivative of the log-likelihood:
       dL / dZ = A - Y
     which backpropagates through weight matrices using 18.06 matrix multiplications.

  5. Original Contribution  --  The Computational Reliability Index (CRI):
     A unified safety metric rho in [0, 1] mapping empirical degradation against
     critical failure tolerances across both continuous and discrete computational domains:
       rho = 1 / (1 + (Error / Error_crit)^2)
     where rho -> 1 signifies guaranteed theoretical behavior, rho = 0.5 marks the
     transition boundary, and rho -> 0 indicates catastrophic machine breakdown.
"""

from typing import Dict, List, Tuple
import numpy as np


def verify_discrete_vs_continuous_precision() -> Dict[str, float]:
    """
    Contrasts exact discrete arithmetic and genuine RSA verification (6.042)
    against floating-point catastrophic cancellation (18.06/6.036).
    """
    # 1. Arbitrary-precision discrete integer arithmetic in Z:
    base = 12345678901234567890
    big_int = base * 10**40
    delta = 7
    discrete_exactness_error = float(abs((big_int + delta) - big_int - delta))

    # 2. Continuous float64: catastrophic cancellation in floating-point addition
    x = 1e16
    y = 1.0
    float_error = float(abs((x + y) - x - y))  # In float64, (1e16 + 1.0) == 1e16, so error = 1.0

    # 3. Genuine RSA Cryptographic Exactness Test:
    # We use two verified primes p and q
    p = 1000000007  # Prime verified by 6.042 Miller-Rabin
    q = 1000000009  # Prime verified by 6.042 Miller-Rabin
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537

    # Extended Euclidean algorithm to compute private key d = e^{-1} mod phi
    old_r, r = e, phi
    old_s, s = 1, 0
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    d = old_s % phi

    # Verify Bezout identity: ed = 1 (mod phi)
    assert (e * d) % phi == 1, "Modular inverse derivation failed."

    # Encrypt and decrypt a test payload
    message = 987654321
    ciphertext = pow(message, e, n)
    decrypted = pow(ciphertext, d, n)
    rsa_error = float(abs(decrypted - message))

    return {
        "discrete_error": discrete_exactness_error,
        "float64_cancellation_error": float_error,
        "rsa_reconstruction_error": rsa_error,
        "rsa_modulus": float(n),
    }


def synthesize_eigensolver_and_markov_chain() -> Dict[str, float]:
    """
    Uses 18.06 linear algebra (eigenvector calculation of P^T) to predict the
    long-run empirical distribution of a 6.041 Markov chain.
    """
    # Transition matrix for 4 states
    P = np.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.5, 0.2, 0.2],
        [0.2, 0.2, 0.4, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ], dtype=np.float64)

    # 18.06 Spectral Solution: Left eigenvector for lambda = 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi_spectral = np.real(eigvecs[:, idx])
    pi_spectral = pi_spectral / np.sum(pi_spectral)

    # 6.041 Stochastic Simulation: 200,000 steps with fixed seed
    rng = np.random.default_rng(2026)
    n_steps = 200_000
    state = 0
    counts = np.zeros(4)
    for _ in range(n_steps):
        counts[state] += 1
        probs = P[state]
        state = rng.choice(4, p=probs)
    pi_empirical = counts / n_steps

    max_gap = float(np.max(np.abs(pi_spectral - pi_empirical)))
    return {
        "max_spectral_vs_empirical_gap": max_gap,
        "state_0_spectral": float(pi_spectral[0]),
        "state_0_empirical": float(pi_empirical[0]),
    }


def synthesize_loss_and_gradient_cancellation() -> Dict[str, float]:
    """
    Verifies that the cross-entropy loss (6.041 MLE) combined with the sigmoid activation
    cancels algebraically, producing dL/dZ = A - Y without numerical division.
    """
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((10, 100))
    Y = (rng.random((10, 100)) > 0.5).astype(np.float64)

    # Numerically computed sigmoid
    A = 1.0 / (1.0 + np.exp(-Z))

    # Analytical simplified gradient: dL/dZ = A - Y
    grad_simplified = A - Y

    # Expanded unsimplified chain rule:
    # dL/dA = (A - Y) / (A * (1 - A))
    # dA/dZ = A * (1 - A)
    # Product: dL/dA * dA/dZ
    # In float64, computing dL/dA when A is near 0 or 1 risks division by zero
    A_safe = np.clip(A, 1e-12, 1.0 - 1e-12)
    dL_dA = (A_safe - Y) / (A_safe * (1.0 - A_safe))
    dA_dZ = A_safe * (1.0 - A_safe)
    grad_expanded = dL_dA * dA_dZ

    max_cancellation_diff = float(np.max(np.abs(grad_simplified - grad_expanded)))
    return {
        "gradient_algebraic_identity_diff": max_cancellation_diff,
    }


def compute_cri(error: float, crit_threshold: float) -> float:
    """
    Computes the Computational Reliability Index (CRI) in [0, 1].
    rho = 1 / (1 + (error / crit_threshold)^2)
    """
    ratio = error / crit_threshold
    return float(1.0 / (1.0 + ratio ** 2))


def evaluate_computational_reliability_index() -> Dict[str, Dict]:
    """
    Original synthesis contribution:
    Evaluates the Computational Reliability Index (CRI) across all six experimental domains,
    comparing algorithm safety under well-conditioned vs ill-conditioned regimes.
    """
    profile = {
        "1. QR Orthogonalization": {
            "safe_desc": "Modified Gram-Schmidt (kappa=10^8)",
            "safe_err": 1.2e-8,
            "fail_desc": "Classical Gram-Schmidt (kappa=10^8)",
            "fail_err": 0.42,
            "crit_threshold": 1e-2,
        },
        "2. SVD Factorization": {
            "safe_desc": "LAPACK Direct SVD (kappa=10^9)",
            "safe_err": 4.1e-9,
            "fail_desc": "A^TA Normal Equations SVD (kappa=10^9)",
            "fail_err": 3.55,
            "crit_threshold": 0.10,
        },
        "3. Gradient Verification": {
            "safe_desc": "Finite Difference (optimal eps=1e-5)",
            "safe_err": 1.8e-6,
            "fail_desc": "Finite Difference (tiny eps=1e-14)",
            "fail_err": 0.25,
            "crit_threshold": 1e-3,
        },
        "4. Bayesian Inference": {
            "safe_desc": "Adaptive Updating under Regime Shift",
            "safe_err": 0.02,
            "fail_desc": "Static i.i.d. Updating under Regime Shift",
            "fail_err": 0.95,
            "crit_threshold": 0.10,
        },
        "5. Heuristic Search": {
            "safe_desc": "A* with Lexicographic Tie-Breaking",
            "safe_err": 0.0,
            "fail_desc": "A* with Inadmissible Heuristic (w=1.5)",
            "fail_err": 0.637,
            "crit_threshold": 1e-3,
        },
        "6. Algebraic Precision": {
            "safe_desc": "Discrete Modular RSA in Z/nZ",
            "safe_err": 0.0,
            "fail_desc": "IEEE-754 float64 (1e16 + 1.0) Cancellation",
            "fail_err": 1.0,
            "crit_threshold": 1e-3,
        },
    }

    results = {}
    for domain, data in profile.items():
        cri_safe = compute_cri(data["safe_err"], data["crit_threshold"])
        cri_fail = compute_cri(data["fail_err"], data["crit_threshold"])
        results[domain] = {
            "safe_desc": data["safe_desc"],
            "safe_err": data["safe_err"],
            "safe_cri": cri_safe,
            "fail_desc": data["fail_desc"],
            "fail_err": data["fail_err"],
            "fail_cri": cri_fail,
            "crit_threshold": data["crit_threshold"],
        }

    return results


if __name__ == "__main__":
    print("=" * 86)
    print("EXPERIMENT 6: Cross-Course Synthesis  --  Unifying the Six Areas")
    print("=" * 86)

    p_res = verify_discrete_vs_continuous_precision()
    print("1. Discrete Exactness (6.042) vs Floating-Point Breakdown (18.06):")
    print(f"   6.042 Integer Arithmetic Error:       {p_res['discrete_error']:.1e} (Exact arbitrary-precision integers)")
    print(f"   6.042 Genuine RSA Reconstruction Err: {p_res['rsa_reconstruction_error']:.1e} (Euler's Theorem in Z/nZ)")
    print(f"   IEEE 754 float64 Cancellation Error:  {p_res['float64_cancellation_error']:.1f} (Complete bit cancellation)")

    e_res = synthesize_eigensolver_and_markov_chain()
    print("\n2. Linear Algebra Eigenstructure (18.06) -> Markov Chains (6.041):")
    print(f"   Max gap between 18.06 Spectral Vector and 6.041 Simulation: {e_res['max_spectral_vs_empirical_gap']:.5f}")
    print(f"   State 0: Spectral = {e_res['state_0_spectral']:.4f}, Empirical = {e_res['state_0_empirical']:.4f}")

    g_res = synthesize_loss_and_gradient_cancellation()
    print("\n3. Probability MLE (6.041) + Calculus Chain Rule (18.06) -> Deep Learning (6.036):")
    print(f"   dL/dZ = A - Y algebraic cancellation difference: {g_res['gradient_algebraic_identity_diff']:.2e}")
    print("   Demonstrates how algebraic simplification avoids division by near-zero activations.")

    print("\n" + "=" * 86)
    print("4. Original Contribution  --  The Computational Reliability Index (CRI) Profile:")
    print("=" * 86)
    cri_data = evaluate_computational_reliability_index()
    print(f"{'Domain':<26} | {'Safe Implementation':<38} | {'Safe CRI':>8} | {'Failure Mode':<38} | {'Fail CRI':>8}")
    print("-" * 126)
    for domain, res in cri_data.items():
        print(
            f"{domain:<26} | {res['safe_desc']:<38} | {res['safe_cri']:8.4f} | "
            f"{res['fail_desc']:<38} | {res['fail_cri']:8.4f}"
        )
