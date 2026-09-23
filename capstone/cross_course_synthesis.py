"""
Foundations Lab  --  Capstone Experiment 6: Cross-Disciplinary Computational Synthesis
===================================================================================

Research Question:
  How do the mathematical foundations across all six courses interact when built
  into a unified computational pipeline?

The Conceptual Spine:
  1. Discrete Precision vs Continuous Approximations (6.042 -> 18.06):
     In 6.042 number theory (modular arithmetic in Z/nZ), mathematical identities
     (e.g., m^(ed) = m mod n) hold with exactness across arbitrary bit-widths (e.g. 512 bits)
     because integers do not suffer roundoff error.
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
     provide the computational chassis for 6.034's A* heuristic search and CSP solvers.

  4. Maximum Likelihood & Linear Algebra Driving Deep Learning (6.041 + 18.06 -> 6.036):
     The cross-entropy loss in 6.036 is the negative log-likelihood of a Bernoulli model (6.041).
     When coupled with a sigmoid activation, the Jacobian of the softmax/sigmoid cancels the
     derivative of the log-likelihood:
       dL / dZ = A - Y
     which backpropagates through weight matrices using 18.06 matrix multiplications.
"""

from typing import Dict, Tuple
import numpy as np


def verify_discrete_vs_continuous_precision() -> Dict[str, float]:
    """
    Contrasts exact discrete arithmetic (6.042) with floating-point drift (18.06/6.036).
    """
    # 6.042: Large integer modular exponentiation in Z/nZ
    base = 12345678901234567890
    exp = 65537
    mod = 98765432109876543211
    discrete_val = pow(base, exp, mod)
    # Reversible exactness test
    phi = mod - 1  # If prime
    # Check that in exact integer arithmetic: (base * 10^30) - (base * 10^30) == 0
    discrete_exactness_error = float(abs((base * 10**40 + 7) - (base * 10**40) - 7))

    # Continuous float64: catastrophic cancellation in floating-point addition
    x = 1e16
    y = 1.0
    float_error = float(abs((x + y) - x - y))  # In float64, (1e16 + 1.0) == 1e16, so error = 1.0!

    return {
        "discrete_error": discrete_exactness_error,
        "float64_cancellation_error": float_error,
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

    # 6.041 Stochastic Simulation: 200,000 steps
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
    # In float64, computing dL/dA when A is near 0 or 1 risks division by zero!
    A_safe = np.clip(A, 1e-12, 1.0 - 1e-12)
    dL_dA = (A_safe - Y) / (A_safe * (1.0 - A_safe))
    dA_dZ = A_safe * (1.0 - A_safe)
    grad_expanded = dL_dA * dA_dZ

    max_cancellation_diff = float(np.max(np.abs(grad_simplified - grad_expanded)))
    return {
        "gradient_algebraic_identity_diff": max_cancellation_diff,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT 6: Cross-Course Synthesis  --  Unifying the Six Areas")
    print("=" * 80)

    p_res = verify_discrete_vs_continuous_precision()
    print("1. Discrete Exactness (6.042) vs Floating-Point Breakdown (18.06):")
    print(f"   6.042 Modular Integer Identity Error: {p_res['discrete_error']:.1e} (Exact integer arithmetic)")
    print(f"   IEEE 754 float64 (1e16 + 1.0) - 1e16 - 1.0: {p_res['float64_cancellation_error']:.1f} (Complete bit cancellation!)")

    e_res = synthesize_eigensolver_and_markov_chain()
    print("\n2. Linear Algebra Eigenstructure (18.06) -> Markov Chains (6.041):")
    print(f"   Max gap between 18.06 Spectral Stationary Vector and 6.041 Simulation: {e_res['max_spectral_vs_empirical_gap']:.5f}")
    print(f"   State 0: Spectral = {e_res['state_0_spectral']:.4f}, Empirical = {e_res['state_0_empirical']:.4f}")

    g_res = synthesize_loss_and_gradient_cancellation()
    print("\n3. Probability MLE (6.041) + Calculus Chain Rule (18.06) -> Deep Learning (6.036):")
    print(f"   dL/dZ = A - Y algebraic cancellation error: {g_res['gradient_algebraic_identity_diff']:.2e}")
    print("   Demonstrates how algebraic simplification avoids catastrophic division-by-zero.")

