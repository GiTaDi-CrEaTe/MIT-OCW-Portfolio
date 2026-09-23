"""
Foundations Lab  --  Capstone Experiment 1: Numerical Stability in Linear Algebra
=============================================================================

Research Question:
  When does mathematically correct linear algebra become numerically unreliable?

Focus:
  Classical Gram-Schmidt (CGS) vs. Modified Gram-Schmidt (MGS) under
  increasing condition numbers kappa(A) in [10^1, 10^14], and on ill-conditioned
  Hilbert matrices.

Mathematical Background:
  In exact arithmetic, CGS and MGS produce identical orthonormal bases Q and
  upper-triangular matrices R.
  In floating-point arithmetic (IEEE 754 float64, eps_mach ~ 2.22e-16):
    - CGS projects each column against all previous vectors using the original
      column components, making it susceptible to catastrophic cancellation:
        ||Q^T Q - I||_2 = O(eps_mach * kappa(A)^2)
    - MGS updates the working vector sequentially after each projection, removing
      components in the currently computed coordinates:
        ||Q^T Q - I||_2 = O(eps_mach * kappa(A))
  When kappa(A) >= 10^8, CGS completely loses orthogonality (error ~ 1.0),
  producing columns that are nowhere near orthogonal, even though the
  reconstruction residual ||A - QR|| remains relatively small.
"""

from typing import Dict, List, Tuple
import numpy as np


def classical_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical Gram-Schmidt (CGS) QR factorization.
    All projections are computed against the original column a_k.
    """
    A = A.astype(np.float64)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], A[:, k])
            v -= R[j, k] * Q[:, j]
        norm_v = np.linalg.norm(v)
        R[k, k] = norm_v
        if norm_v > 1e-15:
            Q[:, k] = v / norm_v
        else:
            Q[:, k] = 0.0

    return Q, R


def modified_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Modified Gram-Schmidt (MGS) QR factorization.
    Projections are applied sequentially to the partially orthogonalized vector.
    """
    A = A.astype(np.float64)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    V = A.copy()
    for k in range(n):
        norm_v = np.linalg.norm(V[:, k])
        R[k, k] = norm_v
        if norm_v > 1e-15:
            Q[:, k] = V[:, k] / norm_v
        else:
            Q[:, k] = 0.0

        for j in range(k + 1, n):
            R[k, j] = np.dot(Q[:, k], V[:, j])
            V[:, j] -= R[k, j] * Q[:, k]

    return Q, R


def generate_controlled_matrix(m: int, n: int, cond_number: float, seed: int = 42) -> np.ndarray:
    """
    Constructs an m x n matrix with an exact prescribed 2-norm condition number
    using SVD: A = U * Sigma * V^T.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    # Geometrically spaced singular values from 1.0 down to 1.0 / cond_number
    s = np.geomspace(1.0, 1.0 / cond_number, num=n)
    Sigma = np.zeros((m, n), dtype=np.float64)
    np.fill_diagonal(Sigma, s)

    return U @ Sigma @ V.T


def hilbert_matrix(n: int) -> np.ndarray:
    """Generates an n x n Hilbert matrix: H_{i,j} = 1 / (i + j + 1)."""
    i, j = np.indices((n, n))
    return 1.0 / (i + j + 1.0)


def evaluate_qr_accuracy(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Dict[str, float]:
    """
    Measures:
      - Residual: ||A - Q R||_2 / ||A||_2
      - Orthogonality loss: ||Q^T Q - I_n||_2
    """
    n = Q.shape[1]
    norm_A = np.linalg.norm(A, 2)
    residual = np.linalg.norm(A - Q @ R, 2) / (norm_A if norm_A > 0 else 1.0)
    ortho_error = np.linalg.norm(Q.T @ Q - np.eye(n), 2)
    return {
        "residual": float(residual),
        "orthogonality_error": float(ortho_error),
    }


def run_condition_number_sweep() -> Dict[str, List]:
    """
    Sweeps condition numbers from 10^1 to 10^14 and compares CGS vs MGS.
    """
    cond_numbers = [10**p for p in range(1, 15)]
    results: Dict[str, List] = {
        "cond_numbers": cond_numbers,
        "cgs_ortho": [],
        "mgs_ortho": [],
        "cgs_resid": [],
        "mgs_resid": [],
    }

    m, n = 30, 15
    for cond in cond_numbers:
        A = generate_controlled_matrix(m, n, cond, seed=42)

        Q_cgs, R_cgs = classical_gram_schmidt(A)
        res_cgs = evaluate_qr_accuracy(A, Q_cgs, R_cgs)

        Q_mgs, R_mgs = modified_gram_schmidt(A)
        res_mgs = evaluate_qr_accuracy(A, Q_mgs, R_mgs)

        results["cgs_ortho"].append(res_cgs["orthogonality_error"])
        results["mgs_ortho"].append(res_mgs["orthogonality_error"])
        results["cgs_resid"].append(res_cgs["residual"])
        results["mgs_resid"].append(res_mgs["residual"])

    return results


def run_hilbert_experiment() -> Dict[str, Dict[str, float]]:
    """Evaluates CGS vs MGS on a 10x10 Hilbert matrix (notoriously ill-conditioned)."""
    H = hilbert_matrix(10)
    cond = float(np.linalg.cond(H))

    Q_cgs, R_cgs = classical_gram_schmidt(H)
    Q_mgs, R_mgs = modified_gram_schmidt(H)

    return {
        "condition_number": cond,
        "cgs": evaluate_qr_accuracy(H, Q_cgs, R_cgs),
        "mgs": evaluate_qr_accuracy(H, Q_mgs, R_mgs),
    }


if __name__ == "__main__":
    print("=" * 72)
    print("EXPERIMENT 1: Classical Gram-Schmidt vs Modified Gram-Schmidt")
    print("=" * 72)

    sweep = run_condition_number_sweep()
    print(f"{'Condition':>12} | {'CGS Ortho Error':>17} | {'MGS Ortho Error':>17} | {'Ratio CGS/MGS':>13}")
    print("-" * 72)
    for c, c_err, m_err in zip(sweep["cond_numbers"], sweep["cgs_ortho"], sweep["mgs_ortho"]):
        ratio = c_err / m_err if m_err > 0 else float("inf")
        print(f"{c:12.1e} | {c_err:17.4e} | {m_err:17.4e} | {ratio:13.2e}")

    print("\n" + "=" * 72)
    print("Hilbert Matrix Benchmark (n = 10)")
    print("=" * 72)
    h_res = run_hilbert_experiment()
    print(f"Condition number kappa(H_10) = {h_res['condition_number']:.3e}")
    print(f"CGS Orthogonality Loss: {h_res['cgs']['orthogonality_error']:.4e}")
    print(f"MGS Orthogonality Loss: {h_res['mgs']['orthogonality_error']:.4e}")
    print(f"CGS Residual:           {h_res['cgs']['residual']:.4e}")
    print(f"MGS Residual:           {h_res['mgs']['residual']:.4e}")

