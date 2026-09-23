"""
Foundations Lab  --  Capstone Experiment 2: From-Scratch SVD via Normal Equations vs LAPACK Baseline
====================================================================================================

Research Question:
  Why does computing the SVD via the eigendecomposition of A^T A fail for
  moderately ill-conditioned matrices?

Mathematical Mechanism:
  The textbook derivation of SVD constructs:
    A^T A = V Sigma^2 V^T
  and then computes:
    u_i = (1 / sigma_i) * A * v_i

  The fundamental numerical limitation:
    1. Condition Number Squaring:
       kappa(A^T A) = (sigma_1^2) / (sigma_n^2) = (sigma_1 / sigma_n)^2 = kappa(A)^2.
    2. Floating-Point Truncation:
       In IEEE 754 float64 arithmetic, machine epsilon eps_mach ~= 2.22e-16.
       When kappa(A) >= 10^8, kappa(A^T A) >= 10^16 ~= 1 / eps_mach.
       Any singular value sigma_i < sqrt(eps_mach) * sigma_1 ~= 1.49e-8 * sigma_1
       is completely swamped by roundoff error when A^T A is computed.
    3. Error Amplification in U:
       u_i = (1 / sigma_i) * A * v_i divides by sigma_i. When sigma_i is corrupted,
       the calculated u_i vector loses both normality and orthogonality.

  In contrast, production direct SVD implementations (such as LAPACK's dgesdd,
  which uses Golub-Kahan bidiagonalization with divide-and-conquer) work directly
  on A without ever squaring its condition number, reliably recovering singular
  values down to eps_mach * sigma_1. In our benchmark, we contrast our from-scratch
  A^TA implementation against this trusted LAPACK baseline.
"""

from typing import Dict, List, Tuple
import numpy as np


def svd_via_ata(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes SVD via eigendecomposition of A^T A (the classical textbook algorithm).
    """
    A = A.astype(np.float64)
    m, n = A.shape
    AtA = A.T @ A  # Condition number is squared: kappa(AtA) = kappa(A)^2

    # Compute eigenvalues and eigenvectors of symmetric matrix AtA
    eigvals, V = np.linalg.eigh(AtA)

    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Clamping negative eigenvalues produced by floating-point roundoff
    non_negative_eigvals = np.maximum(eigvals, 0.0)
    singular_values = np.sqrt(non_negative_eigvals)

    # Recover U using u_i = (1 / sigma_i) * A * v_i
    U = np.zeros((m, n), dtype=np.float64)
    for i in range(n):
        if singular_values[i] > 1e-15:
            u_i = A @ V[:, i]
            norm_u = np.linalg.norm(u_i)
            if norm_u > 0:
                U[:, i] = u_i / singular_values[i]
        else:
            U[:, i] = 0.0

    return U, singular_values, V.T


def run_svd_condition_experiment() -> Dict[str, List]:
    """
    Evaluates from-scratch SVD via A^T A against the production LAPACK SVD baseline (dgesdd)
    across condition numbers from 10^1 to 10^12.
    """
    cond_numbers = [10**p for p in range(1, 13)]
    results: Dict[str, List] = {
        "cond_numbers": cond_numbers,
        "true_smallest_sv": [],
        "ata_smallest_sv": [],
        "direct_smallest_sv": [],
        "ata_sv_rel_error": [],
        "direct_sv_rel_error": [],
        "ata_reconstruction_error": [],
        "direct_reconstruction_error": [],
        "ata_u_ortho_error": [],
        "direct_u_ortho_error": [],
    }

    m, n = 20, 10
    rng = np.random.default_rng(1806)

    for cond in cond_numbers:
        # Construct matrix with exact prescribed singular values
        U_true, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V_true, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s_true = np.geomspace(1.0, 1.0 / cond, num=n)
        Sigma = np.zeros((m, n))
        np.fill_diagonal(Sigma, s_true)
        A = U_true @ Sigma @ V_true.T

        # Method 1: From-scratch SVD via A^T A
        U_ata, s_ata, Vt_ata = svd_via_ata(A)
        A_recon_ata = U_ata @ np.diag(s_ata) @ Vt_ata
        ata_recon_err = np.linalg.norm(A - A_recon_ata, 2) / np.linalg.norm(A, 2)
        ata_u_ortho = np.linalg.norm(U_ata.T @ U_ata - np.eye(n), 2)
        ata_sv_err = abs(s_ata[-1] - s_true[-1]) / s_true[-1]

        # Method 2: Trusted production LAPACK SVD baseline (dgesdd)
        U_dir, s_dir, Vt_dir = np.linalg.svd(A, full_matrices=False)
        A_recon_dir = U_dir @ np.diag(s_dir) @ Vt_dir
        dir_recon_err = np.linalg.norm(A - A_recon_dir, 2) / np.linalg.norm(A, 2)
        dir_u_ortho = np.linalg.norm(U_dir.T @ U_dir - np.eye(n), 2)
        dir_sv_err = abs(s_dir[-1] - s_true[-1]) / s_true[-1]

        results["true_smallest_sv"].append(s_true[-1])
        results["ata_smallest_sv"].append(s_ata[-1])
        results["direct_smallest_sv"].append(s_dir[-1])
        results["ata_sv_rel_error"].append(ata_sv_err)
        results["direct_sv_rel_error"].append(dir_sv_err)
        results["ata_reconstruction_error"].append(ata_recon_err)
        results["direct_reconstruction_error"].append(dir_recon_err)
        results["ata_u_ortho_error"].append(ata_u_ortho)
        results["direct_u_ortho_error"].append(dir_u_ortho)

    return results


if __name__ == "__main__":
    print("=" * 82)
    print("EXPERIMENT 2: From-Scratch SVD via A^T A Normal Equations vs LAPACK SVD Baseline")
    print("=" * 82)
    res = run_svd_condition_experiment()

    print(f"{'kappa(A)':>10} | {'True sigma_min':>15} | {'A^TA sigma_min':>15} | {'A^TA Rel Err':>13} | {'LAPACK Rel Err':>14}")
    print("-" * 82)
    for i in range(len(res["cond_numbers"])):
        c = res["cond_numbers"][i]
        s_true = res["true_smallest_sv"][i]
        s_ata = res["ata_smallest_sv"][i]
        err_ata = res["ata_sv_rel_error"][i]
        err_dir = res["direct_sv_rel_error"][i]
        print(f"{c:10.1e} | {s_true:15.6e} | {s_ata:15.6e} | {err_ata:13.4e} | {err_dir:14.4e}")

    print("\n" + "=" * 82)
    print("Finding: Beyond kappa(A) = 10^8, A^TA squares condition number past 1/eps_mach,")
    print("completely wiping out the smallest singular values.")
    print("=" * 82)
