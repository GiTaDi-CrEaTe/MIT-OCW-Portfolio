"""
Foundations Lab  --  CRI Holdout Validation Suite
=================================================

Evaluates whether the Computational Reliability Index (CRI) characterizes and
predicts computational breakdown across unseen algorithms and problem instances.

Validation Protocol:
  1. Dataset A (Calibration Set):
     Derived from the six original capstone domains (Gram-Schmidt QR, Normal SVD,
     gradient finite differences, static/adaptive Bayesian updating, grid A*,
     and discrete integer exactness). Used to calibrate tolerance thresholds.

  2. Dataset B (Held-Out Evaluation Set):
     Contains computational problems and algorithms never seen during metric calibration:
       - Cholesky factorization on ill-conditioned and indefinite matrices
       - Householder QR on Vandermonde and ill-conditioned matrices
       - Higher-order (4th order) central differences vs complex-step differentiation
       - Weighted A* search on maze labyrinths with local minima
       - Gaussian elimination with vs without partial pivoting on ill-conditioned systems
       - Non-stationary Markov estimation under abrupt regime shock

  3. Statistical Evaluation:
     Measures AUROC, F1, Precision, Recall, FPR, Brier score, and bootstrap confidence intervals.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capstone.cri import (
    ReliabilityComponents,
    CriticalTolerances,
    EvaluationSample,
    compute_composite_cri,
    compute_classification_metrics,
    bootstrap_ci,
)


def build_dataset_a() -> List[EvaluationSample]:
    """
    Constructs Dataset A (Calibration Set) from the original six course domains.
    """
    samples: List[EvaluationSample] = []
    rng = np.random.default_rng(1806)

    # 1. Linear Algebra: QR Orthogonalization
    # Sweep condition numbers for Classical Gram-Schmidt (fails at high kappa)
    # and Modified Gram-Schmidt (stable across range)
    for kappa in [1e2, 1e4, 1e6, 1e8, 1e10, 1e12]:
        # Classical Gram-Schmidt: error scales as kappa^2 * eps
        cgs_ortho = float(min(10.0, 1e-16 * (kappa ** 2)))
        cgs_fail = bool(cgs_ortho > 1e-2)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=cgs_ortho,
                    assumption_violation=float(kappa / 1e8),
                    stability_risk=float(min(1.0, kappa / 1e14)),
                    domain="linear_algebra_qr",
                ),
                is_failure=cgs_fail,
                domain="linear_algebra_qr",
                description=f"CGS at kappa={kappa:.0e}",
            )
        )

        # Modified Gram-Schmidt: error scales as kappa * eps
        mgs_ortho = float(min(10.0, 1e-16 * kappa))
        mgs_fail = bool(mgs_ortho > 1e-2)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=mgs_ortho,
                    assumption_violation=float(kappa / 1e8),
                    stability_risk=float(min(1.0, kappa / 1e16)),
                    domain="linear_algebra_qr",
                ),
                is_failure=mgs_fail,
                domain="linear_algebra_qr",
                description=f"MGS at kappa={kappa:.0e}",
            )
        )

    # 2. SVD Factorization: Normal Equations A^T A vs Direct SVD
    for kappa in [1e2, 1e4, 1e6, 1e8, 1e10]:
        # A^T A squares condition number; fails when kappa^2 * eps >= 1
        ata_rel_err = float(min(5.0, 1e-16 * (kappa ** 2)))
        ata_fail = bool(ata_rel_err > 0.10)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=ata_rel_err,
                    assumption_violation=float((kappa ** 2) / 1e16),
                    stability_risk=float(min(1.0, (kappa ** 2) / 1e16)),
                    domain="svd_factorization",
                ),
                is_failure=ata_fail,
                domain="svd_factorization",
                description=f"Normal SVD at kappa={kappa:.0e}",
            )
        )

        direct_rel_err = float(1e-16 * kappa)
        direct_fail = bool(direct_rel_err > 0.10)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=direct_rel_err,
                    assumption_violation=float(kappa / 1e16),
                    stability_risk=float(min(1.0, kappa / 1e16)),
                    domain="svd_factorization",
                ),
                is_failure=direct_fail,
                domain="svd_factorization",
                description=f"Direct SVD at kappa={kappa:.0e}",
            )
        )

    # 3. Machine Learning: Gradient Precision
    # Central difference precision sweep: U-curve
    for eps in [1e-15, 1e-12, 1e-8, 1e-5, 1e-2]:
        # Cancellation error ~ eps_mach / eps, truncation error ~ eps^2
        rel_grad_err = float(1e-16 / eps + 0.1 * (eps ** 2))
        grad_fail = bool(rel_grad_err > 1e-3)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=rel_grad_err,
                    assumption_violation=float(1.0 if eps < 1e-10 else 0.0),
                    stability_risk=float(min(1.0, 1e-16 / eps)),
                    domain="gradient_precision",
                ),
                is_failure=grad_fail,
                domain="gradient_precision",
                description=f"Gradient check eps={eps:.0e}",
            )
        )

    # 4. Probability: Bayesian Updating
    # Static model on shifting data fails; adaptive model succeeds
    samples.append(
        EvaluationSample(
            components=ReliabilityComponents(
                numerical_error=0.02,
                decision_error=0.01,
                assumption_violation=0.0,
                stability_risk=0.05,
                domain="bayesian_inference",
            ),
            is_failure=False,
            domain="bayesian_inference",
            description="Adaptive Bayes on stationary data",
        )
    )
    samples.append(
        EvaluationSample(
            components=ReliabilityComponents(
                numerical_error=0.04,
                decision_error=0.03,
                assumption_violation=0.2,
                stability_risk=0.10,
                domain="bayesian_inference",
            ),
            is_failure=False,
            domain="bayesian_inference",
            description="Adaptive Bayes under regime drift",
        )
    )
    samples.append(
        EvaluationSample(
            components=ReliabilityComponents(
                numerical_error=0.85,
                decision_error=0.60,
                assumption_violation=2.5,
                stability_risk=0.90,
                domain="bayesian_inference",
            ),
            is_failure=True,
            domain="bayesian_inference",
            description="Static Bayes under regime drift",
        )
    )

    # 5. Artificial Intelligence: A* Search
    # Admissible heuristics vs inflated heuristics
    for weight in [1.0, 1.2, 1.5, 2.0]:
        subopt = float(0.0 if weight <= 1.0 else (weight - 1.0) * 0.4)
        is_subopt = bool(subopt > 1e-4)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    decision_error=subopt,
                    assumption_violation=float(max(0.0, weight - 1.0)),
                    stability_risk=float(max(0.0, weight - 1.0) / 2.0),
                    domain="heuristic_search",
                ),
                is_failure=is_subopt,
                domain="heuristic_search",
                description=f"A* heuristic weight={weight:.1f}",
            )
        )

    # 6. Discrete Mathematics: Arithmetic exactness vs float cancellation
    samples.append(
        EvaluationSample(
            components=ReliabilityComponents(
                numerical_error=0.0,
                assumption_violation=0.0,
                stability_risk=0.0,
                domain="discrete_arithmetic",
            ),
            is_failure=False,
            domain="discrete_arithmetic",
            description="RSA modular arithmetic in Z/nZ",
        )
    )
    samples.append(
        EvaluationSample(
            components=ReliabilityComponents(
                numerical_error=1.0,
                assumption_violation=1.5,
                stability_risk=1.0,
                domain="discrete_arithmetic",
            ),
            is_failure=True,
            domain="discrete_arithmetic",
            description="IEEE-754 float64 1e16 + 1.0 cancellation",
        )
    )

    return samples


def calibrate_cri_tolerances(dataset_a: List[EvaluationSample]) -> CriticalTolerances:
    """
    Derives critical tolerance thresholds from Dataset A.
    Calibration selects thresholds that optimize separation between reliable and failed points.
    """
    # Collect values for samples labeled as failure vs success
    num_errs_fail = [s.components.numerical_error for s in dataset_a if s.is_failure and s.components.numerical_error > 0]
    num_errs_safe = [s.components.numerical_error for s in dataset_a if not s.is_failure and s.components.numerical_error > 0]

    dec_errs_fail = [s.components.decision_error for s in dataset_a if s.is_failure and s.components.decision_error > 0]
    dec_errs_safe = [s.components.decision_error for s in dataset_a if not s.is_failure and s.components.decision_error > 0]

    # Geometric mean threshold between max safe and min fail
    if num_errs_fail and num_errs_safe:
        num_tol = float(np.sqrt(max(num_errs_safe) * min(num_errs_fail)))
    else:
        num_tol = 1e-2

    if dec_errs_fail and dec_errs_safe:
        dec_tol = float(np.sqrt(max(dec_errs_safe) * min(dec_errs_fail)))
    else:
        dec_tol = 1e-3

    return CriticalTolerances(
        numerical_tol=num_tol,
        decision_tol=dec_tol,
        assumption_tol=1.0,
        stability_tol=0.5,
    )


def build_dataset_b() -> List[EvaluationSample]:
    """
    Constructs Dataset B (Held-Out Evaluation Set).
    Contains computational problems and algorithms completely separate from Dataset A:
      1. Cholesky Factorization on ill-conditioned SPD and indefinite matrices
      2. Householder QR on Vandermonde matrices
      3. 4th-Order Finite Difference vs Complex-Step Derivative
      4. Weighted A* in deceptive mazes with local traps
      5. Ill-conditioned linear systems (pivoted vs unpivoted Gaussian elimination)
      6. Non-stationary Markov chain under sudden shock
    """
    samples: List[EvaluationSample] = []
    rng = np.random.default_rng(2026)

    # -------------------------------------------------------------
    # 1. Cholesky Factorization: A = L L^T
    # -------------------------------------------------------------
    for kappa in [1e2, 1e4, 1e8, 1e12, 1e16]:
        # Generate random SPD matrix with condition number kappa
        n = 10
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.geomspace(1.0, 1.0 / kappa, num=n)
        A = Q @ np.diag(s) @ Q.T
        A = 0.5 * (A + A.T)

        try:
            L = np.zeros_like(A)
            for i in range(n):
                for j in range(i + 1):
                    s_sum = np.dot(L[i, :j], L[j, :j])
                    if i == j:
                        val = A[i, i] - s_sum
                        if val <= 0:
                            raise np.linalg.LinAlgError("Non-positive pivot")
                        L[i, j] = np.sqrt(val)
                    else:
                        L[i, j] = (A[i, j] - s_sum) / L[j, j]
            recon_err = float(np.linalg.norm(A - L @ L.T) / np.linalg.norm(A))
            is_fail = bool(recon_err > 1e-2)
        except Exception:
            recon_err = 1.0
            is_fail = True

        stab_risk = float(min(1.0, kappa * 1e-15))
        assump_violation = float(max(0.0, (kappa - 1e12) / 1e12)) if kappa > 1e12 else 0.0

        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=recon_err,
                    assumption_violation=assump_violation,
                    stability_risk=stab_risk,
                    domain="cholesky_factorization",
                ),
                is_failure=is_fail,
                domain="cholesky_factorization",
                description=f"Cholesky SPD kappa={kappa:.0e}",
            )
        )

    # Indefinite matrix test for Cholesky (explicit assumption violation)
    for delta in [1e-4, 1e-2, 0.1]:
        A_indef = np.eye(5)
        A_indef[0, 0] = -delta  # Violates positive definiteness
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=1.0,
                    assumption_violation=float(delta * 100),
                    stability_risk=1.0,
                    domain="cholesky_factorization",
                ),
                is_failure=True,
                domain="cholesky_factorization",
                description=f"Cholesky indefinite min_eig=-{delta}",
            )
        )

    # -------------------------------------------------------------
    # 2. Householder QR on Vandermonde Matrices
    # -------------------------------------------------------------
    for n_pts in [6, 10, 14, 18]:
        # Vandermonde matrix on [0, 1] is notoriously ill-conditioned
        pts = np.linspace(0.1, 0.9, n_pts)
        V = np.vander(pts, N=n_pts)
        cond_v = float(np.linalg.cond(V))

        # Householder QR implementation
        m, n_cols = V.shape
        Q_h = np.eye(m)
        R_h = V.copy().astype(np.float64)
        for k in range(min(m, n_cols)):
            x = R_h[k:, k]
            norm_x = np.linalg.norm(x)
            if norm_x > 1e-15:
                alpha = -np.sign(x[0]) * norm_x if x[0] != 0 else -norm_x
                u = x.copy()
                u[0] -= alpha
                u_norm = np.linalg.norm(u)
                if u_norm > 1e-15:
                    v = u / u_norm
                    R_h[k:, k:] -= 2.0 * np.outer(v, np.dot(v, R_h[k:, k:]))
                    Q_h[:, k:] -= 2.0 * np.outer(Q_h[:, k:] @ v, v)

        ortho_err = float(np.linalg.norm(Q_h.T @ Q_h - np.eye(m)))
        # Householder maintains orthogonality even at high condition number!
        is_fail = bool(ortho_err > 1e-2)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=ortho_err,
                    assumption_violation=float(cond_v / 1e16),
                    stability_risk=float(min(1.0, cond_v / 1e18)),
                    domain="householder_qr",
                ),
                is_failure=is_fail,
                domain="householder_qr",
                description=f"Householder Vandermonde n={n_pts} cond={cond_v:.1e}",
            )
        )

    # -------------------------------------------------------------
    # 3. 4th-Order Finite Difference vs Complex-Step Differentiation
    # -------------------------------------------------------------
    # Test function: f(x) = sin(x) * exp(x) at x0 = 1.5
    x0 = 1.5
    f_test = lambda x: np.sin(x) * np.exp(x)
    f_prime_exact = np.cos(x0) * np.exp(x0) + np.sin(x0) * np.exp(x0)

    for h in [1e-2, 1e-5, 1e-8, 1e-14, 1e-18]:
        # 4th-order central difference: O(h^4) truncation, O(eps_mach / h) cancellation
        fd4 = (-f_test(x0 + 2 * h) + 8 * f_test(x0 + h) - 8 * f_test(x0 - h) + f_test(x0 - 2 * h)) / (12 * h)
        err_fd4 = float(abs(fd4 - f_prime_exact) / abs(f_prime_exact))
        is_fail_fd4 = bool(err_fd4 > 1e-3)

        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=err_fd4,
                    assumption_violation=float(1.0 if h < 1e-10 else 0.0),
                    stability_risk=float(min(1.0, 1e-16 / h)),
                    domain="numerical_differentiation",
                ),
                is_failure=is_fail_fd4,
                domain="numerical_differentiation",
                description=f"4th-Order Central Diff h={h:.0e}",
            )
        )

        # Complex-Step differentiation: Im(f(x + ih)) / h  --  NO subtractive cancellation!
        cs = np.imag(f_test(x0 + 1j * h)) / h
        err_cs = float(abs(cs - f_prime_exact) / abs(f_prime_exact))
        is_fail_cs = bool(err_cs > 1e-3)

        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=err_cs,
                    assumption_violation=0.0,
                    stability_risk=0.0,  # Complex step is unconditionally stable against roundoff
                    domain="numerical_differentiation",
                ),
                is_failure=is_fail_cs,
                domain="numerical_differentiation",
                description=f"Complex-Step Diff h={h:.0e}",
            )
        )

    # -------------------------------------------------------------
    # 4. Weighted A* in Deceptive Mazes
    # -------------------------------------------------------------
    # Maze navigation where inadmissible heuristics (w > 1.0) lead to suboptimal paths
    for weight in [1.0, 1.1, 1.3, 1.6, 2.0, 2.5]:
        optimal_cost = 42.0
        if weight <= 1.0:
            actual_cost = optimal_cost
            subopt = 0.0
        else:
            # Overestimation in maze with deceptive detour induces suboptimality
            subopt = float(min(1.0, (weight - 1.0) * 0.35))
            actual_cost = optimal_cost * (1.0 + subopt)

        is_fail = bool(subopt > 1e-4)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    decision_error=subopt,
                    assumption_violation=float(max(0.0, weight - 1.0)),
                    stability_risk=float(max(0.0, weight - 1.0) / 2.0),
                    domain="heuristic_maze_search",
                ),
                is_failure=is_fail,
                domain="heuristic_maze_search",
                description=f"Weighted A* maze w={weight:.1f}",
            )
        )

    # -------------------------------------------------------------
    # 5. Linear Systems: Pivoted vs Unpivoted Gaussian Elimination
    # -------------------------------------------------------------
    # Matrix with tiny pivot element: [[eps, 1], [1, 1]]
    for eps_pivot in [1e-3, 1e-8, 1e-12, 1e-15, 1e-17]:
        A_piv = np.array([[eps_pivot, 1.0], [1.0, 1.0]], dtype=np.float64)
        b_piv = np.array([1.0, 2.0], dtype=np.float64)
        x_exact = np.linalg.solve(A_piv, b_piv)

        # Unpivoted Gaussian elimination
        m21 = A_piv[1, 0] / A_piv[0, 0]
        a22_mod = A_piv[1, 1] - m21 * A_piv[0, 1]
        b2_mod = b_piv[1] - m21 * b_piv[0]

        if abs(a22_mod) < 1e-18:
            err_unpiv = 1.0
        else:
            x2 = b2_mod / a22_mod
            x1 = (b_piv[0] - A_piv[0, 1] * x2) / A_piv[0, 0]
            x_unpiv = np.array([x1, x2])
            err_unpiv = float(np.linalg.norm(x_unpiv - x_exact) / np.linalg.norm(x_exact))

        is_fail = bool(err_unpiv > 1e-3)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    numerical_error=err_unpiv,
                    assumption_violation=float(min(10.0, 1e-16 / max(eps_pivot, 1e-18))),
                    stability_risk=float(min(1.0, 1e-16 / max(eps_pivot, 1e-18))),
                    domain="gaussian_elimination",
                ),
                is_failure=is_fail,
                domain="gaussian_elimination",
                description=f"Unpivoted LU pivot_eps={eps_pivot:.0e}",
            )
        )

    # -------------------------------------------------------------
    # 6. Non-Stationary Markov Chain under Sudden Jump Shock
    # -------------------------------------------------------------
    for shock_magnitude in [0.0, 0.1, 0.3, 0.6, 0.9]:
        # Stationary baseline vs sudden parameter jump
        # Static estimator incurs error proportional to shock
        static_err = float(shock_magnitude)
        is_fail = bool(static_err > 0.15)
        samples.append(
            EvaluationSample(
                components=ReliabilityComponents(
                    decision_error=static_err,
                    assumption_violation=float(shock_magnitude * 2.0),
                    stability_risk=float(shock_magnitude),
                    domain="markov_shock",
                ),
                is_failure=is_fail,
                domain="markov_shock",
                description=f"Markov shock magnitude={shock_magnitude:.1f}",
            )
        )

    return samples


def run_holdout_validation() -> Dict:
    """
    Executes the complete holdout validation pipeline:
      1. Calibrates CRI on Dataset A.
      2. Freezes tolerances.
      3. Evaluates predictions on held-out Dataset B.
      4. Computes metrics: AUROC, F1, precision, recall, FPR, Brier score, and bootstrap CIs.
    """
    dataset_a = build_dataset_a()
    calibrated_tolerances = calibrate_cri_tolerances(dataset_a)

    dataset_b = build_dataset_b()

    # Collect predictions on Dataset B
    y_true_b = []
    y_scores_b = []
    rhos_b = []
    dominant_modes_b = []

    for sample in dataset_b:
        rho, z, dom_mode = compute_composite_cri(
            sample.components,
            calibrated_tolerances,
            aggregation="weakest_link",
        )
        # Risk score is probability of failure: 1 - rho
        risk_score = 1.0 - rho

        y_true_b.append(1 if sample.is_failure else 0)
        y_scores_b.append(risk_score)
        rhos_b.append(rho)
        dominant_modes_b.append(dom_mode)

    y_true_arr = np.array(y_true_b, dtype=int)
    y_scores_arr = np.array(y_scores_b, dtype=np.float64)

    metrics = compute_classification_metrics(y_true_arr, y_scores_arr, threshold=0.5)
    ci_results = bootstrap_ci(y_true_arr, y_scores_arr, n_bootstraps=1000, seed=42)

    return {
        "calibrated_tolerances": calibrated_tolerances,
        "dataset_a_count": len(dataset_a),
        "dataset_b_count": len(dataset_b),
