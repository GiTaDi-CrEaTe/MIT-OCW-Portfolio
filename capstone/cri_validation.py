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
