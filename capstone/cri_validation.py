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
