"""
Computational Reliability Index (CRI)
=====================================

Research Question:
  Can computational failure across different mathematical domains be characterized
  by a unified reliability model?

Mathematical Formulation:
  For any algorithm executing on physical hardware, we define the composite risk
  state vector as:
    x = (e_num, e_dec, v_assump, r_stab)

  where:
    1. e_num (numerical error):
       Relative deviation from exact algebraic or analytical value,
       e.g., ||Q^T Q - I||_2, |sigma_hat - sigma| / sigma, ||grad_num - grad_ana|| / ||grad||.
    2. e_dec (decision error):
       Discrete combinatorial suboptimality or decision penalty,
       e.g., (cost - cost_opt) / cost_opt, classification error on critical branches.
    3. v_assump (assumption violation):
       Degree of divergence from theoretical domain premises,
       e.g., condition number kappa / kappa_limit, heuristic overestimation factor, drift rate.
    4. r_stab (instability risk):
       Inverse stability margin: r_stab = 1 - m_stab, where m_stab in [0, 1] represents
       remaining distance to machine epsilon, rank deficiency, or numerical singular points.

  Weakest-Link Principle:
    A computational pipeline fails if ANY single component breaches its critical tolerance.
    We normalize each component against its calibrated critical tolerance tau_k:
      z_k = x_k / tau_k
      z = max_k (z_k)

    The Computational Reliability Index rho in [0, 1] is computed as:
      rho = 1 / (1 + z^2)

  Properties:
    - rho -> 1.0: Safe computational regime (all z_k << 1).
    - rho = 0.5: Critical boundary (the worst component reaches its tolerance, z = 1.0).
    - rho -> 0.0: Catastrophic breakdown (at least one component severely exceeds tolerance, z >> 1).
    - Predicted Failure: Failure is predicted when rho < 0.5 (equivalently z > 1.0).

  Engineering Note on Condition Estimation:
    In these benchmark suites (n <= 15), exact 2-norm condition numbers kappa(A)
    are computed via SVD. For large-scale production linear algebra, computing full
    SVD costs O(n^3); an O(n^2) 1-norm condition estimator such as LAPACK's dgecon
    should be substituted to avoid matching the cost of the linear solve itself.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ReliabilityComponents:
    """
    Decomposition of computational state into four foundational axes.
    All values are non-negative.
    """
    numerical_error: float = 0.0
    decision_error: float = 0.0
    assumption_violation: float = 0.0
    stability_risk: float = 0.0
    domain: str = "generic"
    metadata: Optional[Dict[str, Union[float, str, bool]]] = None


@dataclass
class CriticalTolerances:
    """
    Calibrated tolerance thresholds for each reliability component.
    """
    numerical_tol: float = 1e-2
    decision_tol: float = 1e-3
    assumption_tol: float = 1.0
    stability_tol: float = 0.5


@dataclass
class EvaluationSample:
    """
    A single experiment instance with measured state and empirical binary failure label.
    """
    components: ReliabilityComponents
    is_failure: bool
    domain: str
    description: str


def compute_scalar_cri(error: float, crit_threshold: float) -> float:
    """
    Single-variable CRI computation for backward compatibility.
    rho = 1 / (1 + (error / crit_threshold)^2)
    """
    if crit_threshold <= 0:
        raise ValueError("Critical threshold must be strictly positive.")
    ratio = max(0.0, float(error)) / float(crit_threshold)
    return float(1.0 / (1.0 + ratio ** 2))


def compute_composite_cri(
    components: ReliabilityComponents,
    tolerances: CriticalTolerances,
    aggregation: str = "weakest_link",
    p_norm: float = 4.0,
) -> Tuple[float, float, str]:
    """
    Evaluates the CRI across all four axes.

    Args:
        components: Measured reliability components.
        tolerances: Domain or global critical tolerances.
        aggregation: 'weakest_link' (max normalized ratio) or 'p_norm' (smooth soft-max).
        p_norm: Order of norm if aggregation is 'p_norm'.

    Returns:
        (rho, z, dominant_mode)
        rho: Reliability index in [0, 1].
        z: Normalized composite stress score.
        dominant_mode: The component name driving the highest risk.
    """
    ratios = {
        "numerical_error": max(0.0, float(components.numerical_error)) / max(tolerances.numerical_tol, 1e-15),
        "decision_error": max(0.0, float(components.decision_error)) / max(tolerances.decision_tol, 1e-15),
        "assumption_violation": max(0.0, float(components.assumption_violation)) / max(tolerances.assumption_tol, 1e-15),
        "stability_risk": max(0.0, float(components.stability_risk)) / max(tolerances.stability_tol, 1e-15),
    }

    dominant_mode = max(ratios, key=ratios.get)

    if aggregation == "weakest_link":
        z = float(max(ratios.values()))
    elif aggregation == "p_norm":
        vals = np.array(list(ratios.values()), dtype=np.float64)
        z = float(np.sum(vals ** p_norm) ** (1.0 / p_norm))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    if np.isnan(z) or np.isinf(z):
        z = 1e12
    z = max(0.0, z)
    rho = float(1.0 / (1.0 + z ** 2))
    return rho, z, dominant_mode


def predict_failure(rho: float, threshold: float = 0.5) -> bool:
    """
    Predicts computational failure if CRI drops below transition threshold.
    Default threshold is 0.5 (corresponding to z = 1.0).
    """
    return bool(rho < threshold)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes statistical classification metrics:
    AUROC, F1, Precision, Recall, FPR, Brier calibration score.

    Args:
        y_true: Binary ground truth array (1 = failure, 0 = reliable).
        y_scores: Risk score array where HIGHER values indicate higher failure probability.
                  For CRI, risk_score = 1 - rho, or failure predicted if (1 - rho) >= 0.5.
        threshold: Decision threshold for risk score (default 0.5).

    Returns:
        Dict of computed metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if len(y_true) == 0:
