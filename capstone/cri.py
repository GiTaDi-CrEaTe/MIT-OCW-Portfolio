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


def compute_scalar_cri(error: Union[float, int], crit_threshold: Union[float, int]) -> float:
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
        raise ValueError("Cannot compute metrics on empty arrays.")

    y_pred = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Brier calibration score: mean squared difference between predicted probability and label
    brier_score = float(np.mean((y_scores - y_true) ** 2))

    # AUROC calculation via Mann-Whitney U test with exact mid-rank tie handling
    auroc = compute_auroc(y_true, y_scores)

    return {
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "brier_score": brier_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sample_count": len(y_true),
    }


def _rank_data_average(scores: np.ndarray) -> np.ndarray:
    """
    Computes 1-based ranks with average ties in pure NumPy.
    Assigns the mid-rank to duplicate values.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Computes Area Under ROC curve using Mann-Whitney U statistic with mid-rank tie handling.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined when one class is completely missing

    ranks = _rank_data_average(y_scores)
    rank_sum_pos = np.sum(ranks[y_true == 1])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(np.clip(auroc, 0.0, 1.0))


def bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes bootstrap confidence intervals for AUROC and F1 score.
    Addresses small sample size uncertainty transparently.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    f1s = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_scores = y_scores[indices]

        # Ensure both classes present in resample
        if len(np.unique(sample_true)) < 2:
            continue

        metrics = compute_classification_metrics(sample_true, sample_scores)
        aurocs.append(metrics["auroc"])
        f1s.append(metrics["f1"])

    if len(aurocs) == 0:
        return {"auroc_ci": (0.0, 1.0), "f1_ci": (0.0, 1.0)}

    lower_p = 100 * (alpha / 2.0)
    upper_p = 100 * (1.0 - alpha / 2.0)

    auroc_low, auroc_high = np.percentile(aurocs, [lower_p, upper_p])
    f1_low, f1_high = np.percentile(f1s, [lower_p, upper_p])

    return {
        "auroc_ci": (float(auroc_low), float(auroc_high)),
        "f1_ci": (float(f1_low), float(f1_high)),
    }
