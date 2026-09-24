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
