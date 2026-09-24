"""
Foundations Lab  --  External Library Challenge & Falsification Suite
=====================================================================

Research Objective:
  Take a computational system we did not build  --  production SciPy and LAPACK
  linear algebra solvers  --  and test whether our Computational Reliability Index (CRI)
  predicts where the external system becomes unreliable.

Scientific Progression:
  1. Hypothesis:
     External production solvers provide sufficient reliability indications via
     backward residual norms (||b - Ax|| / ||b||) and library exception handling.

  2. Experiment:
     Challenge scipy.linalg.solve and scipy.linalg.cholesky against ill-conditioned
     Hilbert matrices (n = 4 to 15, condition numbers spanning 10^4 to 10^17)
     and perturbed positive definite systems.

  3. Falsification:
     The hypothesis fails. At n >= 12 (kappa >= 10^16), scipy.linalg.solve produces
     forward errors exceeding 100% (reaching 1580% at n=13) while the backward
     residual norm remains at 3.59e-16 (machine precision). The solver does not crash,
     silently delivering corrupted vectors.
     Similarly, scipy.linalg.cholesky produces factors with residual 3.1e-17 at n=13,
     but triangular back-substitution produces 335% forward error.

  4. Revision:
     A computational reliability index cannot rely solely on backward error or
     absence of runtime exceptions. We revise the CRI formulation to incorporate
     condition-based stability risk:
       r_stab = min(1.0, kappa(A) * eps_mach)
     Under this revision, CRI correctly flags failure (rho < 0.5) at n >= 11
     (kappa >= 5.2e14), successfully predicting breakdown.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import scipy.linalg

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capstone.cri import (
    ReliabilityComponents,
    CriticalTolerances,
    compute_composite_cri,
    compute_scalar_cri,
    predict_failure,
)


def run_scipy_solver_challenge(n_range: Tuple[int, int] = (4, 15)) -> Dict:
    """
    Evaluates scipy.linalg.solve on Hilbert matrices across condition numbers.
    Compares naive residual-based reliability against condition-aware CRI.
    """
    eps_mach = np.finfo(np.float64).eps
    tolerances = CriticalTolerances(
