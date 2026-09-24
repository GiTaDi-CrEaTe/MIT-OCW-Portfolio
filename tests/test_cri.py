"""
Unit Tests for Computational Reliability Index (CRI),
Holdout Validation, and External Library Challenge
"""

import numpy as np
import pytest

from capstone.cri import (
    ReliabilityComponents,
    CriticalTolerances,
    compute_scalar_cri,
    compute_composite_cri,
    predict_failure,
    compute_classification_metrics,
    compute_auroc,
    bootstrap_ci,
)
from capstone.cri_validation import (
    build_dataset_a,
    build_dataset_b,
    calibrate_cri_tolerances,
    run_holdout_validation,
)
from capstone.cri_external import (
    run_scipy_solver_challenge,
    run_scipy_cholesky_challenge,
    run_external_challenge,
)


def test_cri_monotonicity_and_bounds():
    # Boundary tests
    assert compute_scalar_cri(0.0, 1.0) == 1.0
    assert compute_scalar_cri(1.0, 1.0) == 0.5
    assert compute_scalar_cri(100.0, 1.0) < 0.001

    # Monotonicity test
    errs = np.linspace(0.0, 10.0, 50)
    rhos = [compute_scalar_cri(e, 1.0) for e in errs]
    for i in range(len(rhos) - 1):
        assert rhos[i] >= rhos[i + 1]
        assert 0.0 <= rhos[i] <= 1.0

    # Negative error handled gracefully or invalid threshold rejected
    assert compute_scalar_cri(-1.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        compute_scalar_cri(1.0, 0.0)


