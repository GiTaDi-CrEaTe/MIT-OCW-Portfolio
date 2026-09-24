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


def test_composite_cri_weakest_link():
    tolerances = CriticalTolerances(
        numerical_tol=0.01,
        decision_tol=0.001,
        assumption_tol=1.0,
        stability_tol=0.5,
    )

    # All safe: rho should be near 1.0
    safe_comp = ReliabilityComponents(
        numerical_error=1e-5,
        decision_error=1e-6,
        assumption_violation=0.01,
        stability_risk=0.01,
    )
    rho_safe, z_safe, dom_safe = compute_composite_cri(safe_comp, tolerances)
    assert rho_safe > 0.99
    assert z_safe < 0.1
    assert not predict_failure(rho_safe)

    # Single catastrophic component (weakest link):
    # Numerical error is tiny, but decision error is 10x its threshold
    weak_link_comp = ReliabilityComponents(
        numerical_error=1e-6,
        decision_error=0.01,  # 10x decision_tol
        assumption_violation=0.0,
        stability_risk=0.0,
    )
    rho_fail, z_fail, dom_fail = compute_composite_cri(weak_link_comp, tolerances)
    assert dom_fail == "decision_error"
    assert z_fail == 10.0
    assert rho_fail < 0.02
    assert predict_failure(rho_fail)


def test_classification_metrics_and_auroc():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    m = compute_classification_metrics(y_true, y_scores, threshold=0.5)
    assert m["auroc"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    assert m["tp"] == 3
    assert m["tn"] == 3

    # Test imperfect classification
    y_scores_imperfect = np.array([0.1, 0.6, 0.2, 0.4, 0.8, 0.9])
    m_imp = compute_classification_metrics(y_true, y_scores_imperfect, threshold=0.5)
    assert 0.5 < m_imp["auroc"] < 1.0
    assert 0.0 < m_imp["f1"] < 1.0
    assert m_imp["fp"] == 1
    assert m_imp["fn"] == 1


def test_bootstrap_confidence_intervals():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.95])

    ci = bootstrap_ci(y_true, y_scores, n_bootstraps=200, seed=42)
    assert "auroc_ci" in ci
    assert "f1_ci" in ci

    auroc_low, auroc_high = ci["auroc_ci"]
    f1_low, f1_high = ci["f1_ci"]

    assert 0.0 <= auroc_low <= auroc_high <= 1.0
    assert 0.0 <= f1_low <= f1_high <= 1.0
    assert auroc_high >= 0.95


def test_holdout_validation_pipeline():
    dataset_a = build_dataset_a()
