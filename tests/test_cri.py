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
    dataset_b = build_dataset_b()

    assert len(dataset_a) >= 20
    assert len(dataset_b) >= 20

    # Ensure domains in Dataset B are genuinely distinct from Dataset A
    domains_a = {s.domain for s in dataset_a}
    domains_b = {s.domain for s in dataset_b}
    overlap = domains_a.intersection(domains_b)
    assert len(overlap) == 0, f"Held-out dataset B contains domains seen in A: {overlap}"

    # Execute full validation
    val_res = run_holdout_validation()
    m = val_res["metrics"]
    ci = val_res["bootstrap_ci"]

    # Verify quantitative performance targets
    assert m["auroc"] > 0.85, f"Held-out AUROC too low: {m['auroc']}"
    assert m["f1"] > 0.75, f"Held-out F1 too low: {m['f1']}"
    assert m["recall"] > 0.80, f"Held-out Recall too low: {m['recall']}"
    assert m["fpr"] < 0.20, f"Held-out FPR too high: {m['fpr']}"
    assert m["brier_score"] < 0.15, f"Brier score too poor: {m['brier_score']}"

    # Bootstrap intervals must bound point estimates reasonably
    assert ci["auroc_ci"][0] <= m["auroc"] <= ci["auroc_ci"][1] + 1e-6
    assert ci["f1_ci"][0] <= m["f1"] <= ci["f1_ci"][1] + 1e-6


def test_external_library_challenge():
    # Challenge SciPy solver and Cholesky across dimensions 4 to 15
    # On different BLAS/LAPACK platforms, naive accuracy ranges from 58% to 67%
    ext_res = run_external_challenge()

    # The naive residual-only approach fails to detect ill-conditioned breakdown
    assert ext_res["naive_accuracy"] <= 0.70

    # The revised stability-aware CRI accurately flags failure
    assert ext_res["revised_accuracy"] >= 0.90
    assert ext_res["revised_accuracy"] > ext_res["naive_accuracy"] + 0.20

    # Check Cholesky breakdown on high-order Hilbert matrix
    chol_records = ext_res["cholesky_challenge"]["records"]
    high_order_fail = [r for r in chol_records if r["n"] >= 14]
    assert len(high_order_fail) > 0
    for r in high_order_fail:
        assert not r["cholesky_succeeded"] or r["solve_forward_error"] > 0.5


def test_auroc_ties_and_edge_cases():
    # Completely tied scores must yield 0.5 (random guess)
    y_true_tied = np.array([0, 1])
    y_scores_tied = np.array([0.5, 0.5])
    assert compute_auroc(y_true_tied, y_scores_tied) == 0.5

    # Partial ties with known exact mid-rank value: 7/8 = 0.875
    y_true_part = np.array([0, 0, 1, 1])
    y_scores_part = np.array([0.2, 0.5, 0.5, 0.8])
    assert compute_auroc(y_true_part, y_scores_part) == 0.875

    # Single-class edge cases must return 0.5 safely
    assert compute_auroc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) == 0.5
    assert compute_auroc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) == 0.5

    # Inverted ranking must yield 0.0
    y_true_inv = np.array([0, 0, 1, 1])
    y_scores_inv = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_auroc(y_true_inv, y_scores_inv) == 0.0


def test_composite_cri_robustness_and_bounds():
    tolerances = CriticalTolerances(1e-2, 1e-3, 1.0, 0.5)

    # All zero inputs
    zero_comp = ReliabilityComponents(0.0, 0.0, 0.0, 0.0)
    rho_z, z_z, _ = compute_composite_cri(zero_comp, tolerances)
    assert rho_z == 1.0
    assert z_z == 0.0

    # Negative inputs should be clamped safely, not crash or produce negative z
    neg_comp = ReliabilityComponents(-0.5, -0.1, -1.0, -0.2)
    rho_neg, z_neg, _ = compute_composite_cri(neg_comp, tolerances)
    assert rho_neg == 1.0
    assert z_neg == 0.0

    # Huge/infinite inputs
    huge_comp = ReliabilityComponents(1e30, 0.0, 0.0, 0.0)
    rho_huge, z_huge, _ = compute_composite_cri(huge_comp, tolerances)
    assert rho_huge < 1e-10
    assert z_huge > 1e10

    # p-norm aggregation method
    rho_p, z_p, dom_p = compute_composite_cri(
        ReliabilityComponents(numerical_error=0.01),
        tolerances,
        aggregation="p_norm",
        p_norm=4.0,
    )
    assert 0.0 <= rho_p <= 1.0
    assert z_p > 0.0

    # Invalid aggregation raises ValueError
    with pytest.raises(ValueError):
        compute_composite_cri(zero_comp, tolerances, aggregation="invalid_mode")


def test_scipy_external_warning_metadata():
    records = run_scipy_solver_challenge((10, 14))["records"]
    assert len(records) == 5

    # Check warning and exception metadata fields exist
    for r in records:
        assert "warning_raised" in r
        assert "warning_type" in r
        assert "warning_msg" in r
        assert "exception_raised" in r

    # At n >= 12, SciPy emits LinAlgWarning
    high_n = [r for r in records if r["n"] >= 12]
    for r in high_n:
        assert r["warning_raised"] is True
        assert r["warning_type"] == "LinAlgWarning"


def test_cri_negative_error_raises():
    # Negative error magnitudes must be rejected
    from capstone.cri import compute_scalar_cri
    with pytest.raises(ValueError):
        compute_scalar_cri(-1e-5, 1.0)
    with pytest.raises(ValueError):
        compute_scalar_cri(0.5, -0.1)


def test_cri_zero_error_yields_one():
    # Zero error must yield exact 1.0 reliability
    from capstone.cri import compute_scalar_cri
    assert compute_scalar_cri(0.0, 1.0) == 1.0
    assert compute_scalar_cri(0.0, 1e-12) == 1.0


def test_stability_risk_saturation():
    # Extremely ill-conditioned matrices must saturate risk to 1.0
    from capstone.cri_external import compute_stability_risk
    assert compute_stability_risk(1e20) == 1.0
    assert compute_stability_risk(1e30) == 1.0
