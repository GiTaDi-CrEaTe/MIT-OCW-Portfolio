"""
Tests for Capstone Experiments
"""

import numpy as np
import pytest
from capstone.numerical_stability import classical_gram_schmidt, modified_gram_schmidt, evaluate_qr_accuracy, generate_controlled_matrix
from capstone.svd_investigation import svd_via_ata
from capstone.search_efficiency import GridMap, run_astar, h_manhattan, h_zero
from capstone.gradient_precision import MiniMLP
from capstone.model_misspecification import generate_regime_switching_stream, run_bayesian_iid_updating
from capstone.cross_course_synthesis import (
    verify_discrete_vs_continuous_precision,
    synthesize_eigensolver_and_markov_chain,
    evaluate_computational_reliability_index,
)


def test_capstone_cgs_vs_mgs_loss_of_orthogonality():
    # At high condition number, CGS must have significantly worse orthogonality than MGS
    A = generate_controlled_matrix(20, 10, cond_number=1e10, seed=42)
    Q_cgs, R_cgs = classical_gram_schmidt(A)
    Q_mgs, R_mgs = modified_gram_schmidt(A)

    eval_cgs = evaluate_qr_accuracy(A, Q_cgs, R_cgs)
    eval_mgs = evaluate_qr_accuracy(A, Q_mgs, R_mgs)

    assert eval_cgs["orthogonality_error"] > 0.1  # CGS breaks down
    assert eval_mgs["orthogonality_error"] < 1e-4  # MGS remains far better conditioned
    assert eval_cgs["orthogonality_error"] > 1e4 * eval_mgs["orthogonality_error"]


def test_capstone_svd_squaring_breakdown():
    # Matrix with condition number 10^9: sigma_min = 1e-9
    # In A^T A, sigma_min^2 = 1e-18 < eps_mach (2.22e-16).
    # Forming A^T A wipes out or corrupts the smallest singular value.
    m, n = 15, 8
    rng = np.random.default_rng(42)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.geomspace(1.0, 1e-9, num=n)
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, s)
    A = U @ Sigma @ V.T

    _, s_ata, _ = svd_via_ata(A)
    s_dir = np.linalg.svd(A, compute_uv=False)

    rel_err_ata = abs(s_ata[-1] - s[-1]) / s[-1]
    rel_err_dir = abs(s_dir[-1] - s[-1]) / s[-1]

    # SVD via A^T A completely breaks down on the smallest singular value
    assert rel_err_ata > 0.5  # Swamped by roundoff error (clamped to 0 or heavily corrupted)
    # Direct SVD (bidiagonalization) recovers it reliably
    assert rel_err_dir < 1e-5
    # The normal equations error is orders of magnitude worse than direct SVD
    assert rel_err_ata > 1e3 * rel_err_dir


def test_capstone_astar_optimality():
    grid = GridMap(20, 20, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (19, 19)

    dij_cost, dij_nodes = run_astar(grid, start, goal, h_zero, tie_break=False)
    a_cost, a_nodes = run_astar(grid, start, goal, h_manhattan, tie_break=False)
    tb_cost, tb_nodes = run_astar(grid, start, goal, h_manhattan, tie_break=True)

    if dij_cost is not None:
        assert np.isclose(dij_cost, a_cost)
        assert np.isclose(dij_cost, tb_cost)
        assert a_nodes <= dij_nodes
        assert tb_nodes <= a_nodes


def test_capstone_cross_synthesis_discrete_precision():
    res = verify_discrete_vs_continuous_precision()
    assert res["discrete_error"] == 0.0
    assert res["float64_cancellation_error"] == 1.0
    assert res["rsa_reconstruction_error"] == 0.0


def test_capstone_markov_spectral_synthesis():
    res = synthesize_eigensolver_and_markov_chain()
    assert res["max_spectral_vs_empirical_gap"] < 0.02


def test_capstone_computational_reliability_index():
    cri_data = evaluate_computational_reliability_index()
    assert len(cri_data) == 6
    for domain, res in cri_data.items():
        assert res["safe_cri"] > 0.8, f"{domain} safe CRI too low: {res['safe_cri']}"
        assert res["fail_cri"] < 0.2, f"{domain} fail CRI too high: {res['fail_cri']}"
