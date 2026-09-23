"""
Tests for Flagship Capstone Experiments
"""

import numpy as np
import pytest
from capstone.numerical_stability import classical_gram_schmidt, modified_gram_schmidt, evaluate_qr_accuracy, generate_controlled_matrix
from capstone.svd_investigation import svd_via_ata
from capstone.search_efficiency import GridMap, run_astar, h_manhattan, h_zero
from capstone.gradient_precision import MiniMLP
from capstone.model_misspecification import generate_regime_switching_stream, run_bayesian_iid_updating
from capstone.cross_course_synthesis import verify_discrete_vs_continuous_precision, synthesize_eigensolver_and_markov_chain


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
    # Matrix with condition number 10^9
    m, n = 15, 8
    rng = np.random.default_rng(42)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.geomspace(1.0, 1e-9, num=n)
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, s)
    A = U @ Sigma @ V.T

    _, s_ata, _ = svd_via_ata(A)
    # The smallest singular value is 1e-9; in A^TA its square is 1e-18 < eps_mach, so it must be 0
    assert s_ata[-1] == 0.0


def test_capstone_astar_optimality():
    grid = GridMap(20, 20, obstacle_density=0.15, seed=42)
    start = (0, 0)
    goal = (19, 19)

    dij_cost, dij_nodes = run_astar(grid, start, goal, h_zero)
    a_cost, a_nodes = run_astar(grid, start, goal, h_manhattan)

    if dij_cost is not None:
        assert np.isclose(dij_cost, a_cost)
        assert a_nodes <= dij_nodes


def test_capstone_cross_synthesis_discrete_precision():
    res = verify_discrete_vs_continuous_precision()
    assert res["discrete_error"] == 0.0
    assert res["float64_cancellation_error"] == 1.0


def test_capstone_markov_spectral_synthesis():
    res = synthesize_eigensolver_and_markov_chain()
    assert res["max_spectral_vs_empirical_gap"] < 0.02

