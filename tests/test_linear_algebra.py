"""
Tests for 18.06 Linear Algebra from Scratch
"""

import numpy as np
import pytest
from conftest import load_course_module

la = load_course_module("la_scratch", "18.06-Linear-Algebra/Applied-Theory/linear_algebra_from_scratch.py")


def test_qr_gram_schmidt_reconstruction_and_orthogonality():
    rng = np.random.default_rng(1806)
    A = rng.standard_normal((6, 4))

    qr_fn = getattr(la, "qr_gram_schmidt", getattr(la, "qr_factorization", None))
    Q, R = qr_fn(A)

    recon_err = np.max(np.abs(Q @ R - A))
    assert recon_err < 1e-10

    ortho_err = np.max(np.abs(Q.T @ Q - np.eye(4)))
    assert ortho_err < 1e-10

    assert np.allclose(np.tril(R, -1), 0.0, atol=1e-12)


def test_qr_algorithm_eigensolver():
    rng = np.random.default_rng(42)
    S = rng.standard_normal((4, 4))
    S = S @ S.T

    eig_fn = getattr(la, "eig_qr_algorithm", getattr(la, "qr_algorithm", None))
    my_vals, my_vecs = eig_fn(S)

    ref_vals, _ = np.linalg.eigh(S)

    my_sorted = np.sort(my_vals)
    ref_sorted = np.sort(ref_vals)
    assert np.allclose(my_sorted, ref_sorted, atol=1e-4)


def test_svd_from_scratch():
    rng = np.random.default_rng(77)
    A = rng.standard_normal((5, 3))

    svd_fn = getattr(la, "svd_from_scratch", getattr(la, "svd", None))
    U, s, Vt = svd_fn(A)

    ref_s = np.linalg.svd(A, compute_uv=False)
    s_arr = np.array(s) if not isinstance(s, np.ndarray) or s.ndim == 1 else np.diag(s)
    assert np.allclose(np.sort(s_arr)[::-1][:len(ref_s)], np.sort(ref_s)[::-1], atol=1e-4)


def test_pagerank_or_lu():
    if hasattr(la, "pagerank_power_iteration"):
        link_matrix = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
        ], dtype=float)
        pi = la.pagerank_power_iteration(link_matrix, damping=0.85, iterations=300)
        assert len(pi) == 4
        assert np.isclose(np.sum(pi), 1.0, atol=1e-6)
        assert np.all(pi > 0)
    elif hasattr(la, "lu_decomposition"):
        A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
        P, L, U = la.lu_decomposition(A)
        assert np.allclose(P @ A, L @ U, atol=1e-10)



