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


