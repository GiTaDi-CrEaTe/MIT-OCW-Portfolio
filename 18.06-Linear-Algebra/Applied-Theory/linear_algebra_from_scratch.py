"""
18.06 Applied Theory  --  Decompositions and Eigenstructure from First Principles
================================================================================

Implements, from scratch, four ideas chained together exactly as they build on
each other in the course:

    Gram-Schmidt CGS & MGS (Pset 7)  -->  QR algorithm eigensolver (Pset 9)
                                     |
                    +----------------+----------------+
                    v                                 v
       From-scratch SVD (Pset 12)          PageRank power iteration (Pset 11)

`numpy` is used strictly as an array container and for basic arithmetic
(dot products, norms). Every *algorithm* -- Gram-Schmidt, the QR iteration,
the SVD construction, power iteration -- is hand-written. `numpy.linalg` is
used ONLY inside the self-tests, as a ground-truth oracle to check this code
against, never as part of the implementation itself.
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# 1. Gram-Schmidt orthogonalization -> QR decomposition
# ---------------------------------------------------------------------------

def qr_gram_schmidt(A: np.ndarray):
    """
    Constructive proof that a matrix A (m x n, independent columns) can be
    written A = QR, where Q has orthonormal columns spanning the same column
    space as A, and R is upper triangular.

    Theory: build Q's columns one at a time. For each new column a_k of A,
    subtract off its projection onto every orthonormal vector already found
    (this removes everything already "explained" by previous directions),
    then normalize what's left. The projection coefficients removed at each
    step are exactly the entries of R -- R records "how much of each new
    column was already covered by earlier directions."

    Complexity: O(mn^2) where A is m x n.
    """
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = Q[:, j] @ A[:, k]     # projection coefficient of a_k onto q_j
            v = v - R[j, k] * Q[:, j]       # strip off that component
        R[k, k] = np.linalg.norm(v)
        if R[k, k] < 1e-12:
            raise ValueError("Columns are not linearly independent (or nearly so).")
        Q[:, k] = v / R[k, k]

    return Q, R


