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


def qr_modified_gram_schmidt(A: np.ndarray):
    """
    Modified Gram-Schmidt (MGS) QR factorization -- O(mn^2) operations.

    Unlike Classical GS (above), which projects each new column against the
    ORIGINAL columns of A, MGS updates the working vector sequentially after
    each projection step. In exact arithmetic the two methods are identical;
    in floating-point they differ dramatically:

        CGS orthogonality error: O(kappa(A)^2 * eps_mach)
        MGS orthogonality error: O(kappa(A)   * eps_mach)

    This difference is documented experimentally in capstone/numerical_stability.py
    and in the lab notebook (Log Entry 1).

    Complexity: O(mn^2) where A is m x n.
    """
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()

    for k in range(n):
        R[k, k] = np.linalg.norm(V[:, k])
        if R[k, k] < 1e-12:
            raise ValueError("Columns are not linearly independent (or nearly so).")
        Q[:, k] = V[:, k] / R[k, k]
        for j in range(k + 1, n):
            R[k, j] = Q[:, k] @ V[:, j]
            V[:, j] = V[:, j] - R[k, j] * Q[:, k]

    return Q, R


# ---------------------------------------------------------------------------
# 2. QR algorithm  --  iterative eigenvalue/eigenvector solver built on top of
#    the Gram-Schmidt QR decomposition above.
# ---------------------------------------------------------------------------

def eig_qr_algorithm(A: np.ndarray, iterations: int = 500):
    """
    Computes eigenvalues and eigenvectors of a symmetric matrix A using the
    (unshifted) QR algorithm.

    Theory: given A_0 = A, repeat A_{k+1} = R_k Q_k where A_k = Q_k R_k is a
    QR decomposition. Each A_{k+1} is similar to A_k (A_{k+1} = Q_k^T A_k Q_k),
    so all A_k share the same eigenvalues. For a symmetric matrix with
    distinct eigenvalues, this sequence provably converges to a diagonal
    matrix whose entries ARE the eigenvalues, and the accumulated product of
    all the Q_k's converges to a matrix whose columns are the corresponding
    eigenvectors. This works because repeated QR factorization is secretly
    performing simultaneous power iteration on all of A's eigenvectors at once.

    Restricted here to symmetric A, matching the course's emphasis (Pset 10)
    on the spectral theorem, and because convergence to a strictly diagonal
    (not just upper-triangular) form is guaranteed in that case.
    """
    A = A.astype(float)
    n = A.shape[0]
    Ak = A.copy()
    Q_total = np.eye(n)

    for _ in range(iterations):
        Q, R = qr_gram_schmidt(Ak)
        Ak = R @ Q
        Q_total = Q_total @ Q

    eigenvalues = np.diag(Ak).copy()
    eigenvectors = Q_total
    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# 3. Singular Value Decomposition from scratch, built on the eigensolver above
# ---------------------------------------------------------------------------

def svd_from_scratch(A: np.ndarray):
    """
    Computes A = U * Sigma * V^T from scratch.

    Theory (Pset 12): A^T A is symmetric and positive semi-definite, so it has
    real, non-negative eigenvalues and an orthonormal eigenbasis. Those
    eigenvalues are the squared singular values of A, and the eigenvectors ARE
    V. Once V and the singular values sigma_i are known, the corresponding
    left singular vectors are recovered via u_i = (1/sigma_i) * A * v_i --
    directly from the definition A v_i = sigma_i u_i.

    Complexity: O(n^3 * iterations) for the eigensolver, plus O(mn) for U recovery.
    """
    A = A.astype(float)
    m, n = A.shape

    AtA = A.T @ A                                    # n x n, symmetric PSD
    eigvals, V = eig_qr_algorithm(AtA, iterations=800)

    # Eigenvalues of AtA can emerge with tiny numerical noise; clip and sort.
    eigvals = np.clip(eigvals, 0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = V[:, order]

    singular_values = np.sqrt(eigvals)

    U = np.zeros((m, n))
    for i in range(n):
        if singular_values[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / singular_values[i]
        else:
            U[:, i] = 0.0  # degenerate direction; contributes nothing to A

    return U, singular_values, V.T


# ---------------------------------------------------------------------------
# 4. PageRank via power iteration  --  the steady-state eigenvector (eigenvalue 1)
#    of a Markov transition matrix, computed WITHOUT ever building an
#    eigensolver for the non-symmetric transition matrix directly.
# ---------------------------------------------------------------------------

def pagerank_power_iteration(link_matrix: np.ndarray, damping: float = 0.85,
                              iterations: int = 200):
    """
    Theory (Pset 11): a Markov transition matrix M (columns sum to 1, entry
    M[i, j] = probability of moving from page j to page i) has a steady-state
    distribution pi satisfying M pi = pi -- i.e. pi is the eigenvector of M
    for eigenvalue 1. Rather than solving this as a linear system, power
    iteration exploits the fact that repeatedly applying M to *any* starting
    distribution converges to the dominant eigenvector, because eigenvalue 1
    is the largest eigenvalue of a (damped, irreducible) transition matrix --
    this is a direct consequence of the Perron-Frobenius theorem, which the
    course gestures at when introducing Markov matrices.

    `damping` mixes in a uniform "random surfer" term, guaranteeing the chain
    is irreducible and aperiodic (so convergence is guaranteed regardless of
    the raw link structure) -- this is the actual PageRank formulation.

    Complexity: O(n^2 * iterations) -- one matrix-vector multiply per iteration.
    """
    n = link_matrix.shape[0]

    # Column-normalize so each column is a probability distribution over
    # where a random surfer on that page goes next.
    col_sums = link_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1  # avoid division by zero for dangling pages
    M = link_matrix / col_sums

    teleport = np.ones((n, n)) / n
    google_matrix = damping * M + (1 - damping) * teleport

    pi = np.ones(n) / n
    for _ in range(iterations):
        pi = google_matrix @ pi
        pi = pi / pi.sum()  # renormalize (guards against numerical drift)

    return pi


# ---------------------------------------------------------------------------
# 5. Self-verification against numpy.linalg (used ONLY as a ground-truth oracle)
# ---------------------------------------------------------------------------

