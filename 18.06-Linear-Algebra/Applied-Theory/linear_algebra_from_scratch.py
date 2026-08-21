"""
18.06 Applied Theory — Decompositions and Eigenstructure from First Principles
================================================================================

Implements, from scratch, four ideas chained together exactly as they build on
each other in the course:

    Gram-Schmidt (Pset 7)  -->  QR algorithm eigensolver (Pset 9)
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


# ---------------------------------------------------------------------------
# 2. QR algorithm — iterative eigenvalue/eigenvector solver built on top of
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
# 4. PageRank via power iteration — the steady-state eigenvector (eigenvalue 1)
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

def _self_test():
    rng = np.random.default_rng(1806)

    print("=" * 70)
    print("SELF-TEST 1: Gram-Schmidt QR reconstructs A and yields Q^T Q = I")
    print("=" * 70)
    A = rng.standard_normal((6, 4))
    Q, R = qr_gram_schmidt(A)
    reconstruction_error = np.max(np.abs(Q @ R - A))
    orthonormality_error = np.max(np.abs(Q.T @ Q - np.eye(4)))
    print(f"max |A - QR|        = {reconstruction_error:.2e}")
    print(f"max |Q^T Q - I|     = {orthonormality_error:.2e}")
    assert reconstruction_error < 1e-8 and orthonormality_error < 1e-8
    print("PASSED\n")

    print("=" * 70)
    print("SELF-TEST 2: QR-algorithm eigenvalues match numpy.linalg.eigh")
    print("=" * 70)
    S = rng.standard_normal((5, 5))
    S = S @ S.T  # force symmetric positive semi-definite
    my_vals, my_vecs = eig_qr_algorithm(S)
    ref_vals, ref_vecs = np.linalg.eigh(S)
    my_sorted = np.sort(my_vals)
    ref_sorted = np.sort(ref_vals)
    eigval_error = np.max(np.abs(my_sorted - ref_sorted))
    print(f"My eigenvalues (sorted):     {my_sorted}")
    print(f"NumPy eigenvalues (sorted):  {ref_sorted}")
    print(f"max |eigenvalue difference| = {eigval_error:.2e}")
    assert eigval_error < 1e-6
    print("PASSED\n")

    print("=" * 70)
    print("SELF-TEST 3: From-scratch SVD reconstructs A and matches singular values")
    print("=" * 70)
    A2 = rng.standard_normal((5, 3))
    U, s, Vt = svd_from_scratch(A2)
    reconstruction = U @ np.diag(s) @ Vt
    recon_error = np.max(np.abs(reconstruction - A2))
    ref_s = np.linalg.svd(A2, compute_uv=False)
    s_error = np.max(np.abs(np.sort(s) - np.sort(ref_s)))
    print(f"My singular values:    {s}")
    print(f"NumPy singular values: {ref_s}")
    print(f"max |A - U*Sigma*V^T| = {recon_error:.2e}")
    print(f"max |singular value difference| = {s_error:.2e}")
    assert recon_error < 1e-6 and s_error < 1e-6
    print("PASSED\n")

    print("=" * 70)
    print("SELF-TEST 4: PageRank power iteration matches direct eigen-solve")
    print("=" * 70)
    # A small 5-page link graph, link_matrix[i, j] = 1 if page j links to page i
    link_matrix = np.array([
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=float)

    pi_power = pagerank_power_iteration(link_matrix, damping=0.85, iterations=500)

    # Ground truth: build the same Google matrix and solve directly for the
    # eigenvector of eigenvalue 1 using numpy's general eigensolver.
    n = link_matrix.shape[0]
    col_sums = link_matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    M = link_matrix / col_sums
    google_matrix = 0.85 * M + 0.15 * np.ones((n, n)) / n
    eigvals, eigvecs = np.linalg.eig(google_matrix)
    idx = np.argmin(np.abs(eigvals - 1))
    pi_direct = np.real(eigvecs[:, idx])
    pi_direct = pi_direct / pi_direct.sum()

    print(f"PageRank via power iteration: {pi_power}")
    print(f"PageRank via direct eig-solve: {pi_direct}")
    diff = np.max(np.abs(np.sort(pi_power) - np.sort(pi_direct)))
    print(f"max sorted-vector difference: {diff:.2e}")
    assert diff < 1e-4
    print("PASSED\n")

    print("All self-tests passed. Gram-Schmidt -> QR-algorithm eigensolver ->")
    print("SVD -> PageRank chain is internally consistent and matches NumPy's")
    print("reference LAPACK-backed routines.")


if __name__ == "__main__":
    _self_test()
