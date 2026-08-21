# 18.06 — Linear Algebra

**MIT OCW subject:** 18.06, Department of Mathematics (Gilbert Strang's course).

## What this course is

18.06 treats linear algebra computationally and geometrically rather than purely axiomatically: the four fundamental subspaces, elimination and LU decomposition, orthogonality and projections, determinants, eigenvalues/eigenvectors, and the Singular Value Decomposition as the course's capstone idea — "the right basis for the matrix." The course's central habit of mind is to always ask *what does this operation do to space*, not just how to compute it.

## Why it matters for this portfolio

Linear algebra is the shared vocabulary underneath both the algorithms course and the ML course in this repository:
- 6.036's linear regression is literally a least-squares projection problem — Applied-Theory subspace projections done in a supervised-learning wrapper.
- PageRank-style ranking algorithms and Markov chain steady states (touched on again in 6.041) are eigenvector problems.
- The SVD is the tool that makes precise the idea of "directions of maximum variance," which resurfaces implicitly in any dimensionality-reduction or feature-learning context.

## What I focused on

The `Applied-Theory/` script in this folder builds three of the course's central algorithms entirely from scratch, verified against `numpy.linalg`'s black-box routines (used only as a correctness oracle, never as the implementation):

1. **QR decomposition via Gram-Schmidt** — the constructive proof that any matrix with independent columns has an orthonormal basis for its column space.
2. **Eigenvalues/eigenvectors via the QR algorithm** — an iterative method that repeatedly applies QR decomposition to converge to a matrix's eigenstructure, used here as the engine for a from-scratch SVD.
3. **PageRank as a power-iteration eigenvector problem** — the steady-state vector of a Markov transition matrix is the eigenvector for eigenvalue 1, computed here via repeated matrix-vector multiplication rather than direct linear solving.

## Folder contents

- [`Psets/pset_roadmap.md`](./Psets/pset_roadmap.md) — topic-by-topic syllabus breakdown.
- [`Applied-Theory/linear_algebra_from_scratch.py`](./Applied-Theory/linear_algebra_from_scratch.py) — Gram-Schmidt QR, QR-algorithm eigensolver, from-scratch SVD, and power-iteration PageRank, each checked against NumPy's reference implementation.
