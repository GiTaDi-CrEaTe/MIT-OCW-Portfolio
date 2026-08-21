# 18.06 — Problem Set Roadmap

## Part 1 — Solving Linear Systems

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 1 | Vectors, matrix multiplication as combinations | Row picture vs. column picture of Ax = b | The "column picture" reframing (Ax is a combination of A's columns) is the single most useful mental model in the whole course. |
| Pset 2 | Elimination, LU decomposition | Gaussian elimination as matrix factorization | Elimination is just recording row operations as a lower-triangular matrix — obvious in hindsight, not obvious on first pass. |
| Pset 3 | Vector spaces, column space, null space | Rank, dimension, basis | Null space computation is where "solve Ax=0" stops being mechanical and starts being structural. |
| Pset 4 | The four fundamental subspaces | Row space, column space, null space, left null space, and how they relate via rank | Strang's "Fundamental Theorem of Linear Algebra" — orthogonality between row space and null space — is used directly in the least-squares derivation later. |

## Part 2 — Orthogonality and Projections

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 5 | Orthogonality, projections onto a line/subspace | Projection matrices, P = A(AᵀA)⁻¹Aᵀ | This is exactly the linear regression solution derivation reused in 6.036. |
| Pset 6 | Least squares | Normal equations, AᵀAx̂ = Aᵀb | Directly implemented and compared against gradient descent in the 6.036 Applied-Theory script. |
| Pset 7 | Gram-Schmidt and QR decomposition | Orthonormalizing a basis constructively | Implemented from scratch in this folder's Applied-Theory script — the constructive proof *is* the algorithm. |

## Part 3 — Eigenvalues and Eigenvectors

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 8 | Determinants | Cofactor expansion, properties, connection to invertibility | Mostly a tool-building pset; determinants matter here mainly because det(A - λI) = 0 defines eigenvalues. |
| Pset 9 | Eigenvalues and eigenvectors | Characteristic polynomial, diagonalization | The QR algorithm implemented in Applied-Theory sidesteps computing the characteristic polynomial directly — a much more numerically stable approach, and the one real solvers actually use. |
| Pset 10 | Symmetric matrices, positive definiteness | Spectral theorem, quadratic forms | The spectral theorem (real symmetric matrices have real eigenvalues and orthogonal eigenvectors) is the theoretical backbone of the SVD. |
| Pset 11 | Markov matrices and steady states | Perron-Frobenius intuition, steady-state as eigenvector for λ=1 | This pset is implemented directly as the PageRank power-iteration demo in Applied-Theory. |

## Part 4 — The Singular Value Decomposition

| Pset | Topic | Key techniques | Notes |
|---|---|---|---|
| Pset 12 | Singular Value Decomposition | A = UΣVᵀ, relation to AᵀA and AAᵀ eigenstructure | The capstone idea of the course: every matrix, not just square symmetric ones, has an eigenvector-like decomposition. Implemented from scratch by eigendecomposing AᵀA. |

## Applied-Theory connection

`Applied-Theory/linear_algebra_from_scratch.py` chains Psets 7, 9, 11, and 12 together: Gram-Schmidt (Pset 7) is used to build the QR algorithm (Pset 9), which is then reused as the eigen-engine for the from-scratch SVD (Pset 12) and — with a small change of matrix — for the PageRank steady-state computation (Pset 11).
