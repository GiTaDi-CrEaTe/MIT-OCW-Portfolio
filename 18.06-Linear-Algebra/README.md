# 18.06 --  Linear Algebra
*Foundations Lab: Vector Decompositions and Numerical Stability*

---

### Core Question
When does mathematically exact linear algebra become numerically unreliable on floating-point hardware?

### The Mathematical Guarantee
- **Orthogonal Basis (QR):** Every full-rank matrix $A \in \mathbb{R}^{m \times n}$ can be factored as $A = QR$, where $Q$ has orthonormal columns ($Q^T Q = I$) and $R$ is upper triangular.
- **Spectral Theorem:** Every real symmetric matrix $S = S^T$ has real eigenvalues and an orthonormal eigenbasis: $S = V \Lambda V^T$.
- **Singular Value Decomposition (SVD):** Every real matrix $A$ decomposes as $A = U \Sigma V^T$, where singular values are $\sigma_i = \sqrt{\lambda_i(A^TA)}$.
- **Perron-Frobenius Theorem:** A column-stochastic irreducible matrix $M$ has a unique steady-state vector satisfying $M \pi = \pi$.

### What the Code Investigates
[`linear_algebra_from_scratch.py`](./Applied-Theory/linear_algebra_from_scratch.py) implements the core matrix factorizations from first principles using NumPy arrays alone (no `numpy.linalg` for solver routines):
1. **Classical & Modified Gram-Schmidt:** Constructive orthogonalization and upper-triangular coefficient extraction.
2. **Iterative QR Algorithm:** Eigenvalue and eigenvector computation via similarity transformations ($A_{k+1} = R_k Q_k = Q_k^T A_k Q_k$).
3. **From-Scratch SVD:** Derives $\Sigma$ and $V$ from the eigenstructure of $A^TA$, and recovers $U$ via $u_i = \frac{1}{\sigma_i} A v_i$.
4. **PageRank Power Iteration:** Dominant eigenvector computation of the Google Markov transition matrix.

### Empirical Findings & Failure Modes
- **Catastrophic Loss of Orthogonality in CGS:** When matrix condition number $\kappa(A) \ge 10^8$, Classical Gram-Schmidt completely loses orthogonality ($\|Q^T Q - I\|_2 > 0.4$, reaching $3.0$ on a $10 \times 10$ Hilbert matrix), even while the backward residual $\|A - QR\|$ remains $O(\epsilon_{\text{mach}})$. Modified Gram-Schmidt reduces this error by up to $10^7 \times$.
- **Singular Value Obliteration via Normal Equations:** Forming $A^TA$ squares the condition number $\kappa(A^TA) = \kappa(A)^2$. For $\kappa(A) \ge 10^8$, small singular values fall below $\epsilon_{\text{mach}}$ and are wiped out to exact zero.

### Capstone & Cross-Course Connection
Directly powers [Capstone Experiment 1](../capstone/README.md#experiment-1--numerical-stability-classical-vs-modified-gram-schmidt) and [Capstone Experiment 2](../capstone/README.md#experiment-2--svd-via-ata-vs-direct-bidiagonalization). In addition, 18.06's spectral decomposition is reused to solve the stationary distributions in 6.041, and its Jacobian matrix calculus directly drives the backpropagation derivation in 6.036.
