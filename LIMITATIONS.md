# Computational Limitations and Theoretical Scope

An explicit, honest accounting of what this repository implements, what it deliberately omits, and why educational from-scratch implementations diverge from production systems.

---

## 1. Philosophical Scope
This repository is an **experimental laboratory in algorithmic foundations**, not a drop-in replacement for production software. Every line of code was written to expose the mathematical mechanics beneath everyday computing abstractions. 

Production libraries (such as LAPACK, BLAS, PyTorch, OpenSSL, and SciPy) contain decades of hardware-specific tuning, vectorization, cache blocking, and numerical safeguards. Below is an inventory of the deliberate trade-offs made in this codebase.

---

## 2. Cryptography & Discrete Mathematics (6.042)

### Textbook RSA vs. Production Public-Key Cryptography
- **No Padding (Textbook RSA Vulnerability):**
  Our implementation computes $c = m^e \pmod n$ directly. This "textbook RSA" is deterministic and multiplicatively homomorphic: $E(m_1) E(m_2) = E(m_1 m_2) \pmod n$. In production, RSA must always use randomized Optimal Asymmetric Encryption Padding (**OAEP**, RFC 8017) to prevent chosen-ciphertext attacks, Coppersmith lattice attacks, and Bleichenbacher padding-oracle attacks.
- **Key Sizes:**
  Interactive tests and self-checks use 256-bit to 512-bit moduli so tests finish in milliseconds. Modern security standards (NIST SP 800-57) require at least **2048-bit to 4096-bit** keys.
- **Side-Channel Vulnerability:**
  Python integers have arbitrary bit-width and variable-time arithmetic. Our modular exponentiation (`mod_pow`) and BigInt multiplication do not run in constant time. A real-world adversary could extract private exponent $d$ through cache-timing or power-analysis side-channel attacks.
- **Entropy Source:**
  Prime candidate generation uses Python's standard `random` module (Mersenne Twister), which is **not cryptographically secure**. Production systems require an OS-backed entropy collector (`secrets` or `/dev/urandom`).

---

## 3. Linear Algebra (18.06)

### Gram-Schmidt vs. Householder Reflections
- Classical Gram-Schmidt (CGS) and Modified Gram-Schmidt (MGS) form an $O(mn^2)$ constructive orthogonal basis. As shown in [Capstone Experiment 1](capstone/README.md#experiment-1--numerical-stability-classical-vs-modified-gram-schmidt), MGS is significantly more stable than CGS, but still yields orthogonality loss $\|Q^T Q - I\|_2 = O(\kappa(A) \epsilon_{\text{mach}})$.
- Production QR factorizations (LAPACK `dgeqrf`) use **Householder elementary reflectors**, which guarantee $\|Q^T Q - I\|_2 \le c \cdot \epsilon_{\text{mach}}$ unconditionally, regardless of the matrix condition number $\kappa(A)$.

### Eigendecomposition: Unshifted QR Algorithm vs. Francis QR Step
- Our eigensolver runs the basic iterative QR algorithm ($A_{k+1} = R_k Q_k$) on symmetric matrices. It requires $O(n^3)$ operations per iteration and converges slowly when eigenvalues are clustered.
- Production solvers (LAPACK `dsyev`) first reduce dense symmetric matrices to **tridiagonal form** via Householder transformations in $O(n^3)$ once, and then apply the **implicitly shifted QR algorithm** (Wilkinson shift / Francis double-shift) which converges cubically, requiring only $O(n)$ to $O(n^2)$ operations per iteration.

### SVD via $A^T A$ vs. Golub-Kahan Bidiagonalization
- Our pedagogical SVD forms $A^T A$, computes its eigenvalues, and derives singular values $\sigma_i = \sqrt{\lambda_i}$.
- As proven in [Capstone Experiment 2](capstone/README.md#experiment-2--svd-via-ata-vs-direct-bidiagonalization), forming $A^T A$ squares the condition number: $\kappa(A^T A) = \kappa(A)^2$. Any singular value smaller than $\sqrt{\epsilon_{\text{mach}}} \sigma_1 \approx 1.5 \times 10^{-8} \sigma_1$ is lost to machine roundoff.
- Production SVDs (LAPACK `dgesdd`) avoid forming $A^T A$ entirely by bidiagonalizing $A$ directly using Householder reflections, preserving singular values down to $\epsilon_{\text{mach}} \sigma_1 \approx 2.2 \times 10^{-16} \sigma_1$.

---

## 4. Algorithms & Data Structures (6.006)

### Tree Structures & Cache Hierarchy
- Our AVL tree is a node-and-pointer reference implementation in pure Python. While maintaining the theoretical $O(\log n)$ height balance invariant, each node is a heap-allocated Python object with `__slots__` or dictionary overhead.
- Production systems rely on **B-Trees, B+ Trees, or cache-oblivious search trees** that pack hundreds of keys per cache line to minimize hardware cache misses and page faults.
- Python recursion limits (`sys.setrecursionlimit`) restrict deep tree traversals unless implemented iteratively.

### Shortest Paths: Binary Heaps vs. Fibonacci Heaps
- Dijkstra's algorithm uses Python's `heapq` (a binary min-heap) with a running time of $O((V + E) \log V)$.
- Theoretical literature frequently cites **Fibonacci heaps**, which achieve $O(E + V \log V)$ by making `decrease-key` amortized $O(1)$. However, in practice, Fibonacci heaps suffer from massive constant factors, complex pointer manipulation, and poor memory locality, which is why binary or $d$-ary heaps remain preferred in real engines.

---

## 5. Probability & Systems Analysis (6.041)

### Conjugate Models vs. General Posterior Sampling
- Our Bayesian updating implementations rely on exact **conjugate priors** (Beta-Binomial, Normal-Normal). Conjugacy yields closed-form update formulas where parameters update analytically.
- Real-world Bayesian inference involves non-conjugate likelihoods, non-linear forward models, and high-dimensional parameter spaces ($D > 10^3$). These require Markov Chain Monte Carlo (**MCMC**) algorithms (Metropolis-Hastings, Hamiltonian Monte Carlo, NUTS) or **Variational Inference** (VI), which introduce their own convergence diagnostics and computational trade-offs.

### The Curse of Dimensionality in Grid Inference
- When evaluating continuous posteriors without conjugacy, our code evaluates likelihoods over discrete grids. While illustrative for 1D or 2D parameter spaces, a $D$-dimensional grid requires $K^D$ evaluations, becoming computationally intractable for $D > 4$.

---

## 6. Machine Learning (6.036)

### Pure NumPy CPU Implementation vs. Tensor Accelerators
- Our feedforward network runs in IEEE 754 float64 on a single CPU thread with full-batch or mini-batch SGD. Production systems (PyTorch, JAX) execute in float32, bfloat16, or FP8 on massively parallel SIMD/GPU tensor cores.
- The forward and backward passes compute matrix multiplications using NumPy's BLAS bindings (`@`), but do not perform operator fusion, kernel compiling (e.g. `torch.compile`), or distributed memory transfers.

### Numerical Gradient Checking Scale
- Finite-difference gradient checking requires evaluating the forward pass $2P$ times, where $P$ is the number of parameters.
- For a small network with $P = 10^3$ weights, gradient checking is fast. For modern networks ($P > 10^8$), finite-difference checking is computationally impossible ($2 \times 10^8$ forward passes per gradient step). Hence, we use numerical gradient checks exclusively as an automated unit test for backpropagation derivations on toy architectures.

---

## 7. Artificial Intelligence (6.034)

### Discrete Grid Search vs. Continuous Robot Kinematics
- Our A* search evaluates 4-connected discrete grids with unit step costs.
- Real autonomous systems operate in continuous state spaces $\mathbb{R}^d$ subject to non-holonomic kinematic and dynamic constraints (curvature, maximum acceleration, jerk). They require algorithms like **Hybrid A\*, Rapidly-exploring Random Trees (RRT\*), or Model Predictive Control (MPC)**.

### Game Tree Search: Minimax vs. MCTS
- Minimax with alpha-beta pruning is optimal for finite, deterministic, zero-sum two-player games with small branching factors (e.g., Connect-Four, small Chess subtrees).
- For large games (Go, Shogi, Full Chess), the game tree size ($b^d \gg 10^{100}$) makes exhaustive minimax intractable even with pruning. Modern systems use **Monte Carlo Tree Search (MCTS)** paired with deep neural networks for state evaluation and move prior guidance.
