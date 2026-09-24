# The Computational Reliability of Mathematical Guarantees: When Theory Meets Finite-Precision Machines

**Author:** Adityajyoti Kar (GitHub: GiTaDi-CrEaTe)  
**Affiliation:** Independent study based on MIT OpenCourseWare (6.042, 18.06, 6.006, 6.041, 6.036, 6.034). Not affiliated with or endorsed by MIT.  
**Execution Environment:** Python 3.11 -- 3.13, NumPy, SciPy, IEEE-754 Float64 ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$)  
**Artifacts & Verification:** 8 Figures in `artifacts/`, 55 Passing Pytest Unit Tests  

---

## 1. Research Question

Can computational breakdown across different mathematical disciplines be characterized and predicted by a domain-independent reliability index?

Mathematical theorems prove properties under assumptions of exact real arithmetic ($\mathbb{R}$), infinite precision, or strict model adherence. When translated into software, these guarantees encounter finite-precision hardware, heuristic approximations, and non-stationary data streams. Does computational failure happen as disconnected domain-specific accidents, or does it follow a predictable transition boundary that can be quantified by a common index?

---

## 2. Hypothesis

Computational failure across numerical, combinatorial, and probabilistic domains can be modeled through four coupled state variables:
1. Numerical Error ($e_{\text{num}}$): Relative discrepancy from exact algebraic or analytical values.
2. Decision Error ($e_{\text{dec}}$): Suboptimality penalty in discrete or combinatorial branching.
3. Assumption Violation ($v_{\text{assump}}$): Degree of divergence from theoretical premises.
4. Stability Risk ($r_{\text{stab}}$): Proximity to numerical or structural breakdown ($1 - m_{\text{stab}}$).

Under a weakest-link formulation, failure occurs when the worst component exceeds its critical tolerance $\tau_k$:
$$z = \max_k \left( \frac{x_k}{\tau_k} \right), \quad \rho = \frac{1}{1 + z^2}$$
where $\rho \in [0, 1]$ is the Computational Reliability Index (CRI).

I chose the rational quadratic curve $\rho = \frac{1}{1 + z^2}$ over the standard logistic sigmoid $\frac{1}{1 + e^z}$ for three mathematical reasons:
1. Exact upper bound: At zero normalized penalty ($z = 0$), $\rho(0) = 1.0$ exactly, representing unbroken theoretical guarantees without requiring an arbitrary offset.
2. Natural transition boundary: When the dominant risk hits its critical tolerance ($z = 1$), $\rho(1) = 0.5$ exactly, creating an unambiguous threshold between acceptable operation and failure.
3. Power-law decay: As $z \to \infty$, $\rho$ decays quadratically as $O(z^{-2})$, matching classical second-order error propagation rather than exponential vanishing, which prematurely underflows float64 values to zero.

I hypothesized that tolerances calibrated on one set of experimental domains (Dataset A) would generalize to predict failure ($\rho < 0.5$) on unseen computational algorithms and external production libraries (Dataset B) with an AUROC exceeding 0.85.

---

## 3. Methods

- **First-Principles Implementations:** I wrote every foundational algorithm (LU decomposition, Classical and Modified Gram-Schmidt QR, QR eigensolvers, normal-equations SVD, AVL trees, BFS/DFS, Dijkstra, Beta-Binomial Bayesian updating, Markov simulation, multi-layer perceptron backpropagation, A* search, Minimax with Alpha-Beta pruning, CSP constraint propagation, and RSA number theory) from scratch in raw Python and NumPy arrays alone.
- **Controlled Stress Testing:** I subjected the algorithms to sweeps across 14 orders of magnitude of matrix condition numbers ($\kappa \in [10^1, 10^{16}]$), finite-difference step sizes ($\epsilon \in [10^{-16}, 10^{-1}]$), varying obstacle densities, and non-stationary Markov drift.
- **Holdout Validation Protocol:** 
  - **Dataset A (Calibration Set, $N=36$):** Derived from the six primary course domains to calibrate tolerances $\tau_k$.
  - **Dataset B (Held-Out Evaluation Set, $N=38$):** Composed of completely unseen algorithms and problem formulations (Cholesky decomposition on ill-conditioned/indefinite systems, Householder reflections on Vandermonde matrices, 4th-order central differences vs complex-step differentiation, weighted A* on deceptive mazes, unpivoted Gaussian elimination, and Markov jump shocks).
- **External Challenge Environment:** Evaluated whether CRI predicts breakdown in production routines (`scipy.linalg.solve`, `scipy.linalg.cholesky`) on ill-conditioned Hilbert matrices ($n = 4 \dots 15$).
- **Statistical Evaluation:** Measured AUROC, F1 score, precision, recall, false-positive rate (FPR), Brier calibration score, and 1,000 bootstrap resamples for 95% confidence intervals.

---

## 4. Experiments

I built eight reproducible experiments to evaluate these boundaries:
1. **QR Orthogonality (18.06):** Controlled condition sweeps comparing Classical vs Modified Gram-Schmidt.
2. **SVD Condition Squaring (18.06):** From-scratch $A^T A$ normal equations vs direct SVD.
3. **A* Search Efficiency (6.034):** Node expansion and suboptimality across obstacle densities and heuristic inflations.
4. **Gradient Precision U-Curve (6.036):** Truncation vs cancellation error in finite-difference gradient checks.
5. **Model Misspecification (6.041):** Static i.i.d. vs adaptive Bayesian updating under parameter drift.
6. **Algebraic Exactness (6.042):** Discrete modular RSA arithmetic vs IEEE-754 float64 cancellation.
7. **Holdout Validation:** Calibrated CRI applied to held-out Dataset B.
8. **External Library Challenge:** Testing CRI against production SciPy and LAPACK solvers.

---

## 5. Results

| Experiment Domain | Theoretical Guarantee | Hardware Failure Mode | Quantitative Finding | Generated Artifact |
|---|---|---|---|---|
| **1. QR Orthogonality** | Gram-Schmidt produces $Q^T Q = I$ | Catastrophic cancellation in projections | CGS $\|Q^T Q - I\|_2 > 0.42$ at $\kappa \ge 10^8$ (3.01 on Hilbert); MGS remains stable ($1.38 \times 10^{-3}$) | [Fig 1](./artifacts/fig1_gram_schmidt_orthogonality.png) |
| **2. SVD Factorization** | $A^T A = V \Sigma^2 V^T$ recovers $\sigma_i = \sqrt{\lambda_i}$ | Condition squaring wipes out $\sigma_{\min}$ | At $\kappa = 10^9$, $A^T A$ SVD error reaches **354.5%**; direct SVD error is $2.44 \times 10^{-8}$ | [Fig 2](./artifacts/fig2_svd_condition_squaring.png) |
| **3. Heuristic Search** | Admissible $h(n) \le h^*(n)$ guarantees optimal path | Overestimation yields suboptimal paths; open space plateaus expand $O(V)$ nodes | $1.5\times$ inflation produces **63.7% suboptimal paths**; lexicographic tie-breaking cuts expansions by **96.0%** with 0% error | [Fig 3](./artifacts/fig3_astar_search_efficiency.png) |
| **4. Gradient Precision** | $\lim_{\epsilon \to 0} \Delta f / (2\epsilon) = \nabla f$ | U-curve: truncation $O(\epsilon^2)$ vs cancellation $O(\epsilon_{\text{mach}}/\epsilon)$ | Optimal step $\epsilon^* \approx 10^{-5}$ ($10^{-9}$ error); at $\epsilon = 10^{-15}$, error hits **100%** | [Fig 4](./artifacts/fig4_gradient_finite_difference_u_curve.png) |
| **5. Bayesian Updating** | Posterior contracts onto true parameter $\theta$ | Static model reports false certainty under parameter drift | Static credible interval coverage drops to **4.2%**; adaptive discount updating restores coverage to **56.9%** | [Fig 5](./artifacts/fig5_model_misspecification.png) |
| **6. Algebraic Precision** | Exact group identities in $\mathbb{Z}/n\mathbb{Z}$ | Float64 addition fails associativity | RSA modular decryption error is **0.0**; float64 $(10^{16} + 1) - 10^{16} - 1$ yields **1.0 (100% loss)** | [Fig 6](./artifacts/fig6_computational_reliability.png) |
| **7. Holdout Validation** | CRI predicts failure on unseen domains | Evaluated on held-out Dataset B ($N=38$) | **AUROC = 0.9494** (95% CI: [0.8472, 1.0000]), **F1 = 0.8966**, Precision = 0.8667, Recall = 0.9286 | [Fig 7](./artifacts/fig7_cri_holdout_validation.png) |
