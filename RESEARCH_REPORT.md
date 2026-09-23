# Research Report: The Computational Reliability of Mathematical Guarantees
**Author:** Adityajyoti Kar  
**Repository:** [github.com/GiTaDi-CrEaTe/MIT-OCW-Portfolio](https://github.com/GiTaDi-CrEaTe/MIT-OCW-Portfolio)  
**Execution Environment:** Python 3.11 -- 3.13, NumPy, IEEE-754 Float64 ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$)  
**Verification:** Automated CI Matrix, 46 Pytest Unit Tests, 6 Generated Empirical Figures  

---

## Executive Summary

Mathematical proofs in computer science and applied mathematics typically assume exact real arithmetic ($\mathbb{R}$), infinite precision, or strict model adherence. This portfolio investigated the exact failure boundaries of these assumptions across six core MIT OpenCourseWare subjects: Discrete Mathematics (6.042J), Linear Algebra (18.06), Algorithms (6.006), Probabilistic Systems (6.041), Machine Learning (6.036), and Artificial Intelligence (6.034).

Rather than relying on black-box libraries, each algorithm was implemented from first principles in raw Python and NumPy. We subjected each algorithm to controlled stress tests across 14 orders of magnitude of matrix conditioning, varying obstacle density fields, floating-point step sizes, and non-stationary Markov data streams.

---

## Key Experimental Findings

| Experiment | Theoretical Mathematical Guarantee | Physical Machine Failure Mode | Quantitative Finding | Generated Artifact |
|---|---|---|---|---|
| **1. Orthogonal Basis (18.06)** | Classical Gram-Schmidt constructs orthonormal $Q$ ($Q^T Q = I$) | Catastrophic cancellation in projections; loss of orthogonality | At $\kappa(A) \ge 10^8$, CGS $\|Q^TQ - I\|_2 > 0.42$ ($3.01$ on Hilbert), while MGS remains stable ($1.38 \times 10^{-3}$) | [Figure 1](./artifacts/fig1_gram_schmidt_orthogonality.png) |
| **2. SVD Factorization (18.06)** | $A^TA = V \Sigma^2 V^T$ recovers exact singular values $\sigma_i = \sqrt{\lambda_i}$ | Condition squaring: $\kappa(A^TA) = \kappa(A)^2 \ge 1/\epsilon_{\text{mach}}$ wipes out small singular values | At $\kappa(A) = 10^9$, from-scratch $A^TA$ SVD relative error reaches **354.5%**, while LAPACK baseline (`dgesdd`) error is $2.44 \times 10^{-8}$ | [Figure 2](./artifacts/fig2_svd_condition_squaring.png) |
| **3. Heuristic Search (6.034)** | Admissible heuristic $h(n) \le h^*(n)$ guarantees optimal shortest paths | Heuristic overestimation produces suboptimal paths; plateaus expand $O(V)$ nodes | Scaling $h$ by $1.5\times$ yields **63.7% suboptimal paths**; lexicographic tie-breaking cuts expansions by **96.0%** with **0% suboptimality** | [Figure 3](./artifacts/fig3_astar_search_efficiency.png) |
| **4. Gradient Precision (6.036)** | $\lim_{\epsilon \to 0} \frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon} = \nabla f(\theta)$ | Finite-difference step balance: truncation $O(\epsilon^2)$ vs cancellation $O(\epsilon_{\text{mach}}/\epsilon)$ | Precision forms a U-curve: optimal step size is $\epsilon^* \approx 10^{-5}$ ($10^{-9}$ error); at $\epsilon = 10^{-15}$, error reaches **100%** | [Figure 4](./artifacts/fig4_gradient_finite_difference_u_curve.png) |
| **5. Model Misspecification (6.041)** | Posterior distribution contracts onto true parameter $\theta$ ($O(1/N)$ variance) | Static i.i.d. assumptions under Markov drift yield overconfident convergence to wrong mean | Static model credible interval coverage drops to **4.2%** (false certainty); adaptive discount updating restores coverage to **56.9%** | [Figure 5](./artifacts/fig5_model_misspecification.png) |
| **6. Discrete vs Continuous (6.042)** | Algebraic identities hold under group isomorphism in $\mathbb{Z}/n\mathbb{Z}$ | Continuous float64 additions fail associativity and cancellation bounds | 512-bit RSA modular Euler theorem has exact **0.0 error**; float64 $(10^{16} + 1) - 10^{16} - 1$ produces **1.0 error (100% loss)** | [Figure 6](./artifacts/fig6_computational_reliability.png) |

---

## Original Theoretical Synthesis: The Computational Reliability Index (CRI)

To unify these six distinct computational domains, we formulated a normalized metric, the **Computational Reliability Index (CRI)** $\rho \in [0, 1]$:
$$\rho = \frac{1}{1 + \left(\frac{\mathcal{E}}{\mathcal{E}_{\text{crit}}}\right)^2}$$
where $\mathcal{E}$ is the measured empirical deviation and $\mathcal{E}_{\text{crit}}$ is the domain-specific critical failure tolerance. 
- $\rho \to 1.0$: Safe computational regime where hardware execution faithfully implements mathematical theory.
- $\rho = 0.5$: Critical transition boundary.
- $\rho \to 0.0$: Catastrophic failure regime where theoretical guarantees are completely violated by physical limitations.

```
Domain                     | Safe Implementation                    | Safe CRI | Failure Mode                           | Fail CRI
------------------------------------------------------------------------------------------------------------------------------
1. QR Orthogonalization    | Modified Gram-Schmidt (kappa=10^8)     |   1.0000 | Classical Gram-Schmidt (kappa=10^8)    |   0.0006
2. SVD Factorization       | LAPACK Direct SVD (kappa=10^9)         |   1.0000 | A^TA Normal Equations SVD (kappa=10^9) |   0.0008
3. Gradient Verification   | Finite Difference (optimal eps=1e-5)   |   1.0000 | Finite Difference (tiny eps=1e-14)     |   0.0000
4. Bayesian Inference      | Adaptive Updating under Regime Shift   |   0.9615 | Static i.i.d. Updating under Regime Shift |   0.0110
5. Heuristic Search        | A* with Lexicographic Tie-Breaking     |   1.0000 | A* with Inadmissible Heuristic (w=1.5) |   0.0000
6. Algebraic Precision     | Discrete Modular RSA in Z/nZ           |   1.0000 | IEEE-754 float64 Bit Cancellation      |   0.0000
```

---

## Methodological Summary & Reproducibility

1. **First-Principles Implementations:** No high-level frameworks (`scikit-learn`, `PyTorch`, or `sympy`). All decompositions, backpropagation, and search structures built in raw Python/NumPy.
2. **Double-Blind Verification Oracles:** Production libraries (`numpy.linalg`, LAPACK) were used strictly as correctness oracles in automated unit tests.
3. **Statistical Uncertainty & Seeds:** All experiments use deterministic master RNG seeds (`1806, 6042, 6006, 6041, 6036, 6034`). Neural network training was evaluated across 50 independent seeds ($100.0\% \pm 0.0\%$ convergence with Xavier initialization vs $50.0\% \pm 0.0\%$ zero-init collapse).
4. **Automated Reproduction:** Full suite verifiable via `python3 capstone/run_experiments.py` and `pytest -v`.
