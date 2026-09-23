# Foundations Lab  --  Capstone Experiments
*When Mathematical Guarantees Meet Real Computers*

---

## Overview

This directory contains the empirical foundation of the portfolio: six controlled, reproducible experiments that test where pure mathematical theory diverges from computer implementation.

All six experiments can be executed and their visual artifacts regenerated with a single command:
```bash
python3 capstone/run_experiments.py
```

All generated publication-quality figures are saved directly to [`../artifacts/`](../artifacts/).

---

## The Six Investigations

### Experiment 1 --  Numerical Stability: Classical vs. Modified Gram-Schmidt
- **Code:** [`numerical_stability.py`](./numerical_stability.py)
- **Artifact:** [`fig1_gram_schmidt_orthogonality.png`](../artifacts/fig1_gram_schmidt_orthogonality.png)

#### The Experiment
We generated $30 \times 15$ matrices with condition numbers $\kappa(A)$ controlled across 14 orders of magnitude ($10^1$ to $10^{14}$) using exact singular value synthesis ($A = U \Sigma V^T$). We also benchmarked a $10 \times 10$ Hilbert matrix ($\kappa \approx 1.60 \times 10^{13}$).

#### Quantitative Results
| Condition $\kappa(A)$ | CGS Orthogonality Loss $\|Q^T Q - I\|_2$ | MGS Orthogonality Loss $\|Q^T Q - I\|_2$ | Error Ratio CGS / MGS |
|---|---|---|---|
| $10^1$ | $1.72 \times 10^{-15}$ | $1.41 \times 10^{-15}$ | $1.22 \times$ |
| $10^4$ | $3.89 \times 10^{-12}$ | $4.10 \times 10^{-13}$ | $9.49 \times$ |
| $10^6$ | $4.21 \times 10^{-8}$ | $2.14 \times 10^{-11}$ | $1,967 \times$ |
| **$10^8$** | **0.4217** (Complete Breakdown) | **$1.85 \times 10^{-9}$** | **$2.28 \times 10^8 \times$** |
| $10^{10}$ | **1.0000** (Totally Degenerate) | $1.76 \times 10^{-7}$ | $5.68 \times 10^6 \times$ |
| $10^{14}$ | **1.0000** | $1.82 \times 10^{-3}$ | $5.49 \times 10^2 \times$ |
| **Hilbert ($n=10$)** | **3.0125** | **$1.38 \times 10^{-3}$** | **$2,182 \times$** |

#### Key Insight
In exact arithmetic, CGS and MGS are mathematically identical. On floating-point hardware (IEEE 754 float64, $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), CGS suffers from catastrophic cancellation error scaling as $O(\epsilon_{\text{mach}} \kappa(A)^2)$, completely losing orthogonality around $\kappa(A) \approx 10^8$. MGS projects sequentially onto updated coordinates, degrading as $O(\epsilon_{\text{mach}} \kappa(A))$ -- maintaining usable basis vectors even on notoriously ill-conditioned Hilbert matrices.

---

### Experiment 2 --  From-Scratch SVD via $A^TA$ vs. LAPACK Baseline
- **Code:** [`svd_investigation.py`](./svd_investigation.py)
- **Artifact:** [`fig2_svd_condition_squaring.png`](../artifacts/fig2_svd_condition_squaring.png)

#### The Experiment
The textbook SVD builds the symmetric normal matrix $A^TA$, computes its eigenvalues $\lambda_i$, and derives singular values as $\sigma_i = \sqrt{\lambda_i}$. We compared this from-scratch textbook construction against the production direct LAPACK SVD baseline (`dgesdd`, using Golub-Kahan bidiagonalization and divide-and-conquer) across $\kappa(A) \in [10^1, 10^{12}]$.

#### Quantitative Results
| $\kappa(A)$ | True $\sigma_{\text{min}}$ | $A^TA$ Estimated $\hat{\sigma}_{\text{min}}$ | $A^TA$ Relative Error | LAPACK Baseline Relative Error |
|---|---|---|---|---|
| $10^2$ | $1.00 \times 10^{-2}$ | $1.00 \times 10^{-2}$ | $2.37 \times 10^{-13}$ | $1.21 \times 10^{-15}$ |
| $10^6$ | $1.00 \times 10^{-6}$ | $1.00 \times 10^{-6}$ | $9.40 \times 10^{-6}$ | $2.05 \times 10^{-11}$ |
| $10^7$ | $1.00 \times 10^{-7}$ | $9.97 \times 10^{-8}$ | $2.52 \times 10^{-3}$ | $1.21 \times 10^{-10}$ |
| $10^8$ | $1.00 \times 10^{-8}$ | $1.27 \times 10^{-8}$ | **27.0%** | $2.30 \times 10^{-9}$ |
| $10^9$ | $1.00 \times 10^{-9}$ | $4.55 \times 10^{-9}$ | **354.5% (Swamped)** | $2.44 \times 10^{-8}$ |
| $10^{12}$ | $1.00 \times 10^{-12}$ | **0.00000** | **100.0% (Zeroed)** | $1.77 \times 10^{-5}$ |

#### Key Insight
$$\kappa(A^TA) = \left(\frac{\sigma_1}{\sigma_n}\right)^2 = \kappa(A)^2$$
When $\kappa(A) \ge 10^8$, $\kappa(A^TA) \ge 10^{16} \approx 1/\epsilon_{\text{mach}}$. At that point, the smallest singular values square to numbers below the floating-point roundoff floor and are swamped or zeroed. Computing $u_i = \frac{1}{\sigma_i} A v_i$ then involves dividing by pure noise. Direct bidiagonalization avoids forming $A^TA$ entirely, preserving accuracy down to $\epsilon_{\text{mach}}$.

---

### Experiment 3 --  A\* Heuristic Search Scaling & Optimality Limits
- **Code:** [`search_efficiency.py`](./search_efficiency.py)
- **Artifact:** [`fig3_astar_search_efficiency.png`](../artifacts/fig3_astar_search_efficiency.png)

#### The Experiment
We benchmarked Dijkstra, Euclidean A\*, Manhattan A\*, Manhattan with Lexicographic Tie-Breaking, and an Inadmissible Heuristic ($1.5 \times h_M$) on random obstacle grids from size 30×30 to 50×50 across obstacle densities $0.0 \le \rho \le 0.25$ over 80 independent runs.

#### Quantitative Results (50×50 Grid)
| Density $\rho$ | Dijkstra Nodes | A\* Euclidean | A\* Manhattan | A\* Lexicographic Tie-Break | Inadmissible (1.5x) | Tie-Break Savings |
|---|---|---|---|---|---|---|
| 0.00 (Open) | 2500.0 | 2500.0 | 2500.0 | **99.0** | 99.0 | **96.0%** |
| 0.10 | 2234.3 | 2178.3 | 1920.3 | **248.5** | 129.3 | **88.9%** |
| 0.20 | 1962.7 | 1754.0 | 1223.9 | **287.6** | 151.2 | **85.3%** |
| 0.25 | 1837.7 | 1539.4 | 816.7 | **248.5** | 168.4 | **86.5%** |

#### Optimality Verification
- **A\* (Euclidean):** 0 / 80 suboptimal paths (**0.0% failure**)
- **A\* (Manhattan):** 0 / 80 suboptimal paths (**0.0% failure**)
- **A\* (Lexicographic Tie-Break):** 0 / 80 suboptimal paths (**0.0% failure, strictly guaranteed**)
- **A\* (Inadmissible 1.5x):** **51 / 80 suboptimal paths (63.7% failure rate)**

#### Key Insight
Admissibility is not a minor guideline; it is a razor-thin boundary. Scaling the heuristic by just 1.5× cuts search nodes in half, but produces suboptimal paths on nearly two-thirds of all runs (63.7% failure rate). Furthermore, on open grids without tie-breaking, standard A\* expands all 2500 nodes because $f(n) = g(n) + h(n)$ is constant along diagonal wavefronts. Rather than scaling $h$ by an ad-hoc factor $(1 + \epsilon)$ which can exceed $h^*(n)$ and violate admissibility, true lexicographic tie-breaking keeps $f = g + h$ strictly unscaled and resolves equal-$f$ ties by preferring states with smaller remaining $h$. This preserves the mathematical admissibility theorem $h(n) \le h^*(n)$ unconditionally while achieving a **96% node reduction** on open grids.

---

### Experiment 4 --  Finite-Difference Precision U-Curve & Multi-Seed ML Stability
- **Code:** [`gradient_precision.py`](./gradient_precision.py)
- **Artifact:** [`fig4_gradient_finite_difference_u_curve.png`](../artifacts/fig4_gradient_finite_difference_u_curve.png)

#### The Experiment
1. **Precision Sweep:** We evaluated centered finite differences $\frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon}$ against hand-derived backpropagation gradients as $\epsilon$ swept from $10^{-1}$ to $10^{-15}$.
2. **50-Seed Statistical Training:** We trained a 2-hidden-layer network on the non-linear concentric two-rings dataset across 50 independent random seeds under three initialization schemes.

#### Quantitative Results (Finite Difference)
| Epsilon $\epsilon$ | Relative Error vs Analytical Gradient | Dominant Error Regime |
|---|---|---|
| $10^{-1}$ | $2.91 \times 10^{-3}$ | Truncation error $O(\epsilon^2)$ |
| $10^{-3}$ | $2.92 \times 10^{-7}$ | Truncation error $O(\epsilon^2)$ |
| **$10^{-5}$** | **$2.79 \times 10^{-9}$** | **Optimal Precision Balance ($\epsilon^* \approx \epsilon_{\text{mach}}^{1/3}$)** |
| $10^{-9}$ | $1.04 \times 10^{-6}$ | Cancellation error beginning |
| $10^{-11}$ | $4.05 \times 10^{-3}$ | Catastrophic cancellation $O(\epsilon_{\text{mach}}/\epsilon)$ |
| $10^{-13}$ | $1.20 \times 10^{-1}$ (12.0%) | Catastrophic cancellation |
| $10^{-15}$ | **1.0000 (100.0%)** | Complete loss of precision ($\text{fl}(f_+ - f_-) = 0$) |

#### Multi-Seed Training Statistics (50 Seeds)
- **Xavier/He Normal:** Mean Accuracy = **$100.0\% \pm 0.00\%$** (50 / 50 converged to perfect separation)
- **Zero Initialization:** Mean Accuracy = **$50.00\% \pm 0.00\%$** (Zero variance; all hidden units mirror each other)
- **Oversized ($\sigma=5.0$):** Mean Accuracy = **$89.43\% \pm 7.85\%$** (Min: 71.0%, Max: 99.7%; tanh activation saturation)

---

### Experiment 5 --  Probability Under Model Misspecification
- **Code:** [`model_misspecification.py`](./model_misspecification.py)
- **Artifact:** [`fig5_model_misspecification.png`](../artifacts/fig5_model_misspecification.png)

#### The Experiment
We fed a stream of coin flips generated by a 2-state Markov regime-switching coin ($\theta_0 = 0.20, \theta_1 = 0.85$, switch rate 3%) into a textbook Bayesian conjugate Beta-Binomial updating model that assumed i.i.d. observations.

#### Quantitative Results
- **Misspecified Static i.i.d. Model:**
  - Mean Squared Error: **0.1049**
  - **Empirical Coverage of Nominal 95% Credible Interval: 4.2%**
  - Final Posterior Standard Deviation: $1.58 \times 10^{-2}$
- **Adaptive (Regime-Aware) Model:**
  - Mean Squared Error: **0.0579**
  - **Empirical Coverage of Nominal 95% Credible Interval: 56.9%**
  - Final Posterior Standard Deviation: $9.49 \times 10^{-2}$

#### Key Insight
The static model's posterior variance shrinks as $O(1/N)$. Because it assumes the world is static, it accumulates **false certainty**: it converges with tight bounds around the blended long-run average ($\sim 0.525$), completely blind to the fact that the coin is currently in the 0.85 state. This connects probability theory directly to the dangers of overconfident machine learning deployment.

---

### Experiment 6 --  Cross-Course Synthesis & The Computational Reliability Index (CRI)
- **Code:** [`cross_course_synthesis.py`](./cross_course_synthesis.py)
- **Artifact:** [`fig6_computational_reliability.png`](../artifacts/fig6_computational_reliability.png)

#### 1. Theoretical Connections
1. **6.042 (Discrete Exactness) $\to$ 18.06 (Continuous Breakdown):**
   In 6.042 integer arithmetic ($\mathbb{Z}/n\mathbb{Z}$), Euler's totient theorem $(m^e)^d \equiv m \pmod n$ was verified using genuine primes $p = 1000000007$ and $q = 1000000009$ with private exponent $d = e^{-1} \pmod{\phi(n)}$, recovering $m$ with exact 0.0 reconstruction error. In float64 arithmetic, $(10^{16} + 1.0) - 10^{16} - 1.0 = 1.0$ (complete catastrophic cancellation of the unit).
2. **18.06 (Eigenstructure) $\leftrightarrow$ 6.041 (Markov Chains):**
   The stationary distribution of a 4-state Markov transition matrix is computed as the left eigenvector of $P$ using 18.06 matrix routines. The spectral vector matched a 200,000-step 6.041 simulation to within $0.0039$.
3. **6.041 (MLE) + 18.06 (Matrix Calculus) $\to$ 6.036 (Neural Networks):**
   The cross-entropy loss is the negative log-likelihood of a Bernoulli model. The combined gradient $\frac{\partial L}{\partial Z} = A - Y$ algebraically cancels the denominator $A(1-A)$, avoiding floating-point division-by-zero when activations saturate.

#### 2. Original Synthesis Framework: The Computational Reliability Index (CRI)
We formulate a unified safety metric $\rho \in [0, 1]$ mapping empirical degradation against critical failure tolerances across all six computational domains:
$$\rho = \frac{1}{1 + \left(\frac{\text{Error}}{\text{Error}_{\text{crit}}}\right)^2}$$
where $\rho \to 1.0$ indicates that machine execution faithfully reflects mathematical guarantees, $\rho = 0.5$ marks the critical transition threshold, and $\rho \to 0.0$ signifies catastrophic breakdown.

#### CRI Profile Across All Six Disciplines
| Domain | Safe Implementation | Safe CRI | Failure Mode | Fail CRI |
|---|---|---|---|---|
| **1. QR Orthogonalization** | Modified Gram-Schmidt ($\kappa=10^8$) | **1.0000** | Classical Gram-Schmidt ($\kappa=10^8$) | **0.0006** |
| **2. SVD Factorization** | LAPACK Direct SVD ($\kappa=10^9$) | **1.0000** | $A^TA$ Normal Equations ($\kappa=10^9$) | **0.0008** |
| **3. Gradient Verification** | Finite Difference ($\epsilon^* = 10^{-5}$) | **1.0000** | Finite Difference ($\epsilon = 10^{-14}$) | **0.0000** |
| **4. Bayesian Inference** | Adaptive Updating (Regime Shift) | **0.9615** | Static i.i.d. Updating (Regime Shift) | **0.0110** |
| **5. Heuristic Search** | A\* with Lexicographic Tie-Break | **1.0000** | A\* with Inadmissible Heuristic ($1.5\times$) | **0.0000** |
| **6. Algebraic Precision** | Discrete Modular RSA in $\mathbb{Z}/n\mathbb{Z}$ | **1.0000** | IEEE-754 float64 Bit Cancellation | **0.0000** |

---

## How to Replicate

All code is self-contained and reproducible. To run the full verification suite and regenerate all six figures:

```bash
python3 capstone/run_experiments.py
```

To run individual investigations:
```bash
python3 capstone/numerical_stability.py
python3 capstone/svd_investigation.py
python3 capstone/search_efficiency.py
python3 capstone/gradient_precision.py
python3 capstone/model_misspecification.py
python3 capstone/cross_course_synthesis.py
```
