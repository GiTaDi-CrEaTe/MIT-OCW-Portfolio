# When Guarantees Meet the Machine
*A Flagship Investigation into the Breakdown of Mathematical Guarantees on Physical Hardware*

---

## The Central Question

In pure mathematics, algorithms are accompanied by unconditional theorems:
- **Gram-Schmidt** constructs an orthonormal basis spanning any set of linearly independent vectors.
- **The SVD theorem** guarantees that any real matrix $A \in \mathbb{R}^{m \times n}$ decomposes as $U \Sigma V^T$.
- **A\* search** with an admissible heuristic is provably optimal.
- **The chain rule** computes exact analytical gradients.
- **Bayes' theorem** updates beliefs rationally under accumulated evidence.

Yet modern computing does not occur in exact arithmetic. It occurs on finite-precision floating-point hardware (IEEE 754 float64 with machine epsilon $\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$), discrete clock cycles, and bounded memory.

This capstone investigates the friction between mathematical proofs and real computers:

$$\text{Pure Mathematical Guarantee} \quad \xrightarrow{\quad\text{IEEE 754 Float64 \& Finite Sampling}\quad} \quad \text{Empirical Computational Reality}$$

---

## Summary of Empirical Investigations

| Investigation | Mathematical Guarantee | Empirical Failure Mode | Root Mechanism | Artifact |
|---|---|---|---|---|
| **1. Gram-Schmidt QR** | Columns of $Q$ are mutually orthogonal ($Q^T Q = I$) | Orthogonality error $\|Q^T Q - I\|_2 > 1.0$ at $\kappa(A) \ge 10^9$ | Classical GS uses corrupted intermediate coordinates; MGS mitigates by sequential updates | [Fig 1](../artifacts/fig1_gram_schmidt_orthogonality.png) |
| **2. SVD via $A^TA$** | Singular values $\sigma_i = \sqrt{\lambda_i(A^TA)}$ | Small singular values obliterated ($\text{rel error} = 100\%$) at $\kappa(A) \ge 10^8$ | Forming $A^TA$ squares the condition number $\kappa(A^TA) = \kappa(A)^2 \ge 1/\epsilon_{\text{mach}}$ | [Fig 2](../artifacts/fig2_svd_condition_squaring.png) |
| **3. A\* Heuristic Search** | Admissible $h(n) \le h^*(n)$ guarantees optimal path | Overestimating $h$ reduces node expansion by 90% but fails optimality in 63.7% of runs | Inadmissible estimates violate the branch-and-bound invariant | [Fig 3](../artifacts/fig3_astar_search_efficiency.png) |
| **4. Gradient Precision** | $\lim_{\epsilon \to 0} \frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon} = \nabla f$ | Error blows up to 100% when $\epsilon \le 10^{-14}$; optimal only near $10^{-5}$ | U-curve: truncation error $O(\epsilon^2)$ vs catastrophic cancellation $O(\epsilon_{\text{mach}}/\epsilon)$ | [Fig 4](../artifacts/fig4_gradient_finite_difference_u_curve.png) |
| **5. Model Misspecification** | Bayesian posterior concentrates on true state parameter | 95% Credible Interval covers true state only 4.2% of the time | Static i.i.d. assumption produces false certainty on non-stationary Markov data | [Fig 5](../artifacts/fig5_model_misspecification.png) |
| **6. Cross-Course Synthesis** | Six isolated subjects form a pipeline | Discrete exactness (6.042) vs continuous numerical drift (18.06/6.036) | Integer rings $\mathbb{Z}/n\mathbb{Z}$ have no roundoff; float64 $(10^{16}+1)-10^{16} = 0$ | Code |

---

## Detailed Experimental Findings

### Experiment 1 --  Numerical Stability: Classical vs. Modified Gram-Schmidt
- **Code:** [`numerical_stability.py`](./numerical_stability.py)
- **Artifact:** [`fig1_gram_schmidt_orthogonality.png`](../artifacts/fig1_gram_schmidt_orthogonality.png)

#### The Experiment
We generated matrices $A \in \mathbb{R}^{30 \times 15}$ with condition numbers systematically swept from $\kappa(A) = 10^1$ to $10^{14}$ using SVD-controlled spectra. We also evaluated a 10×10 Hilbert matrix ($H_{i,j} = \frac{1}{i+j-1}$, $\kappa(H) \approx 1.6 \times 10^{13}$).

#### Quantitative Results
| Condition $\kappa(A)$ | CGS Orthogonality Loss $\|Q^TQ - I\|_2$ | MGS Orthogonality Loss $\|Q^TQ - I\|_2$ | Ratio CGS / MGS |
|---|---|---|---|
| $10^1$ | $2.01 \times 10^{-15}$ | $1.42 \times 10^{-15}$ | 1.4 |
| $10^4$ | $6.00 \times 10^{-10}$ | $5.44 \times 10^{-13}$ | $1.1 \times 10^3$ |
| $10^6$ | $3.10 \times 10^{-6}$ | $8.77 \times 10^{-11}$ | $3.5 \times 10^4$ |
| $10^8$ | $4.17 \times 10^{-1}$ | $5.22 \times 10^{-9}$ | $8.0 \times 10^7$ |
| $10^{10}$ | $2.78 \times 10^{0}$ | $8.33 \times 10^{-7}$ | $3.3 \times 10^6$ |
| $10^{14}$ | $5.49 \times 10^{0}$ | $1.63 \times 10^{-3}$ | $3.4 \times 10^3$ |
| **Hilbert Matrix (n=10)** | **3.00** | **$1.80 \times 10^{-4}$** | **$1.6 \times 10^4$** |

#### Key Insight
In pure mathematics, CGS and MGS compute identical projections. In double precision, CGS computes projection coefficients using the original, un-orthogonalized column, triggering catastrophic cancellation when columns are nearly collinear. By $\kappa(A) = 10^8$, CGS has completely lost orthogonality ($\|Q^TQ - I\| \approx 0.42$), while the reconstruction residual $\|A - QR\|$ remains small ($\sim 10^{-17}$). This reveals a crucial lesson: **a near-zero residual does NOT imply an orthogonal basis.**

---

### Experiment 2 --  SVD via $A^TA$ vs. Direct Bidiagonalization
- **Code:** [`svd_investigation.py`](./svd_investigation.py)
- **Artifact:** [`fig2_svd_condition_squaring.png`](../artifacts/fig2_svd_condition_squaring.png)

#### The Experiment
The textbook SVD builds the symmetric matrix $A^TA$, computes its eigenvalues $\lambda_i$, and derives $\sigma_i = \sqrt{\lambda_i}$. We compared this from-scratch construction against direct bidiagonalization SVD across $\kappa(A) \in [10^1, 10^{12}]$.

#### Quantitative Results
| $\kappa(A)$ | True $\sigma_{\text{min}}$ | $A^TA$ Estimated $\hat{\sigma}_{\text{min}}$ | $A^TA$ Relative Error | Direct SVD Relative Error |
|---|---|---|---|---|
| $10^2$ | $1.00 \times 10^{-2}$ | $1.00 \times 10^{-2}$ | $2.37 \times 10^{-13}$ | $1.21 \times 10^{-15}$ |
| $10^6$ | $1.00 \times 10^{-6}$ | $1.00 \times 10^{-6}$ | $9.40 \times 10^{-6}$ | $2.05 \times 10^{-11}$ |
| $10^7$ | $1.00 \times 10^{-7}$ | $9.97 \times 10^{-8}$ | $2.52 \times 10^{-3}$ | $1.21 \times 10^{-10}$ |
| $10^8$ | $1.00 \times 10^{-8}$ | $1.27 \times 10^{-8}$ | **27.0%** | $2.30 \times 10^{-9}$ |
| $10^9$ | $1.00 \times 10^{-9}$ | **0.00000** | **100.0% (Zeroed)** | $2.44 \times 10^{-8}$ |
| $10^{12}$ | $1.00 \times 10^{-12}$ | **0.00000** | **100.0% (Zeroed)** | $1.77 \times 10^{-5}$ |

#### Key Insight
$$\kappa(A^TA) = \left(\frac{\sigma_1}{\sigma_n}\right)^2 = \kappa(A)^2$$
When $\kappa(A) \ge 10^8$, $\kappa(A^TA) \ge 10^{16} \approx 1/\epsilon_{\text{mach}}$. At that point, the smallest eigenvalues of $A^TA$ fall below the floating-point roundoff floor and are rounded to zero. Computing $u_i = \frac{1}{\sigma_i} A v_i$ then involves dividing by zero or by pure noise. Direct bidiagonalization avoids forming $A^TA$ entirely, preserving accuracy down to $\epsilon_{\text{mach}}$.

---

### Experiment 3 --  A\* Heuristic Search Scaling & Optimality Limits
- **Code:** [`search_efficiency.py`](./search_efficiency.py)
- **Artifact:** [`fig3_astar_search_efficiency.png`](../artifacts/fig3_astar_search_efficiency.png)

#### The Experiment
We benchmarked Dijkstra, Euclidean A\*, Manhattan A\*, Manhattan with Tie-Breaking, and an Inadmissible Heuristic ($1.5 \times h_M$) on random obstacle grids from size 30×30 to 50×50 across obstacle densities $0.0 \le \rho \le 0.25$ over 80 independent runs.

#### Quantitative Results (50×50 Grid)
| Density $\rho$ | Dijkstra Nodes | A\* Euclidean | A\* Manhattan | A\* Tie-Break | Inadmissible (1.5x) | Tie-Break Savings |
|---|---|---|---|---|---|---|
| 0.00 (Open) | 2500.0 | 2500.0 | 2500.0 | **99.0** | 99.0 | **96.0%** |
| 0.10 | 2234.3 | 2178.3 | 1920.3 | **248.5** | 129.3 | **88.9%** |
| 0.20 | 1962.7 | 1754.0 | 1223.9 | **287.6** | 151.2 | **85.3%** |
| 0.25 | 1837.7 | 1539.4 | 816.7 | **248.5** | 168.4 | **86.5%** |

#### Optimality Verification
- **A\* (Euclidean):** 0 / 80 suboptimal paths (**0.0% failure**)
- **A\* (Manhattan):** 0 / 80 suboptimal paths (**0.0% failure**)
- **A\* (Tie-Break):** 0 / 80 suboptimal paths (**0.0% failure**)
- **A\* (Inadmissible 1.5x):** **51 / 80 suboptimal paths (63.7% failure rate)**

#### Key Insight
Admissibility is not a minor guideline; it is a razor-thin boundary. Scaling the heuristic by just 1.5× cuts search nodes in half, but produces suboptimal paths on nearly two-thirds of all runs. Furthermore, on open grids without tie-breaking, standard A\* expands all 2500 nodes because $f(n) = g(n) + h(n)$ is constant along diagonal wavefronts. A $10^{-4}$ tie-breaking bias breaks plateaus and achieves a **96% node reduction** while preserving 100% path optimality.

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
The static model's posterior variance shrinks as $O(1/N)$. Because it assumes the world is static, it accumulates **false certainty**: it converges with tight bounds around the blended long-run average ($\sim 0.525$), completely blind to the fact that the coin is currently in the 0.85 state! This connects probability theory directly to the dangers of overconfident machine learning deployment.

---

### Experiment 6 --  Cross-Course Synthesis: Connecting the Six Disciplines
- **Code:** [`cross_course_synthesis.py`](./cross_course_synthesis.py)

1. **6.042 (Discrete Exactness) $\to$ 18.06 (Continuous Breakdown):**
 In 6.042 integer arithmetic ($\mathbb{Z}/n\mathbb{Z}$), Euler's theorem $(m^e)^d \equiv m \pmod n$ holds with exactness for 512-bit numbers with zero roundoff. In float64 arithmetic, $(10^{16} + 1.0) - 10^{16} - 1.0 = -1.0$ (complete loss of the additive unit).
2. **18.06 (Eigenstructure) $\leftrightarrow$ 6.041 (Markov Chains):**
 The stationary distribution of a 4-state Markov transition matrix is computed as the left eigenvector of $P$ using 18.06 matrix routines. The spectral vector matched a 200,000-step 6.041 simulation to within $0.0039$.
3. **6.041 (MLE) + 18.06 (Matrix Calculus) $\to$ 6.036 (Neural Networks):**
 The cross-entropy loss is the negative log-likelihood of a Bernoulli model. The combined gradient $\frac{\partial L}{\partial Z} = A - Y$ algebraically cancels the denominator $A(1-A)$, avoiding floating-point division-by-zero when activations saturate.

---

## How to Replicate

All code is self-contained and reproducible. To run the full verification suite and regenerate all figures:

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
