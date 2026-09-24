# Foundations Lab  --  Capstone Experiments
*When Mathematical Guarantees Meet Real Computers*

---

## Overview

This directory contains eight controlled, reproducible experiments that test where textbook mathematical guarantees break down on physical hardware.

All eight experiments can be executed and their figures regenerated with a single command:
```bash
python3 capstone/run_experiments.py
```

All generated figures are saved directly to [`../artifacts/`](../artifacts/).

---

## The Eight Investigations

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

