# 90-Second Demo: When Gram-Schmidt Quietly Fails

A structured walkthrough of the central discovery in this project. No prerequisites beyond basic linear algebra.

## The Theorem (0:00 -- 0:15)

Gram-Schmidt orthogonalization takes a set of linearly independent vectors and produces an orthonormal basis Q where Q^T Q = I (every column is unit length and perpendicular to every other column).

This is proven in every linear algebra textbook. The proof is correct.

## The Implementation (0:15 -- 0:30)

```bash
# Run the experiment
python3 capstone/numerical_stability.py
```

The code implements Classical Gram-Schmidt from scratch (`capstone/numerical_stability.py`, lines 32-54). It follows the textbook algorithm exactly.

On a well-conditioned matrix (condition number 10^4), it works perfectly:
- Orthogonality error: $3.89 \times 10^{-12}$
- Residual ||A - QR||: near machine epsilon

## The Stress Test (0:30 -- 0:50)

Now increase the condition number to 10^8 (still a perfectly valid matrix with linearly independent columns).

Result:
- **Orthogonality error: 0.42** (the Q matrix is NOT orthogonal)
- **Residual ||A - QR||: still near machine epsilon**

The residual says the answer is correct. The orthogonality check says it is completely wrong.

On a 10x10 Hilbert matrix (condition number ~10^13):
- Orthogonality error: **3.01** (the columns have inner products exceeding 0.8)
- The algorithm has silently produced garbage while passing its own quality check

## The Diagnosis (0:50 -- 1:05)

The failure mechanism is **catastrophic cancellation**.

Classical Gram-Schmidt computes:
$$v_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j$$

When $a_k$ is nearly in the span of the previous columns, $v_k$ is the difference of two nearly identical large vectors. In IEEE 754 float64 arithmetic, this difference loses almost all significant bits. The resulting vector is dominated by roundoff noise, which is then normalized to unit length -- introducing a fundamentally wrong direction into the basis.

The key insight: the **residual** ||A - QR|| stays small because the reconstruction only requires that QR approximately equals A, which is a much weaker condition than Q actually being orthogonal. Small backward error does not imply a correct answer.

## The Fix and the Hypothesis (1:05 -- 1:30)

Modified Gram-Schmidt (MGS) projects sequentially against the updated working vector rather than the original columns. This reduces error from $\mathcal{O}(\kappa^2 \epsilon_{mach})$ to $\mathcal{O}(\kappa \epsilon_{mach})$.

On the same Hilbert matrix:
- CGS orthogonality error: 3.01
- MGS orthogonality error: $1.38 \times 10^{-3}$

This pattern -- a sharp transition from reliable to catastrophic computation -- appears in every domain I studied: linear algebra, search, gradient computation, Bayesian inference, and discrete arithmetic.

The Computational Reliability Index (CRI) attempts to characterize this transition boundary across all six domains. Whether it succeeds is tested in the holdout validation (`capstone/cri_validation.py`).

## Reproduce It

```bash
python3 capstone/numerical_stability.py
python3 capstone/run_experiments.py
pytest -v
```

---
*Independent study by Adityajyoti Kar (GiTaDi-CrEaTe). Based on publicly available MIT OpenCourseWare materials. Not affiliated with or endorsed by MIT.*
