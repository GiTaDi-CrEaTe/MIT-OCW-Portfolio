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
