# 90-Second Demo: When Gram-Schmidt Quietly Fails

A structured walkthrough of the central discovery in this project. No prerequisites beyond basic linear algebra.

## The Theorem (0:00 -- 0:15)

Gram-Schmidt orthogonalization takes a set of linearly independent vectors and produces an orthonormal basis Q where Q^T Q = I (every column is unit length and perpendicular to every other column).

This is proven in every linear algebra textbook. The proof is correct.

## The Implementation (0:15 -- 0:30)

```bash
# Run the experiment
python3 capstone/numerical_stability.py
